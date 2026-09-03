"""
종목 선별이 예측력을 갖는가 — 나쁜 종목을 빼면 앞으로 나아지는가.

    python scripts/symbol_selection.py

과거 성적이 나쁜 종목을 빼고 싶어지는 것은 자연스럽다. 그런데 그 결정이
정당하려면 **종목의 과거 성적이 미래 성적을 예측해야** 한다. 예측하지 못하면
종목 제외는 성과 개선이 아니라 표본 안에서 최악을 지우는 사후 선택일 뿐이다.
BNB 를 최고 종목으로 착각했던 것과 정확히 같은 형태의 오류다.

여기서 재는 것
  1) 전반부 순위 → 후반부 순위 상관 (예측력 그 자체)
  2) 창 단위 자기상관 — 좋았던 종목이 다음 창에서도 좋은가
  3) leave-one-out — 종목 하나를 빼면 통합 통계가 어떻게 변하는가
  4) 실제 규칙 시뮬레이션 — "직전 창 최악 종목을 다음 창에서 뺀다"를 돌려본다
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                                   # noqa: E402
from src.validation.stats import t_test                              # noqa: E402
from src.validation.suite import SuiteConfig, _costs, _params, load_symbols  # noqa: E402
from src.backtest import WeightParams, walk_forward                  # noqa: E402
from src.strategies import build_all                                 # noqa: E402

pd.set_option("display.width", 200)
CACHE = ROOT / "runs" / "symbol_selection_trades.parquet"


def collect_trades(cfg, symbols: list[str]) -> pd.DataFrame:
    if CACHE.exists():
        print(f"거래 캐시 사용: {CACHE.name}")
        return pd.read_parquet(CACHE)

    sc = SuiteConfig(symbols=symbols, run_permutation=False,
                     run_sensitivity=False, run_efficiency=False)
    names = cfg.get("strategies")
    strategies = [s for s in build_all() if not names or s.name in set(names)]
    params, costs = _params(cfg), _costs(cfg, sc)
    wp = WeightParams()

    print("데이터 로드 후 walk-forward (몇 분 걸립니다)")
    frames = []
    for d in load_symbols(cfg, sc, print):
        wf = walk_forward(d.bars, d.feats, d.regimes, strategies, cfg,
                          interval=sc.interval, minute_df=d.minute, symbol=d.symbol,
                          weight_params=wp, exec_params=params, costs=costs)
        if not wf.oos_trades.empty:
            t = wf.oos_trades.copy()
            t["symbol"] = d.symbol
            frames.append(t)
        print(f"  {d.symbol:<9} 거래 {wf.metrics.trades:>4}  "
              f"기대R {wf.oos_trades['r_multiple'].mean():+.4f}")
    trades = pd.concat(frames, ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(CACHE)
    return trades


def main() -> int:
    cfg = load_config()
    symbols = list(cfg.require("universe"))
    tr = collect_trades(cfg, symbols)
    # parquet 왕복 후 object 로 돌아오는 경우가 있다
    tr["entry_time"] = pd.to_datetime(tr["entry_time"], utc=True)
    tr = tr.sort_values("entry_time").reset_index(drop=True)
    print(f"\n총 {len(tr):,} 거래 · {tr['entry_time'].min():%Y-%m} ~ {tr['entry_time'].max():%Y-%m}\n")

    # ── 1) 전반부 → 후반부 순위 예측력
    mid = tr["entry_time"].quantile(0.5)
    first = tr[tr["entry_time"] < mid].groupby("symbol")["r_multiple"].mean()
    second = tr[tr["entry_time"] >= mid].groupby("symbol")["r_multiple"].mean()
    both = pd.DataFrame({"전반부": first, "후반부": second}).dropna()
    both["전반부순위"] = both["전반부"].rank(ascending=False).astype(int)
    both["후반부순위"] = both["후반부"].rank(ascending=False).astype(int)

    print("=" * 74)
    print(f"1) 전반부 성적이 후반부를 예측하는가  (분기점 {mid:%Y-%m})")
    print("=" * 74)
    print(both.round(4).sort_values("전반부순위").to_string())
    rho = both["전반부"].corr(both["후반부"], method="spearman")
    pear = both["전반부"].corr(both["후반부"])
    print(f"\n  순위상관(Spearman) = {rho:+.3f}   값상관(Pearson) = {pear:+.3f}")
    n = len(both)
    # 순위상관의 귀무분포는 순열로 낸다 (n=8 이라 근사식보다 정확하다)
    rng = np.random.default_rng(11)
    a = both["전반부"].to_numpy()
    b = both["후반부"].to_numpy()
    null = np.array([pd.Series(a).corr(pd.Series(rng.permutation(b)), method="spearman")
                     for _ in range(20_000)])
    p = float((np.abs(null) >= abs(rho)).mean())
    print(f"  순열 p = {p:.3f} → "
          f"{'예측력 있음' if p < 0.05 else '예측력을 확인할 수 없다'}")

    # ── 2) 창 단위 자기상관
    print("\n" + "=" * 74)
    print("2) 창 단위 — 이번 창에 좋았던 종목이 다음 창에도 좋은가")
    print("=" * 74)
    tr["창"] = tr["entry_time"].dt.to_period("Q")
    piv = tr.pivot_table(index="창", columns="symbol", values="r_multiple", aggfunc="mean")
    pairs = []
    for i in range(len(piv) - 1):
        cur, nxt = piv.iloc[i], piv.iloc[i + 1]
        ok = cur.notna() & nxt.notna()
        if ok.sum() >= 4:
            pairs.append((cur[ok].rank(), nxt[ok].rank()))
    if pairs:
        rs = [float(c.corr(n_, method="spearman")) for c, n_ in pairs]
        rs = [r for r in rs if np.isfinite(r)]
        m = float(np.mean(rs))
        se = float(np.std(rs, ddof=1) / np.sqrt(len(rs)))
        print(f"  분기 {len(rs)}쌍 · 평균 순위상관 {m:+.3f} (표준오차 {se:.3f}, "
              f"t={m/se if se else float('nan'):+.2f})")
        print(f"  → {'지속성 있음' if abs(m) > 2*se else '지속성 없음 — 이번 분기 순위는 다음 분기와 무관하다'}")

    # ── 3) leave-one-out
    print("\n" + "=" * 74)
    print("3) 종목 하나를 빼면 통합 통계가 어떻게 되는가")
    print("=" * 74)
    base = t_test(tr["r_multiple"].to_numpy())
    print(f"  전체 8종목: 기대R {base.mean:+.4f}  t={base.t:+.2f}  p={base.p:.3f}  "
          f"거래 {base.n:,}")
    rows = []
    for s in symbols:
        sub = tr[tr["symbol"] != s]["r_multiple"].to_numpy()
        r = t_test(sub)
        rows.append({"제외종목": s, "남은거래": r.n, "기대R": round(r.mean, 4),
                     "t": round(r.t, 2), "p": round(r.p, 3),
                     "t변화": round(r.t - base.t, 2)})
    loo = pd.DataFrame(rows).sort_values("t", ascending=False)
    print(loo.to_string(index=False))

    # ── 4) 실제로 쓸 수 있는 규칙인가
    print("\n" + "=" * 74)
    print("4) 실행 가능한 규칙 — '직전 분기 최악 종목을 다음 분기에 뺀다'")
    print("=" * 74)
    kept, dropped = [], []
    qs = list(piv.index)
    for i in range(1, len(qs)):
        prev, cur_q = piv.loc[qs[i - 1]], qs[i]
        if prev.notna().sum() < 4:
            continue
        worst = prev.idxmin()
        blk = tr[tr["창"] == cur_q]
        dropped.append(blk[blk["symbol"] == worst]["r_multiple"])
        kept.append(blk[blk["symbol"] != worst]["r_multiple"])
    kept_r = pd.concat(kept).to_numpy() if kept else np.array([])
    drop_r = pd.concat(dropped).to_numpy() if dropped else np.array([])
    if len(kept_r) and len(drop_r):
        k, dd = t_test(kept_r), t_test(drop_r)
        print(f"  규칙대로 남긴 거래 : 기대R {k.mean:+.4f}  t={k.t:+.2f}  n={k.n:,}")
        print(f"  규칙대로 뺀 거래   : 기대R {dd.mean:+.4f}  t={dd.t:+.2f}  n={dd.n:,}")
        print(f"  차이               : {k.mean - dd.mean:+.4f}R "
              f"({'규칙이 도움' if k.mean > dd.mean else '규칙이 역효과 — 뺀 쪽이 더 좋았다'})")
        print(f"  아무것도 안 뺐을 때: 기대R {base.mean:+.4f}")
        print(f"  → 규칙 적용 후 개선폭 {k.mean - base.mean:+.4f}R")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
