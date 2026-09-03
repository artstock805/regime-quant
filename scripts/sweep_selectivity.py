"""
선별도 스윕 — 신호를 골라 받으면 나아지는가.

    python scripts/sweep_selectivity.py                 # 1단계만 (빠름)
    python scripts/sweep_selectivity.py --confirm        # 2단계까지 (느림)

근거: 메이커 수수료 실험에서 수수료를 낮춰 진입이 늘자(BTC 99→131) 전체 기대값이
희석됐다. 한계 거래가 평균보다 나쁘다는 뜻이므로, 반대로 문턱을 올려 상위 신호만
받으면 나아질 수 있다.

두 단계로 본다.
  1단계 (공짜)  walk-forward 를 한 번만 돌리고, 기록된 거래별 확신도로 층을 나눠
                기대R을 본다. 확신도와 기대R이 단조 관계면 선별도가 유효하다.
  2단계 (확인)  1단계에서 유망한 문턱만 실제 walk-forward 로 재검증한다.
                문턱을 올리면 거래 수가 줄어 t통계량이 오히려 내려갈 수 있으므로
                기대R만 보고 결론 내면 안 된다.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Costs, load_config                          # noqa: E402
from src.data.cache import get_klines                              # noqa: E402
from src.indicators import compute_all                             # noqa: E402
from src.regime import classify                                    # noqa: E402
from src.strategies import build_all                               # noqa: E402
from src.backtest import ExecParams, WeightParams, walk_forward     # noqa: E402

pd.set_option("display.width", 240)


def tstat(r: np.ndarray) -> tuple[float, float]:
    """(기대값, t통계량)"""
    n = len(r)
    if n < 20:
        return (float("nan"), float("nan"))
    mean, sd = float(r.mean()), float(r.std(ddof=1))
    se = sd / math.sqrt(n)
    return (mean, mean / se if se > 0 else 0.0)


def load(sym: str, interval: str, cfg):
    kd = cfg.path("paths.klines_dir")
    start, end = cfg.require("backtest.start"), cfg.require("backtest.end")
    df = get_klines(sym, interval, start, end, market=cfg.require("market"), klines_dir=kd)
    feats = compute_all(df)
    reg = classify(feats)["regime"]
    mn = get_klines(sym, "1m", start, end, market=cfg.require("market"), klines_dir=kd)
    return df, feats, reg, mn


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=cfg.require("universe"))
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--confirm", action="store_true", help="2단계 확인 실행까지 한다")
    ap.add_argument("--thresholds", nargs="*", type=float, default=[0.35, 0.50, 0.65])
    args = ap.parse_args()

    strat_names = cfg.get("strategies")
    strats = [s for s in build_all() if not strat_names or s.name in set(strat_names)]
    params = ExecParams(
        risk_per_trade=float(cfg.require("risk.risk_per_trade")),
        atr_stop_mult=float(cfg.require("risk.atr_stop_mult")),
        reward_ratio=float(cfg.require("risk.reward_ratio")),
        daily_loss_limit=float(cfg.require("risk.daily_loss_limit")),
    )
    costs = Costs.from_config(cfg)

    print(f"{args.interval} · 손절 {params.atr_stop_mult}xATR · 손익비 {params.reward_ratio}:1")
    print(f"전략 {[s.korean for s in strats]}")
    print(f"{len(args.symbols)}종목 · {cfg.require('backtest.start')} → {cfg.require('backtest.end')}\n")

    # ── 1단계: 기준 실행 한 번 + 확신도 층별 분석
    loaded, frames = {}, []
    n_windows = 0
    for sym in args.symbols:
        d = load(sym, args.interval, cfg)
        loaded[sym] = d
        wf = walk_forward(d[0], d[1], d[2], strats, cfg, interval=args.interval,
                          minute_df=d[3], symbol=sym, weight_params=WeightParams(),
                          exec_params=params, costs=costs)
        n_windows = len(wf.windows)
        if not wf.oos_trades.empty:
            frames.append(wf.oos_trades)
        m = wf.metrics
        print(f"  {sym:<9} 거래 {m.trades:>4}  수익률 {m.total_return*100:+7.2f}%  "
              f"샤프 {m.sharpe:+.2f}  PF {m.profit_factor:.2f}", flush=True)

    trades = pd.concat(frames, ignore_index=True)
    print(f"\nwalk-forward 창 {n_windows}개 · 통합 거래 {len(trades):,}건")

    base_mean, base_t = tstat(trades["r_multiple"].to_numpy())
    print(f"기준 통합 순기대R {base_mean:+.4f} (t={base_t:+.2f})")

    print("\n" + "=" * 78)
    print("1단계 — 확신도 층별 기대R (같은 실행에서 사후 분류, 공짜)")
    print("=" * 78)
    q = trades["strength"].quantile([0, .2, .4, .6, .8, 1.0]).to_numpy()
    rows = []
    for i in range(5):
        lo, hi = q[i], q[i + 1]
        sel = trades[(trades["strength"] >= lo) & (trades["strength"] <= hi)]
        m, t = tstat(sel["r_multiple"].to_numpy())
        g, _ = tstat(sel["gross_r"].to_numpy())
        rows.append({"층": f"{i*20}~{(i+1)*20}%", "확신도": f"{lo:.3f}~{hi:.3f}",
                     "거래": len(sel), "순기대R": round(m, 4),
                     "총기대R": round(g, 4), "t": round(t, 2)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n누적 상위 층 (문턱을 올렸을 때의 근사)")
    rows = []
    for pct in (0, 20, 40, 60, 80):
        cut = trades["strength"].quantile(pct / 100)
        sel = trades[trades["strength"] >= cut]
        m, t = tstat(sel["r_multiple"].to_numpy())
        rows.append({"하위절단": f"{pct}%", "확신도>=": round(cut, 3),
                     "거래": len(sel), "순기대R": round(m, 4), "t": round(t, 2)})
    cum = pd.DataFrame(rows)
    print(cum.to_string(index=False))

    corr = trades[["strength", "r_multiple"]].corr(method="spearman").iloc[0, 1]
    print(f"\n확신도 ↔ 순기대R 순위상관: {corr:+.4f}")
    if corr > 0.02:
        print("  → 양의 관계. 문턱을 올리면 기대값이 개선될 여지가 있다.")
    elif corr < -0.02:
        print("  → 음의 관계. 확신도가 높은 신호가 오히려 나쁘다 — 확신도 정의가 틀렸다.")
    else:
        print("  → 관계 없음. 확신도가 거래 품질을 예측하지 못한다. 선별도 상향은 무의미하다.")

    best = cum.loc[cum["t"].idxmax()]
    print(f"\nt 최대 지점: 하위 {best['하위절단']} 절단 · 거래 {best['거래']} · "
          f"기대R {best['순기대R']:+.4f} · t {best['t']:+.2f}  (기준 t {base_t:+.2f})")

    if not args.confirm:
        print("\n2단계 확인 실행은 --confirm 으로. (문턱을 실제로 바꿔 재실행)")
        return 0

    # ── 2단계: 실제 문턱을 바꿔 재검증
    print("\n" + "=" * 78)
    print("2단계 — entry_threshold 를 실제로 바꿔 walk-forward 재실행")
    print("=" * 78)
    rows = []
    for th in args.thresholds:
        wp = WeightParams(entry_threshold=th)
        pooled = []
        for sym, d in loaded.items():
            wf = walk_forward(d[0], d[1], d[2], strats, cfg, interval=args.interval,
                              minute_df=d[3], symbol=sym, weight_params=wp,
                              exec_params=params, costs=costs)
            if not wf.oos_trades.empty:
                pooled.append(wf.oos_trades)
        if not pooled:
            continue
        tr = pd.concat(pooled, ignore_index=True)
        m, t = tstat(tr["r_multiple"].to_numpy())
        pos = sum(1 for _, g in tr.groupby("symbol") if g["r_multiple"].mean() > 0)
        rows.append({"문턱": th, "거래": len(tr), "순기대R": round(m, 4),
                     "t": round(t, 2), "양수종목": f"{pos}/{tr['symbol'].nunique()}"})
        print(f"  문턱 {th:.2f} → 거래 {len(tr):>5}  기대R {m:+.4f}  t {t:+.2f}  "
              f"양수 {pos}/{tr['symbol'].nunique()}", flush=True)

    print()
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n문턱을 올리면 거래 수가 줄어 t가 내려갈 수 있다. 기대R과 t를 함께 봐야 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
