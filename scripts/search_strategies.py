"""
전략 × 국면 × 시간축 전수 탐색 — **다중검정 보정과 홀드아웃을 함께.**

**왜 보정이 먼저인가.**
    11전략 × 4국면 × 3시간축 = 132개 가설을 재고 그중 최고를 고르면, 우위가
    전혀 없어도 t=3 짜리가 하나쯤 나온다. 132번 던지면 p<0.01 짜리가 기대값
    1.3개다. 보정 없이 "최고 전략을 찾았다" 고 말하면 그 숫자는 뜻이 없다.

    이 프로젝트는 이미 그 함정을 안다 — Deflated Sharpe 0.865(기준 0.95),
    다중검정 보정 후 유의성 탈락. 같은 실수를 더 큰 규모로 반복하지 않는다.

**설계 셋.**

1. `gate=False` 로 돌린다. 전략이 선언하지 않은 국면에서도 재서, 선언이
   틀렸을 가능성까지 본다 — "가능한 많은 조합" 이 요청이었다.

2. **종목을 절반으로 나눈다.** 홀수/짝수로 갈라 한쪽에서 후보를 찾고
   다른 쪽에서 확인한다. 같은 데이터로 찾고 확인하면 순환논리다.
   나누는 기준은 정렬 후 번갈아 — 지수·금·에너지·FX 가 양쪽에 고루 들어간다.

3. **보정을 세 가지로 낸다.**
     Bonferroni  가장 보수적. 132 가설이면 p<0.00038 이어야 한다.
     BH(FDR)     덜 보수적. 발견 중 거짓 비율을 5% 로 통제한다.
     홀드아웃    보정과 무관하게, 다른 종목에서 재현되는가.

사용법
    python scripts/search_strategies.py
    python scripts/search_strategies.py --intervals 1d --min-trades 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scipy import stats as st                                       # noqa: E402

from src.backtest import ExecParams                                 # noqa: E402
from src.backtest.walkforward import run_standalone                 # noqa: E402
from src.config import Costs                                        # noqa: E402
from src.fx.engine import trader                                    # noqa: E402
from src.fx.mt5 import bars, symbol_info                            # noqa: E402
from src.indicators import compute_all                              # noqa: E402
from src.regime import classify                                     # noqa: E402

# 코인 제외, 만기 없는 종목 + FX. 비용이 1R 의 17% 를 넘는 것(NGASCash·
# XPDUSD·XPTUSD)은 애초에 거래 대상이 아니므로 뺀다 — 탐색 공간만 넓히고
# 결론에는 못 쓴다.
SYMBOLS = [
    # 지수
    "US500Cash", "US100Cash", "US30Cash", "GER40Cash", "UK100Cash",
    "JP225Cash", "FRA40Cash", "EU50Cash", "AUS200Cash", "HK50Cash",
    "IT40Cash", "NETH25Cash", "SWI20Cash", "ChinaHCash", "US2000Cash",
    "GerMid50Cash", "GerTech30Cash", "CA60Cash",
    # 금속·에너지
    "GOLD", "SILVER", "XAUEUR", "OILCash", "BRENTCash",
    # FX 메이저·크로스
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "GBPJPY", "EURAUD", "CADJPY", "AUDNZD", "GBPAUD",
]

BARS = {"1h": 20000, "4h": 17000, "1d": 8000}


def half_spread(df: pd.DataFrame, si) -> float:
    """
    편도 비용. **spread==0 인 봉을 뺀다** — MT5 옛 이력에는 스프레드가 없고
    0 으로 온다. "비용 없음" 이 아니라 "모름" 이다. 안 빼면 공짜로 거래하는
    셈이 되어 거짓 양수가 나온다(이 프로젝트에서 실제로 한 번 겪었다).
    """
    if "spread" not in df.columns:
        return 0.0
    rel = (df["spread"] * si.point / df["close"]).replace(
        [np.inf, -np.inf], np.nan)
    nz = rel[(df["spread"] > 0) & rel.notna()]
    return float(nz.median()) / 2.0 if len(nz) else 0.0


def trades_for(sym: str, interval: str, t) -> pd.DataFrame:
    """한 종목·한 시간축의 모든 전략 거래. 국면 라벨이 붙어 있다."""
    df = bars(sym, interval, BARS[interval])
    if len(df) < 800:
        return pd.DataFrame()
    si = symbol_info(sym)
    half = half_spread(df, si)
    if half <= 0:
        return pd.DataFrame()
    feats = compute_all(df)
    regs = classify(feats)["regime"]
    # **gate=False**: 선언하지 않은 국면에서도 신호를 살려 전부 잰다.
    res = run_standalone(df, feats, t.strategies(), interval=interval,
                         costs=Costs(fee_rate=half, slippage_bps=0.5),
                         params=ExecParams(), initial_equity=10_000,
                         minute_df=None, regimes=regs, symbol=sym, gate=False)
    rows = [r.trades for r in res.values() if not r.trades.empty]
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["symbol"] = sym
    out["interval"] = interval
    out["cost_bp"] = round(half * 2 * 1e4, 2)
    return out


def cells(tr: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    """
    (전략 × 국면 × 시간축) 마다 거래를 모아 t검정한다.

    **거래 단위로 검정한다.** 종목별 평균을 다시 평균하면 거래 30건인 종목이
    300건인 종목과 같은 무게를 갖는다.
    """
    rows = []
    for (strat, reg, iv), g in tr.groupby(["strategy", "regime", "interval"]):
        r = g["r_multiple"].astype("float64")
        n = len(r)
        if n < min_trades:
            continue
        sd = float(r.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            continue
        tstat = float(r.mean() / (sd / np.sqrt(n)))
        p = float(2 * (1 - st.t.cdf(abs(tstat), n - 1)))
        per_sym = g.groupby("symbol")["r_multiple"].mean()
        rows.append({
            "전략": strat, "국면": reg, "봉": iv,
            "거래": n, "종목": int(g["symbol"].nunique()),
            "기대R": round(float(r.mean()), 4),
            "중위R": round(float(r.median()), 4),
            "t": round(tstat, 2), "p": p,
            "승률": round(float((r > 0).mean()), 3),
            "양수종목": f"{int((per_sym > 0).sum())}/{len(per_sym)}",
            "비용R": round(float(g["cost_r"].mean()), 4),
        })
    return pd.DataFrame(rows)


def correct(df: pd.DataFrame) -> pd.DataFrame:
    """Bonferroni 와 BH(FDR) 보정을 붙인다."""
    if df.empty:
        return df
    m = len(df)
    d = df.sort_values("p").reset_index(drop=True).copy()
    d["가설수"] = m
    d["p_bonf"] = (d["p"] * m).clip(upper=1.0)
    # Benjamini-Hochberg
    ranks = np.arange(1, m + 1)
    bh = (d["p"] * m / ranks).values
    d["p_bh"] = np.minimum.accumulate(bh[::-1])[::-1].clip(0, 1)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervals", default="1d,4h")
    ap.add_argument("--min-trades", type=int, default=100)
    args = ap.parse_args()

    t = trader()
    intervals = [x.strip() for x in args.intervals.split(",") if x.strip()]

    # **정렬 후 번갈아 나눈다.** 지수·금속·FX 가 양쪽에 고루 들어가게.
    ordered = sorted(SYMBOLS)
    disc = ordered[0::2]
    hold = ordered[1::2]
    print(f"탐색군 {len(disc)}종목 · 홀드아웃 {len(hold)}종목")

    frames = []
    for iv in intervals:
        for sym in SYMBOLS:
            try:
                tr = trades_for(sym, iv, t)
            except Exception as exc:
                print(f"  {iv} {sym:14s} 실패 {type(exc).__name__}", flush=True)
                continue
            if tr.empty:
                print(f"  {iv} {sym:14s} 건너뜀", flush=True)
                continue
            frames.append(tr)
            print(f"  {iv} {sym:14s} 거래 {len(tr):>5}", flush=True)
    if not frames:
        return 1

    all_tr = pd.concat(frames, ignore_index=True)
    pd.set_option("display.width", 240)

    print()
    print("=" * 96)
    print(f"전수 탐색 · 거래 {len(all_tr):,}건 · "
          f"전략 {all_tr['strategy'].nunique()} × 국면 "
          f"{all_tr['regime'].nunique()} × 봉 {all_tr['interval'].nunique()}")
    print("=" * 96)

    d_cells = correct(cells(all_tr[all_tr["symbol"].isin(disc)],
                            args.min_trades))
    if d_cells.empty:
        print("탐색군에 최소 거래를 넘는 칸이 없다")
        return 1

    print(f"\n── 탐색군 상위 15 (가설 {len(d_cells)}개, "
          f"Bonferroni 문턱 p<{0.05/len(d_cells):.5f}) ──")
    cols = ["전략", "국면", "봉", "거래", "종목", "기대R", "중위R", "t",
            "승률", "양수종목", "비용R", "p_bonf", "p_bh"]
    top = d_cells.sort_values("t", ascending=False).head(15)
    print(top[cols].assign(
        p_bonf=top["p_bonf"].round(4), p_bh=top["p_bh"].round(4)
    ).to_string(index=False))

    print(f"\n  Bonferroni 통과: {int((d_cells['p_bonf'] < 0.05).sum())}개")
    print(f"  BH(FDR 5%) 통과: {int((d_cells['p_bh'] < 0.05).sum())}개")

    # ── 홀드아웃에서 재현되는가
    h_cells = cells(all_tr[all_tr["symbol"].isin(hold)], args.min_trades)
    if h_cells.empty:
        print("\n홀드아웃에 최소 거래를 넘는 칸이 없다")
        return 0

    key = ["전략", "국면", "봉"]
    merged = d_cells.merge(h_cells, on=key, suffixes=("_탐색", "_홀드"))
    print(f"\n── 홀드아웃 재현 (양쪽 다 있는 {len(merged)}칸) ──")
    show = merged.sort_values("t_탐색", ascending=False).head(15)
    print(show[key + ["기대R_탐색", "t_탐색", "기대R_홀드", "t_홀드",
                      "거래_홀드", "양수종목_홀드"]].to_string(index=False))

    both = merged[(merged["t_탐색"] > 2) & (merged["t_홀드"] > 2)]
    print(f"\n  양쪽 모두 t>2 인 칸: {len(both)}개")
    if not both.empty:
        print(both[key + ["기대R_탐색", "t_탐색", "기대R_홀드",
                          "t_홀드"]].to_string(index=False))

    # 탐색군 t 와 홀드아웃 t 의 상관 — 전체 그림이 우연인지 본다
    if len(merged) > 5:
        c = merged["t_탐색"].corr(merged["t_홀드"])
        print(f"\n  탐색군 t 와 홀드아웃 t 의 상관: {c:+.3f}")
        print("  (0 근처면 탐색군 순위가 홀드아웃에서 아무 의미가 없다는 뜻이다)")

    print()
    print("판단 기준")
    print("  · Bonferroni 또는 BH 를 통과하고, **홀드아웃에서도 t>2** 여야 한다.")
    print("  · 둘 중 하나만 만족하면 채택하지 않는다 — 132 가설 중 최고를")
    print("    고르는 것이므로 보정 없는 p 값은 뜻이 없다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
