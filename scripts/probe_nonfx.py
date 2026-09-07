"""
코인 제외 선물성 종목(금·지수·에너지)에서 기존 전략이 사는지 재는 스크립트.

**왜 이걸 먼저 재는가.** 이 프로젝트에서 전략을 죽인 것은 언제나 비용이었다.
코인 총우위 t=3.5 가 비용을 빼면 t=1.4 였고, FX 는 우위 자체가 없었다
(config 주석: "어떤 값도 이 신호를 수익성 있게 만들지 못한다").

그런데 실측해 보니 비용이 훨씬 싼 시장이 있다 — 손절 4×ATR 기준 왕복비용이
1R 대비:

    GOLD                1.0%      ← FX 의 1/11
    US100Cash JP225Cash 1.5%
    GER40Cash US500Cash 2.2~2.3%
    US30Cash            2.5%
    FX (기준)           10.9%
    NGASCash            33.6%     ← 거래 불가

같은 전략이라도 비용이 1/10 이면 결과가 달라질 수 있다. **엔진을 확장하기
전에 그것부터 확인한다** — 재 보지도 않고 다중자산 엔진을 만들면, 안 되는
것을 위해 큰 구조를 짓는 셈이다.

**FX 심볼 레이어를 타지 않는다.** `symbols.parse()` 는 FX 가 아니면 None 을
주므로 금·지수는 통과하지 못한다. 여기서는 봉 → 지표 → 국면 → 단독 백테스트만
직접 엮어서, 엔진을 건드리지 않고 답을 얻는다.

사용법
    python scripts/probe_nonfx.py
    python scripts/probe_nonfx.py --symbols GOLD,US500Cash --months 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import ExecParams                                 # noqa: E402
from src.backtest.ensemble import score_by_regime                   # noqa: E402
from src.backtest.walkforward import run_standalone                 # noqa: E402
from src.config import Costs                                        # noqa: E402
from src.fx.engine import trader                                    # noqa: E402
from src.fx.mt5 import bars, symbol_info                            # noqa: E402
from src.indicators import compute_all                              # noqa: E402
from src.regime import classify                                     # noqa: E402

# 비용이 FX(10.9%)보다 확실히 싼 것들만. NGASCash(33.6%)·XPDUSD(25.2%)·
# XPTUSD(16.9%) 는 비용이 우위를 먹을 것이 뻔하므로 아예 넣지 않는다.
# US400Cash 는 중위 0.0% 인데 90분위가 24.7% 로 튀어 유동성을 의심한다 — 제외.
DEFAULT = ("GOLD", "US100Cash", "JP225Cash", "XAUEUR", "XAUJPY",
           "GER40Cash", "US500Cash", "US30Cash", "FRA40Cash",
           "OILCash", "BRENTCash", "SILVER")


def half_spread(df: pd.DataFrame, si) -> float:
    """
    편도 비용(비율). 엔진이 쓰는 것과 같은 방식 — 봉마다 저장된 그때의
    스프레드를 쓴다. 지금 호가창의 스프레드를 쓰면 휴장 중에 0 이 나온다.
    """
    if "spread" not in df.columns:
        return 0.0
    rel = (df["spread"] * si.point / df["close"]).replace(
        [np.inf, -np.inf], np.nan).dropna()
    return float(rel.median()) / 2.0 if len(rel) else 0.0


def measure(sym: str, t, months: int) -> pd.DataFrame:
    df = bars(sym, "1h", 6000)
    if len(df) < 1500:
        print(f"  {sym} 이력 부족 ({len(df)}봉)", flush=True)
        return pd.DataFrame()
    feats = compute_all(df)
    si = symbol_info(sym)
    half = half_spread(df, si)
    regs = classify(feats)["regime"]

    res = run_standalone(df, feats, t.strategies(), interval="1h",
                         costs=Costs(fee_rate=half, slippage_bps=0.5),
                         params=ExecParams(), initial_equity=10_000,
                         minute_df=None, regimes=regs, symbol=sym)
    frames = [r.trades for r in res.values() if not r.trades.empty]
    if not frames:
        print(f"  {sym} 거래 없음", flush=True)
        return pd.DataFrame()
    sc = score_by_regime(pd.concat(frames, ignore_index=True),
                         t.weight_params())
    sc["symbol"] = sym
    sc["cost_bp"] = round(half * 2 * 1e4, 2)
    sc["bars"] = len(df)
    print(f"  {sym} 완료 · {len(df)}봉 · 왕복 {half*2*1e4:.1f}bp", flush=True)
    return sc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(DEFAULT))
    ap.add_argument("--months", type=int, default=8)
    args = ap.parse_args()

    t = trader()
    wp = t.weight_params()
    declared = {s.name: {str(r) for r in s.regimes} for s in t.strategies()}

    frames = []
    for sym in args.symbols.split(","):
        try:
            r = measure(sym.strip(), t, args.months)
            if not r.empty:
                frames.append(r)
        except Exception as exc:
            print(f"  {sym} 실패: {type(exc).__name__}: {exc}", flush=True)
    if not frames:
        return 1

    all_ = pd.concat(frames, ignore_index=True)
    # 전략이 선언하지 않은 국면은 후보가 아니다 — 세면 성적이 부풀려진다.
    all_ = all_[[r in declared.get(s, set())
                 for s, r in zip(all_["strategy"], all_["regime"])]]

    pd.set_option("display.width", 200)
    print()
    print("=" * 78)
    print(f"코인 제외 종목 · 기존 {len(declared)}개 전략 · min_trades="
          f"{wp.min_trades}")
    print("=" * 78)

    print("\n── 종목별 요약 ──────────────────────────────")
    g = all_.groupby("symbol").agg(
        칸=("strategy", "count"),
        자격=("eligible", "sum"),
        거래=("trades", "sum"),
        기대R평균=("expectancy_r", "mean"),
        기대R최고=("expectancy_r", "max"),
        비용bp=("cost_bp", "first"))
    print(g.sort_values("자격", ascending=False).round(4).to_string())

    print("\n── 자격을 얻은 칸 (기대R>0 이고 거래 충분) ──────────────")
    ok = all_[all_["eligible"]].sort_values("expectancy_r", ascending=False)
    cols = ["symbol", "regime", "strategy", "trades", "expectancy_r",
            "win_rate", "cost_bp"]
    if ok.empty:
        print("  없음")
    else:
        print(ok[cols].round(4).to_string(index=False))

    print("\n── 국면별 (전 종목) ──────────────────────────────")
    r = all_.groupby("regime").agg(
        칸=("strategy", "count"), 자격=("eligible", "sum"),
        거래=("trades", "sum"), 기대R중위=("expectancy_r", "median"))
    print(r.round(4).to_string())

    print()
    tot = all_["expectancy_r"]
    print(f"전체 {len(all_)}칸 · 기대R 양수 {int((tot>0).sum())}칸 "
          f"({(tot>0).mean():.0%}) · 자격 {int(all_['eligible'].sum())}칸")
    print(f"기대R 평균 {tot.mean():+.4f} · 중위 {tot.median():+.4f}")
    print()
    print("비교: FX 7개 통화쌍에서는 25칸 중 자격 4칸, 기대R 평균 -0.070 이었다")
    print("      (order_flow_fade 측정 기준). 여기가 확실히 나으면 확장할 값이 있다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
