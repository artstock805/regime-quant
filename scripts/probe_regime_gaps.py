"""
국면별 자격 공백이 왜 생기나 — 표본 부족인가 우위 부재인가.

**이 구분이 처방을 바꾼다.**

    기대R 이 음수라서 탈락  → 그 국면에서 그 전략은 쓰면 안 된다. 끝.
    거래 수가 모자라서 탈락 → 우위는 있을 수 있는데 못 재고 있다.
                              학습 구간을 늘리거나 통화쌍을 합쳐야 한다.

EURUSD 6개월에서 하락 국면 최고가 keltner_squeeze 기대R +0.054(14건)였다.
min_trades=20 에 걸려 탈락했다 — 부호는 맞는데 표본이 없다. 그것이 7개
통화쌍에서 공통이면 처방은 "표본을 늘리는 것"이고, 흩어져 있으면 잡음이다.
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import ExecParams                                 # noqa: E402
from src.backtest.ensemble import score_by_regime                   # noqa: E402
from src.backtest.walkforward import run_standalone                 # noqa: E402
from src.config import Costs                                        # noqa: E402
from src.fx.engine import trader                                    # noqa: E402
from src.fx.mt5 import available_symbols, bars, symbol_info          # noqa: E402
from src.fx.symbols import catalogue                                # noqa: E402
from src.indicators import compute_all                              # noqa: E402
from src.regime import classify                                     # noqa: E402


def per_pair(sym: str, t, months: int) -> pd.DataFrame:
    df = bars(sym, "1h", t.bars_needed())
    feats = compute_all(df)
    si = symbol_info(sym)
    half = float((df["spread"] * si.point / df["close"]).median()) / 2.0
    regs = classify(feats)["regime"]

    cutoff = df.index[-1] - timedelta(days=months * 31)
    tr = df.index >= cutoff
    res = run_standalone(df[tr], feats[tr], t.strategies(), interval="1h",
                         costs=Costs(fee_rate=half, slippage_bps=0.5),
                         params=ExecParams(), initial_equity=10_000,
                         minute_df=None, regimes=regs[tr], symbol=sym)
    frames = [r.trades for r in res.values() if not r.trades.empty]
    if not frames:
        return pd.DataFrame()
    sc = score_by_regime(pd.concat(frames, ignore_index=True), t.weight_params())
    sc["symbol"] = sym
    return sc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=6)
    args = ap.parse_args()

    t = trader()
    wp = t.weight_params()
    min_trades = wp.min_trades
    declared = {s.name: {str(r) for r in s.regimes} for s in t.strategies()}

    pairs = [p.mt5 for p in catalogue(available_symbols())][:7]
    frames = []
    for sym in pairs:
        try:
            r = per_pair(sym, t, args.months)
            if not r.empty:
                frames.append(r)
                print(f"  {sym} 완료", flush=True)
        except Exception as exc:
            print(f"  {sym} 실패: {type(exc).__name__}: {exc}", flush=True)
    if not frames:
        return 1

    all_ = pd.concat(frames, ignore_index=True)
    # 전략이 선언하지 않은 국면은 애초에 후보가 아니다 — 세면 안 된다.
    all_ = all_[[r in declared.get(s, set())
                 for s, r in zip(all_["strategy"], all_["regime"])]]

    def why(row) -> str:
        if row["expectancy_r"] <= 0:
            return "기대R 음수"
        if row["trades"] < min_trades:
            return f"거래 부족 (<{min_trades})"
        return "자격 있음"

    all_["판정"] = all_.apply(why, axis=1)

    print()
    print("=" * 74)
    print(f"국면별 자격 판정 · {args.months}개월 · min_trades={min_trades}")
    print("=" * 74)
    piv = (all_.pivot_table(index="regime", columns="판정", values="strategy",
                            aggfunc="count", fill_value=0))
    print(piv.to_string())

    print()
    print("=== 거래 부족으로만 탈락한 칸 (부호는 맞는데 못 재는 것들) ===")
    near = all_[all_["판정"].str.startswith("거래 부족")].copy()
    near = near.sort_values(["regime", "expectancy_r"], ascending=[True, False])
    cols = ["regime", "symbol", "strategy", "trades", "expectancy_r", "win_rate"]
    print(near[cols].round(4).to_string(index=False))

    print()
    print("=== 국면별: 거래 부족 칸의 기대R 이 계통적으로 양수인가 ===")
    for reg, g in near.groupby("regime"):
        pos = int((g["expectancy_r"] > 0).sum())
        print(f"  {reg:10s} 칸 {len(g):>3}개 · 양수 {pos}개 "
              f"· 기대R 중위 {g['expectancy_r'].median():+.4f} "
              f"· 거래 합계 {int(g['trades'].sum())}")
    print()
    print("거래 합계가 min_trades 를 훌쩍 넘으면, 통화쌍을 합쳐서 재면")
    print("자격을 얻을 수 있다는 뜻이다 (같은 전략·같은 국면이므로).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
