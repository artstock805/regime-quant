"""
FX 오더플로 대용 전략을 **여러 통화쌍에서** 측정한다.

EURUSD 한 쌍 6개월 결과로 "쓸 수 없다"고 결론 내릴 수는 없다. 한 쌍에서
음수인 것은 운일 수도 있고, 반대로 한 쌍에서 양수인 것은 더 위험하다
(그걸 근거로 쓰면 과적합이다). 통화쌍마다 부호가 어떻게 나오는지 본다.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import ExecParams, derive_weights
from src.backtest.ensemble import score_by_regime
from src.backtest.walkforward import run_standalone
from src.config import Costs
from src.fx.engine import trader
from src.fx.mt5 import available_symbols, bars, symbol_info
from src.fx.symbols import catalogue
from src.indicators import compute_all
from src.regime import classify

TARGET = "order_flow_fade"


def measure(sym: str, t, months: int = 6) -> pd.DataFrame:
    df = bars(sym, "1h", t.bars_needed())
    feats = compute_all(df)
    si = symbol_info(sym)
    half = float((df["spread"] * si.point / df["close"]).median()) / 2.0
    costs = Costs(fee_rate=half, slippage_bps=0.5)
    regs = classify(feats)["regime"]

    cutoff = df.index[-1] - timedelta(days=months * 31)
    tr = df.index >= cutoff
    res = run_standalone(df[tr], feats[tr], t.strategies(), interval="1h",
                         costs=costs, params=ExecParams(), initial_equity=10_000,
                         minute_df=None, regimes=regs[tr], symbol=sym)
    frames = [r.trades for r in res.values() if not r.trades.empty]
    if not frames:
        return pd.DataFrame()
    sc = score_by_regime(pd.concat(frames, ignore_index=True), t.weight_params())
    sc = sc[sc["strategy"] == TARGET].copy()
    sc["symbol"] = sym
    sc["cost_bp"] = round(half * 2 * 1e4, 2)
    return sc


def main() -> int:
    t = trader()
    avail = available_symbols()
    pairs = [p.mt5 for p in catalogue(avail)][:7]
    out = []
    for sym in pairs:
        try:
            r = measure(sym, t)
            if not r.empty:
                out.append(r)
                print(f"  {sym} 완료", flush=True)
            else:
                print(f"  {sym} 거래 없음", flush=True)
        except Exception as exc:
            print(f"  {sym} 실패: {exc}", flush=True)

    if not out:
        print("측정할 것이 없다")
        return 1
    all_ = pd.concat(out, ignore_index=True)
    cols = ["symbol", "regime", "trades", "expectancy_r", "win_rate",
            "total_r", "eligible", "cost_bp"]
    print()
    print("=== order_flow_fade · 통화쌍 × 국면 ===")
    print(all_[cols].sort_values(["regime", "symbol"]).to_string(index=False))

    print()
    print("=== 국면별 집계 (통화쌍 전체) ===")
    g = all_.groupby("regime").agg(
        쌍수=("symbol", "nunique"), 거래=("trades", "sum"),
        기대R평균=("expectancy_r", "mean"), 기대R중위=("expectancy_r", "median"),
        음수쌍=("expectancy_r", lambda s: int((s < 0).sum())),
        자격=("eligible", "sum"))
    print(g.round(4).to_string())

    print()
    tot = all_["expectancy_r"]
    print(f"전체 {len(all_)}개 (쌍×국면) 중 기대R 음수 {int((tot < 0).sum())}개")
    print(f"기대R 평균 {tot.mean():+.4f} · 중위 {tot.median():+.4f}")
    print(f"자격을 얻은 칸: {int(all_['eligible'].sum())}개")
    return 0


if __name__ == "__main__":
    sys.exit(main())
