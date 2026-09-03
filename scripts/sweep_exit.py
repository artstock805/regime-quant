"""
청산 구조 스윕 — 손익비(reward ratio)를 얼마로 둘 것인가.

    python scripts/sweep_exit.py

두 질문에 답한다.
  1) 1:1 이 2:1 보다 나은가 — 전체적으로.
  2 ) 최적 손익비가 **레짐마다 다른가**. 다르면 고정값이 아니라
      레짐별 청산 구조를 써야 한다.

데이터를 한 번만 로드해서 손익비만 바꿔 돌린다. CLI 를 여러 번 부르면
캔들·지표를 매번 다시 계산해서 훨씬 느리다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Costs, load_config                     # noqa: E402
from src.data.cache import get_klines                         # noqa: E402
from src.indicators import compute_all                        # noqa: E402
from src.regime import KOREAN, Regime, classify               # noqa: E402
from src.strategies import build_all                          # noqa: E402
from src.backtest import (                                    # noqa: E402
    ExecParams, WeightParams, run_standalone, walk_forward,
)

pd.set_option("display.width", 240)


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--stop-mult", type=float, default=4.0)
    ap.add_argument("--ratios", nargs="*", type=float, default=[1.0, 1.5, 2.0, 3.0])
    args = ap.parse_args()

    costs = Costs.from_config(cfg)
    strats = build_all()
    km = {s.name: s.korean for s in strats}
    kd = cfg.path("paths.klines_dir")
    start, end = cfg.require("backtest.start"), cfg.require("backtest.end")

    print(f"{args.interval} · 손절 {args.stop_mult}xATR · 손익비 {args.ratios}")
    print(f"{', '.join(args.symbols)} · {start} → {end}\n")

    # 데이터는 한 번만 로드한다
    loaded = {}
    for sym in args.symbols:
        df = get_klines(sym, args.interval, start, end,
                        market=cfg.require("market"), klines_dir=kd)
        feats = compute_all(df)
        reg = classify(feats)["regime"]
        mn = get_klines(sym, "1m", start, end, market=cfg.require("market"), klines_dir=kd)
        loaded[sym] = (df, feats, reg, mn)
        print(f"  {sym} {len(df):,}봉 로드", flush=True)

    def params_for(ratio: float) -> ExecParams:
        return ExecParams(
            risk_per_trade=float(cfg.require("risk.risk_per_trade")),
            atr_stop_mult=args.stop_mult, reward_ratio=ratio,
            daily_loss_limit=float(cfg.require("risk.daily_loss_limit")),
        )

    # ── 질문 1: 손익비별 out-of-sample 성적
    print("\n━━━ 1) 손익비별 out-of-sample (walk-forward) ━━━")
    wf_rows = []
    for ratio in args.ratios:
        for sym, (df, feats, reg, mn) in loaded.items():
            wf = walk_forward(df, feats, reg, strats, cfg, interval=args.interval,
                              minute_df=mn, symbol=sym, weight_params=WeightParams(),
                              exec_params=params_for(ratio))
            m = wf.metrics
            wf_rows.append({
                "손익비": ratio, "종목": sym, "거래": m.trades,
                "수익률_%": round(m.total_return * 100, 2), "샤프": round(m.sharpe, 2),
                "MDD_%": round(m.max_drawdown * 100, 2), "PF": round(m.profit_factor, 2),
                "승률_%": round(m.win_rate * 100, 1),
            })
            print(f"  {ratio}:1  {sym:<9} {m.total_return*100:+6.2f}%  "
                  f"샤프 {m.sharpe:+.2f}  MDD {m.max_drawdown*100:5.2f}%  "
                  f"PF {m.profit_factor:.2f}  거래 {m.trades}", flush=True)

    wf = pd.DataFrame(wf_rows)
    print("\n  종목 평균")
    agg = wf.groupby("손익비").agg(
        수익률=("수익률_%", "mean"), 샤프=("샤프", "mean"), MDD=("MDD_%", "mean"),
        PF=("PF", "mean"), 승률=("승률_%", "mean"), 거래=("거래", "mean"),
    ).round(3)
    print(agg.to_string())
    best = agg["샤프"].idxmax()
    print(f"\n  샤프 최고 손익비: {best}:1")

    # ── 질문 2: 레짐마다 최적 손익비가 다른가
    print("\n━━━ 2) 레짐별 최적 손익비 (단독 전략, 진입 시점 레짐 기준) ━━━")
    reg_rows = []
    for ratio in args.ratios:
        frames = []
        for sym, (df, feats, reg, mn) in loaded.items():
            res = run_standalone(df, feats, strats, interval=args.interval, costs=costs,
                                 params=params_for(ratio), initial_equity=10_000,
                                 minute_df=mn, regimes=reg, symbol=sym)
            frames.append(pd.concat([r.trades for r in res.values()], ignore_index=True))
        trades = pd.concat(frames, ignore_index=True)
        for regime, g in trades.groupby("regime"):
            reg_rows.append({
                "손익비": ratio, "레짐": regime, "거래": len(g),
                "순기대R": round(float(g["r_multiple"].mean()), 4),
                "총기대R": round(float(g["gross_r"].mean()), 4),
                "승률_%": round(float((g["net_pnl"] > 0).mean()) * 100, 1),
            })

    rt = pd.DataFrame(reg_rows)
    piv = rt.pivot_table(index="레짐", columns="손익비", values="순기대R")
    piv.index = [KOREAN.get(Regime(i), i) if i in {str(x) for x in Regime} else i
                 for i in piv.index]
    print("\n  순기대R (레짐 x 손익비)")
    print(piv.round(4).to_string())
    print("\n  레짐별 최적 손익비")
    for regime in piv.index:
        row = piv.loc[regime]
        print(f"    {regime:<6} {row.idxmax()}:1  (순기대R {row.max():+.4f})")

    spread = piv.idxmax(axis=1).nunique()
    print(f"\n  최적 손익비가 서로 다른 레짐 수: {spread} / {len(piv)}")
    if spread == 1:
        print("  → 전 레짐 동일. 고정 손익비를 쓰면 된다.")
    else:
        print("  → 레짐마다 다르다. 레짐별 청산 구조(적응형)를 써야 한다.")

    print("\n  거래 수 (표본 충분한지 확인)")
    print(rt.pivot_table(index="레짐", columns="손익비", values="거래").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
