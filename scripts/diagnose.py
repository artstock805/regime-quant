"""
'왜 우위가 없는가'를 분해하는 진단 도구.

    python scripts/diagnose.py --interval 1h

세 가지를 본다.
  1) 레짐 게이팅이 값을 하는가 — 선언 국면 안/밖의 총기대R 비교.
     이 프로젝트의 핵심 가설이 여기서 검증되거나 기각된다.
  2) 청산 사유별 기여 — 어디서 돈이 새는지.
  3) 손절폭 적정성 — MAE/MFE 분포로 손절이 너무 좁은지 넓은지.

전부 **총(gross) 기대R** 기준으로 본다. 순기대값은 비용에 가려 신호가 안 보인다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Costs, load_config                    # noqa: E402
from src.data.cache import get_klines                        # noqa: E402
from src.indicators import compute_all                       # noqa: E402
from src.regime import KOREAN, Regime, classify              # noqa: E402
from src.strategies import build_all                         # noqa: E402
from src.backtest import ExecParams, run                     # noqa: E402

pd.set_option("display.width", 240)


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--stop-mult", type=float, default=4.0)
    args = ap.parse_args()

    costs = Costs.from_config(cfg)
    params = ExecParams(
        risk_per_trade=float(cfg.require("risk.risk_per_trade")),
        atr_stop_mult=args.stop_mult,
        reward_ratio=float(cfg.require("risk.reward_ratio")),
        daily_loss_limit=float(cfg.require("risk.daily_loss_limit")),
    )
    strats = build_all()
    km = {s.name: s.korean for s in strats}

    print(f"{args.interval} · 손절 {args.stop_mult}xATR · {', '.join(args.symbols)}\n")

    frames = []
    for sym in args.symbols:
        df, feats, reg, mn = _load(sym, args.interval, cfg)
        for s in strats:
            sig = s.generate(df, feats)
            declared = reg.reindex(df.index).isin([str(r) for r in s.regimes])

            # 게이팅 없이 전 구간에서 돌린다. 그 뒤에 진입 시점의 레짐으로 갈라 본다.
            res = run(df, sig, feats["atr"], interval=args.interval, costs=costs,
                      params=params, initial_equity=10_000, minute_df=mn,
                      regimes=reg, symbol=sym, strategy=s.name)
            t = res.trades
            if t.empty:
                continue
            t = t.copy()
            t["in_declared"] = t["regime"].isin([str(r) for r in s.regimes])
            frames.append(t)

    trades = pd.concat(frames, ignore_index=True)
    print(f"총 {len(trades):,}거래\n")

    # ── 1) 레짐 게이팅의 값
    print("━━━ 1) 선언 국면 안 / 밖 총기대R ━━━")
    rows = []
    for name, g in trades.groupby("strategy"):
        ins, outs = g[g["in_declared"]], g[~g["in_declared"]]
        rows.append({
            "전략": km.get(name, name),
            "안_거래": len(ins), "안_총기대R": round(float(ins["gross_r"].mean()), 4) if len(ins) else None,
            "밖_거래": len(outs), "밖_총기대R": round(float(outs["gross_r"].mean()), 4) if len(outs) else None,
            "차이": (round(float(ins["gross_r"].mean()) - float(outs["gross_r"].mean()), 4)
                   if len(ins) and len(outs) else None),
        })
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))
    diff = tbl["차이"].dropna()
    print(f"\n  게이팅이 도움된 전략: {int((diff > 0).sum())}/{len(diff)}  "
          f"· 평균 차이 {diff.mean():+.4f}R")
    print("  → 차이가 0 근처면 레짐 분류가 전략 선택에 정보를 주지 못한다는 뜻이다.")

    # ── 2) 레짐별 총기대R
    print("\n━━━ 2) 진입 시점 레짐별 총기대R (전 전략) ━━━")
    by_reg = trades.groupby("regime").agg(
        거래=("gross_r", "size"), 총기대R=("gross_r", "mean"),
        순기대R=("r_multiple", "mean"), 승률=("net_pnl", lambda s: (s > 0).mean() * 100),
    ).round(4)
    by_reg.index = [KOREAN.get(Regime(i), i) if i in {str(x) for x in Regime} else i
                    for i in by_reg.index]
    print(by_reg.to_string())

    # ── 3) 청산 사유
    print("\n━━━ 3) 청산 사유별 기여 ━━━")
    by_exit = trades.groupby("exit_reason").agg(
        거래=("gross_r", "size"), 비중_pct=("gross_r", lambda s: len(s) / len(trades) * 100),
        총기대R=("gross_r", "mean"), 순기대R=("r_multiple", "mean"),
    ).round(4)
    print(by_exit.to_string())

    # ── 4) 손절폭 적정성
    print("\n━━━ 4) 손절폭 적정성 (MAE=최대역행, MFE=최대순행, %) ━━━")
    stop_pct = float(trades["risk_amount"].iloc[0]) and None  # 표시용 아님
    print(trades[["mae_pct", "mfe_pct"]].describe().loc[["mean", "50%", "75%", "max"]].round(3).to_string())
    print(f"\n  MFE/MAE 중위 비 = {trades['mfe_pct'].median() / trades['mae_pct'].median():.2f}")
    print("  → 1보다 크면 순행 여력이 있는데 손절이 먼저 걸린 것이고,")
    print("    1보다 작으면 진입 자체가 방향을 못 맞춘 것이다.")
    return 0


def _load(sym, interval, cfg):
    kd = cfg.path("paths.klines_dir")
    start, end = cfg.require("backtest.start"), cfg.require("backtest.end")
    df = get_klines(sym, interval, start, end, market=cfg.require("market"), klines_dir=kd)
    feats = compute_all(df)
    reg = classify(feats)["regime"]
    mn = get_klines(sym, "1m", start, end, market=cfg.require("market"), klines_dir=kd)
    return df, feats, reg, mn


if __name__ == "__main__":
    raise SystemExit(main())
