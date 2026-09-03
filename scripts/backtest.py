"""
백테스트 CLI.

    python scripts/backtest.py --mode standalone --interval 15m
    python scripts/backtest.py --mode sweep          # 타임프레임 x 손절폭
    python scripts/backtest.py --mode walkforward --interval 1h

--mode standalone   전략을 단독으로 돌려 레짐별 리더보드를 뽑는다 (in-sample)
--mode sweep        타임프레임 x 손절배수 격자에서 총/순 기대값을 비교한다
--mode walkforward  학습에서 가중치를 뽑아 검증 구간에 적용한다 (out-of-sample)

합격 판정은 walkforward 결과로만 한다. standalone/sweep 은 진단용이다.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Costs, load_config                                # noqa: E402
from src.data.cache import get_klines                                    # noqa: E402
from src.indicators import compute_all                                   # noqa: E402
from src.regime import KOREAN, Regime, classify                          # noqa: E402
from src.strategies import build_all                                     # noqa: E402
from src.backtest import (                                               # noqa: E402
    ExecParams, WeightParams, average_weights, leaderboard, metrics_table,
    passes_acceptance, run_standalone, score_by_regime, walk_forward, weight_stability,
)

pd.set_option("display.width", 240)


def pick_strategies(args):
    """--strategies 로 지정한 것만. 이름이 틀리면 조용히 넘기지 않고 바로 터진다."""
    strats = build_all()
    if not args.strategies:
        return strats
    known = {s.name for s in strats}
    unknown = [n for n in args.strategies if n not in known]
    if unknown:
        raise SystemExit(
            f"없는 전략: {unknown}\n등록된 전략: {sorted(known)}"
        )
    return [s for s in strats if s.name in set(args.strategies)]


def costs_from(args, cfg) -> Costs:
    base = Costs.from_config(cfg)
    if args.fee_rate is None:
        return base
    return Costs(fee_rate=args.fee_rate, slippage_bps=base.slippage_bps)


def load_symbol(sym: str, interval: str, cfg, with_minutes: bool = True):
    kd = cfg.path("paths.klines_dir")
    start, end = cfg.require("backtest.start"), cfg.require("backtest.end")
    df = get_klines(sym, interval, start, end, market=cfg.require("market"), klines_dir=kd)
    feats = compute_all(df)
    reg = classify(feats)["regime"]
    mn = get_klines(sym, "1m", start, end, market=cfg.require("market"), klines_dir=kd) \
        if with_minutes else None
    return df, feats, reg, mn


def edge_breakdown(trades: pd.DataFrame) -> dict:
    """
    순 기대값과 총 기대값을 분리한다. 둘 다 트레이드별 실제 금액에서 나온다 —
    중위 ATR로 추정하면 판단 근거가 추정치가 된다.

    순 기대값이 음수여도 총 기대값이 양수라면 '전략에 우위가 없다'가 아니라
    '우위가 비용보다 작다'는 뜻이고, 처방이 완전히 다르다.
    """
    if trades.empty:
        return {}
    cost_r = float(trades["cost_r"].mean())
    return {
        "거래": len(trades),
        "비용/1R_%": round(cost_r * 100, 1),
        "순기대R": round(float(trades["r_multiple"].mean()), 4),
        "총기대R": round(float(trades["gross_r"].mean()), 4),
        "승률_%": round(float((trades["net_pnl"] > 0).mean()) * 100, 1),
        "손익분기승률_%": round((1 + cost_r) / 3 * 100, 1),
    }


def mode_standalone(args, cfg) -> int:
    costs = costs_from(args, cfg)
    params = ExecParams(
        risk_per_trade=float(cfg.require("risk.risk_per_trade")),
        atr_stop_mult=args.stop_mult,
        reward_ratio=float(cfg.require("risk.reward_ratio")),
        daily_loss_limit=float(cfg.require("risk.daily_loss_limit")),
        trailing=not args.no_trailing,
    )
    strats = pick_strategies(args)
    km = {s.name: s.korean for s in strats}
    per_symbol: dict[str, pd.DataFrame] = {}
    all_trades = []

    for sym in args.symbols:
        df, feats, reg, mn = load_symbol(sym, args.interval, cfg)
        res = run_standalone(
            df, feats, strats, interval=args.interval, costs=costs, params=params,
            initial_equity=float(cfg.require("backtest.initial_equity")),
            minute_df=mn, regimes=reg, symbol=sym,
        )
        tbl = metrics_table({km[k]: v.metrics for k, v in res.items()})
        per_symbol[sym] = tbl
        trades = pd.concat([r.trades for r in res.values()], ignore_index=True)
        all_trades.append(trades)
        print(f"\n━━━ {sym} · {args.interval} · 손절 {args.stop_mult}xATR ━━━")
        print(tbl.to_string(index=False))
        bd = edge_breakdown(trades)
        print("  " + "  ".join(f"{k} {v}" for k, v in bd.items()))

    trades = pd.concat(all_trades, ignore_index=True)
    scores = score_by_regime(trades, WeightParams())
    print("\n━━━ 레짐별 전략 성과 (전 종목 합산) ━━━")
    for reg in Regime:
        lb = leaderboard(scores, _dummy_weights(strats), reg, km)
        if lb["거래"].sum() == 0:
            continue
        print(f"\n[{KOREAN[reg]}]")
        print(lb.drop(columns=["가중치", "상태"]).to_string(index=False))
    return 0


def _dummy_weights(strats):
    """leaderboard 가 가중치 열을 요구하므로 0 행렬을 만들어 넘긴다."""
    return pd.DataFrame(
        0.0, index=[str(r) for r in Regime], columns=sorted(s.name for s in strats)
    )


def mode_sweep(args, cfg) -> int:
    costs = costs_from(args, cfg)
    strats = pick_strategies(args)
    rows = []
    for interval in args.sweep_intervals:
        for sym in args.symbols:
            df, feats, reg, mn = load_symbol(sym, interval, cfg)
            for mult in args.sweep_mults:
                params = ExecParams(
                    risk_per_trade=float(cfg.require("risk.risk_per_trade")),
                    atr_stop_mult=mult,
                    reward_ratio=float(cfg.require("risk.reward_ratio")),
                    daily_loss_limit=float(cfg.require("risk.daily_loss_limit")),
                    trailing=not args.no_trailing,
                )
                res = run_standalone(
                    df, feats, strats, interval=interval, costs=costs, params=params,
                    initial_equity=float(cfg.require("backtest.initial_equity")),
                    minute_df=mn, regimes=reg, symbol=sym,
                )
                trades = pd.concat([r.trades for r in res.values()], ignore_index=True)
                bd = edge_breakdown(trades)
                rows.append({"TF": interval, "종목": sym, "손절배수": mult, **bd})
                print(f"  {interval:>4} {sym:<9} {mult:>4}x  "
                      f"순 {bd.get('순기대R', 0):+.4f}R  총 {bd.get('총기대R', 0):+.4f}R  "
                      f"거래 {bd.get('거래', 0):,}", flush=True)

    t = pd.DataFrame(rows)
    print("\n━━━ 스윕 결과 (종목 평균) ━━━")
    agg = t.groupby(["TF", "손절배수"]).agg(
        비용_1R_pct=("비용/1R_%", "mean"), 순기대R=("순기대R", "mean"),
        총기대R=("총기대R", "mean"), 승률=("승률_%", "mean"),
        손익분기=("손익분기승률_%", "mean"), 거래=("거래", "mean"),
    ).round(4)
    print(agg.to_string())
    print("\n순기대R > 0 인 조합:")
    good = agg[agg["순기대R"] > 0]
    print(good.to_string() if not good.empty else "  없음")
    return 0


def mode_walkforward(args, cfg) -> int:
    params = ExecParams(
        risk_per_trade=float(cfg.require("risk.risk_per_trade")),
        atr_stop_mult=args.stop_mult,
        reward_ratio=args.reward_ratio,
        daily_loss_limit=float(cfg.require("risk.daily_loss_limit")),
        trailing=not args.no_trailing,
    )
    strats = pick_strategies(args)
    km = {s.name: s.korean for s in strats}
    results = []
    for sym in args.symbols:
        df, feats, reg, mn = load_symbol(sym, args.interval, cfg)
        wf = walk_forward(
            df, feats, reg, strats, cfg, interval=args.interval,
            minute_df=mn, symbol=sym, weight_params=WeightParams(),
            exec_params=params, gate=not args.no_gate,
            adaptive_exit=args.adaptive_exit, costs=costs_from(args, cfg),
        )
        results.append(wf)
        ok, why = passes_acceptance(wf.metrics, cfg)
        print(f"\n━━━ {sym} · {args.interval} · walk-forward ({len(wf.windows)}창) ━━━")
        for line in wf.metrics.summary_lines():
            print("  " + line)
        print(f"  합격: {'통과' if ok else '미달 — ' + '; '.join(why)}")

    print("\n━━━ 전 종목 out-of-sample 요약 ━━━")
    print(metrics_table({r.symbol: r.metrics for r in results}).to_string(index=False))

    if results and results[0].reward_per_window:
        print("\n━━━ 창별로 학습에서 뽑은 레짐별 손익비 ━━━")
        rw = pd.DataFrame([r for res in results for r in res.reward_per_window])
        rw.columns = [KOREAN.get(Regime(c), c) for c in rw.columns]
        print(rw.agg(["mean", "std", "min", "max"]).round(2).to_string())

    if results and results[0].weights_per_window:
        print("\n━━━ 레짐별 평균 가중치 (창간) ━━━")
        avg = average_weights([w for r in results for w in r.weights_per_window])
        avg.columns = [km.get(c, c) for c in avg.columns]
        avg.index = [KOREAN.get(Regime(i), i) for i in avg.index]
        print(avg.round(3).to_string())
        print("\n━━━ 가중치 창간 표준편차 (클수록 과최적화 위험) ━━━")
        sd = weight_stability([w for r in results for w in r.weights_per_window])
        sd.columns = [km.get(c, c) for c in sd.columns]
        sd.index = [KOREAN.get(Regime(i), i) for i in sd.index]
        print(sd.to_string())
    return 0


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="레짐 퀀트 백테스트")
    ap.add_argument("--mode", choices=["standalone", "sweep", "walkforward"], default="standalone")
    ap.add_argument("--symbols", nargs="*", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--stop-mult", type=float, default=float(cfg.require("risk.atr_stop_mult")))
    ap.add_argument("--no-trailing", action="store_true")
    ap.add_argument("--strategies", nargs="*", default=None,
                    help="이 전략만 사용한다 (기본: 등록된 전부)")
    ap.add_argument("--fee-rate", type=float, default=None,
                    help="수수료 override. 메이커 지정가라면 0.0002")
    ap.add_argument("--adaptive-exit", action="store_true",
                    help="레짐별 손익비를 학습 구간에서 뽑아 검증에 적용한다")
    ap.add_argument("--no-gate", action="store_true",
                    help="레짐 게이팅을 끈다 (게이팅이 값을 하는지 보는 대조군)")
    ap.add_argument("--reward-ratio", type=float, default=float(cfg.require("risk.reward_ratio")))
    ap.add_argument("--sweep-intervals", nargs="*", default=["5m", "15m", "1h"])
    ap.add_argument("--sweep-mults", nargs="*", type=float, default=[1.5, 2.5, 4.0])
    args = ap.parse_args()

    print(f"{cfg.require('market')} · {cfg.require('backtest.start')} → {cfg.require('backtest.end')}")
    print(f"수수료 {cfg.require('backtest.fee_rate')*100:.3f}% · "
          f"슬리피지 {cfg.require('backtest.slippage_bps')}bp · "
          f"왕복 {Costs.from_config(cfg).round_trip_rate()*100:.3f}%")

    return {"standalone": mode_standalone, "sweep": mode_sweep,
            "walkforward": mode_walkforward}[args.mode](args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
