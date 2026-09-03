"""
전략별 우위의 통계적 유의성 검정.

    python scripts/significance.py --interval 1h --stop-mult 4.0

`validate.py` 는 **종목별**로 본다. 이 스크립트는 **전략별**로 본다 — 어느 전략에
실재하는 우위가 있고 어느 전략이 손해인지를 가른다. 단위가 다르므로 둘 다 필요하다.

통계 함수는 `src.validation.stats` 하나에서 가져온다. 같은 t검정이 두 곳에 있으면
언젠가 갈라지고, 그러면 어느 쪽 숫자를 믿어야 할지 알 수 없게 된다.

읽는 법
  순기대R  비용 차감 후. 실전 투입 근거는 이것뿐이다.
  총기대R  비용 제외. 여기가 양수인데 순기대R이 음수면 '우위가 비용보다 작다'는
           뜻이고, 처방이 "더 좋은 전략"이 아니라 "비용 절감"이 된다.
  필요거래수  관측된 효과를 t=1.96 으로 만들 표본 수. 현실적으로 도달 불가능하면
             그 전략은 '안 통한다'가 아니라 '통하는지 증명할 수 없다'가 정확하다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Costs, load_config                     # noqa: E402
from src.data.cache import get_klines                          # noqa: E402
from src.data.derivatives import align_to_bars, get_derivatives  # noqa: E402
from src.indicators import compute_all                         # noqa: E402
from src.regime import classify                                # noqa: E402
from src.strategies import build_all                           # noqa: E402
from src.backtest import ExecParams, run_standalone            # noqa: E402
from src.validation.stats import t_test                        # noqa: E402

pd.set_option("display.width", 240)


def _row(name: str, x) -> dict:
    st = t_test(x.to_numpy())
    return {
        "전략": name, "거래": st.n,
        "기대R": None if st.n < 20 else round(st.mean, 4),
        "표준편차": None if st.n < 20 else round(st.sd, 3),
        "t": None if st.n < 20 else round(st.t, 2),
        "CI하한": None if st.n < 20 else round(st.ci_lo, 4),
        "CI상한": None if st.n < 20 else round(st.ci_hi, 4),
        "판정": st.verdict,
        "필요거래수": st.trades_needed(),
    }


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="전략별 유의성 검정")
    ap.add_argument("--symbols", nargs="*", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--stop-mult", type=float,
                    default=float(cfg.require("risk.atr_stop_mult")))
    ap.add_argument("--reward-ratio", type=float,
                    default=float(cfg.require("risk.reward_ratio")))
    ap.add_argument("--derivatives", action="store_true",
                    help="파생 데이터를 붙인다. 파생 전략은 이게 없으면 터진다")
    ap.add_argument("--deriv-dir", default=None)
    ap.add_argument("--all-strategies", action="store_true",
                    help="config 의 strategies 목록을 무시하고 등록된 전부를 본다")
    args = ap.parse_args()
    if args.deriv_dir is None:
        args.deriv_dir = cfg.path("paths.derivatives_dir")

    costs = Costs.from_config(cfg)
    params = ExecParams(
        risk_per_trade=float(cfg.require("risk.risk_per_trade")),
        atr_stop_mult=args.stop_mult, reward_ratio=args.reward_ratio,
        daily_loss_limit=float(cfg.require("risk.daily_loss_limit")),
    )
    names = None if args.all_strategies else cfg.get("strategies")
    strats = [s for s in build_all() if not names or s.name in set(names)]
    km = {s.name: s.korean for s in strats}
    kd = cfg.path("paths.klines_dir")
    start, end = cfg.require("backtest.start"), cfg.require("backtest.end")

    print(f"{args.interval} · 손절 {args.stop_mult}xATR · 손익비 {args.reward_ratio}:1")
    print(f"전략 {[s.korean for s in strats]}")
    print(f"{', '.join(args.symbols)} · {start} → {end}\n")

    frames = []
    for sym in args.symbols:
        df = get_klines(sym, args.interval, start, end,
                        market=cfg.require("market"), klines_dir=kd)
        if args.derivatives:
            d = get_derivatives(sym, start, end, market=cfg.require("market"),
                                deriv_dir=args.deriv_dir)
            df = align_to_bars(df, d, args.interval)
            # 파생이 없는 앞 구간을 남기면 파생 전략이 조용히 발언권을 잃고,
            # 캔들 전략만 더 긴 표본으로 채점돼 비교가 성립하지 않는다.
            df = df[df["open_interest"].notna()]
        feats = compute_all(df)
        mn = get_klines(sym, "1m", start, end,
                        market=cfg.require("market"), klines_dir=kd)
        res = run_standalone(df, feats, strats, interval=args.interval, costs=costs,
                             params=params, initial_equity=10_000, minute_df=mn,
                             regimes=classify(feats)["regime"], symbol=sym)
        frames.append(pd.concat([r.trades for r in res.values()], ignore_index=True))
        print(f"  {sym} 완료", flush=True)
    trades = pd.concat(frames, ignore_index=True)

    print("\n━━━ 순기대R (비용 차감 후 — 실전 근거) ━━━")
    net = pd.DataFrame([_row(km.get(n, n), g["r_multiple"])
                        for n, g in trades.groupby("strategy")])
    print(net.sort_values("기대R", ascending=False, na_position="last").to_string(index=False))

    print("\n━━━ 총기대R (비용 제외 — 우위 자체의 유무) ━━━")
    gross = pd.DataFrame([_row(km.get(n, n), g["gross_r"])
                          for n, g in trades.groupby("strategy")])
    print(gross.sort_values("기대R", ascending=False, na_position="last").to_string(index=False))

    print("\n━━━ 전체 합산 ━━━")
    for label, col in (("순", "r_multiple"), ("총", "gross_r")):
        st = t_test(trades[col].to_numpy())
        print(f"  {label}  {st.mean:+.4f}R  t={st.t:+.2f}  p={st.p:.4f}  "
              f"CI[{st.ci_lo:+.4f}, {st.ci_hi:+.4f}]  → {st.verdict}")
    print(f"  비용 {trades['cost_r'].mean():.4f}R/거래")

    sig = net[net["판정"] == "양의우위"]
    print(f"\n순기대R이 통계적으로 0보다 큰 전략: {len(sig)}개 / {len(net)}개")
    if sig.empty:
        print("  → 어떤 전략도 비용 후 우위를 입증하지 못했다. 실전 투입 근거가 없다.")
    else:
        print(sig[["전략", "거래", "기대R", "t", "CI하한"]].to_string(index=False))

    gsig = gross[gross["판정"] == "양의우위"]
    if not gsig.empty and sig.empty:
        print(f"\n다만 총기대R은 {len(gsig)}개 전략에서 유의하다 — "
              "우위는 실재하고 비용이 그것을 먹는다는 뜻이다:")
        print(gsig[["전략", "거래", "기대R", "t", "CI하한"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
