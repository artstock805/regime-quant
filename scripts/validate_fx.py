"""
FX 에 우위가 있는가 — 캔들만 쓰는 전략들을 브로커 호가로 검증한다.

**왜 처음부터 다시 하는가.** 이 프로젝트가 암호화폐에서 찾은 우위는
perp_taker_ls(바이낸스 무기한 테이커 매수/매도 비율)에 기대고 있었다. FX 에는
그런 데이터가 없다. 그러니 그 결과는 여기로 넘어오지 않는다. 캔들만 쓰는
전략들을 FX 봉에 처음부터 돌려 봐야 한다.

**비용을 실제 스프레드로 문다.** 봉마다 그때의 스프레드가 함께 저장돼 있다.
지금 스프레드로 과거를 평가하면 비용을 과소평가한다 — 뉴스 시간과 아시아
새벽에는 몇 배로 벌어진다. 암호화폐에서 총우위 t=3.52 가 비용을 빼면 t=1.42 가
됐다. 비용을 낙관적으로 잡으면 그 결론 자체를 놓친다.

**t 값을 종목 수로 부풀리지 않는다.** FX 는 상관이 극단적으로 높다. EUR/USD,
GBP/USD, AUD/USD 는 사실상 달러 하나를 세 번 세는 것이고, 엔 크로스끼리도
같이 움직인다. 순진한 t검정은 이걸 독립 관측으로 세어 √N 배까지 부풀린다.
시간 군집(주 단위)으로 묶어 검정한다.

    python scripts/validate_fx.py
    python scripts/validate_fx.py --interval 1h --gross      비용 없이 (대조군)

먼저 python scripts/fetch_fx.py 로 데이터를 받아 두세요.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.engine import ExecParams  # noqa: E402
from src.backtest.walkforward import run_standalone  # noqa: E402
from src.config import Costs, load_config  # noqa: E402
from src.indicators import compute_all  # noqa: E402
from src.regime import classify  # noqa: E402
from src.strategies import build_all  # noqa: E402
from src.validation.stats import (  # noqa: E402
    clustered_mean_test, holm, time_clusters,
)

CACHE = ROOT / "data" / "fx"


def load(interval: str) -> dict[str, pd.DataFrame]:
    paths = sorted(CACHE.glob(f"*_{interval}.parquet"))
    out = {}
    for p in paths:
        name = p.stem.rsplit("_", 1)[0]
        out[name] = pd.read_parquet(p)
    return out


def kline_only_strategies():
    """
    파생 데이터가 필요 없는 전략만 고른다.

    FX 에는 무기한 선물 자금조달률도, 미결제약정도, 테이커 비율도 없다.
    그것을 요구하는 전략은 여기서 돌릴 수 없고, 억지로 0 을 채우면 검증이 아니라
    거짓말이 된다.
    """
    kline = {"open", "high", "low", "close", "volume"}
    ok = []
    for s in build_all():
        needs = set(getattr(s, "needs", ()))
        # 지표 열은 compute_all 이 캔들에서 만들어 준다. 파생 열만 걸러 낸다.
        if any(n.startswith(("perp_", "funding", "oi_", "basis")) for n in needs):
            continue
        ok.append(s)
    return ok


def costs_for(df: pd.DataFrame, gross: bool) -> Costs:
    """
    이 통화쌍의 실제 비용. **스프레드가 전부다** — 표준 계좌에는 별도 수수료가 없다.

    엔진의 fee_rate 는 편도 비율이다. 스프레드를 한 번 건너는 것이 편도이므로
    편도 = 스프레드/2 (중간값 기준). 왕복이면 스프레드 하나가 온전히 나간다.
    슬리피지는 그 위에 얹는다 — 지정가가 아니라 시장가로 들어가기 때문이다.
    """
    if gross:
        return Costs(fee_rate=0.0, slippage_bps=0.0)
    half_spread = float(df["spread_rate"].median()) / 2.0
    return Costs(fee_rate=half_spread, slippage_bps=0.5)


def main() -> int:
    ap = argparse.ArgumentParser(description="FX 우위 검증")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--gross", action="store_true",
                    help="비용을 빼지 않는다 (대조군 — 우위가 애초에 있는지)")
    ap.add_argument("--stop-mult", type=float, default=None,
                    help="ATR 손절 계수. 기본은 config 값")
    args = ap.parse_args()

    data = load(args.interval)
    if not data:
        print(f"{CACHE} 에 {args.interval} 캐시가 없습니다.\n"
              f"  python scripts/fetch_fx.py --interval {args.interval} --crosses")
        return 1

    cfg = load_config()
    params = ExecParams(
        risk_per_trade=float(cfg.require("risk.risk_per_trade")),
        atr_stop_mult=float(args.stop_mult if args.stop_mult is not None
                            else cfg.require("risk.atr_stop_mult")),
        reward_ratio=float(cfg.require("risk.reward_ratio")),
        daily_loss_limit=float(cfg.require("risk.daily_loss_limit")),
    )
    strategies = kline_only_strategies()

    mode = "총우위 (비용 제외)" if args.gross else "순우위 (실제 스프레드 차감)"
    print(f"FX 검증 · {args.interval} · {len(data)}쌍 · 전략 {len(strategies)}개")
    print(f"{mode} · ATR 손절 {params.atr_stop_mult}x · 손익비 {params.reward_ratio}\n")

    rows: list[dict] = []
    for symbol, df in data.items():
        feats = compute_all(df)
        # classify 는 regime/confidence/bars_in/raw 를 담은 표를 준다.
        # 백테스터가 원하는 것은 국면 열 하나다.
        regimes = classify(feats)["regime"]
        costs = costs_for(df, args.gross)
        results = run_standalone(
            df, feats, strategies, interval=args.interval, costs=costs,
            params=params, initial_equity=10_000.0, minute_df=None,
            regimes=regimes, symbol=symbol,
        )
        for name, res in results.items():
            t = res.trades
            if t.empty:
                continue
            for _, tr in t.iterrows():
                rows.append({"symbol": symbol, "strategy": name,
                             "r": tr.get("r_multiple", np.nan),
                             "exit_time": tr.get("exit_time")})
        spread_bp = df["spread_rate"].median() * 10_000
        n = sum(len(r.trades) for r in results.values())
        print(f"  {symbol:8s} 거래 {n:>6,}  스프레드 {spread_bp:.2f}bp")

    if not rows:
        print("\n거래가 하나도 나오지 않았습니다.")
        return 1

    trades = pd.DataFrame(rows).dropna(subset=["r"])
    print(f"\n총 거래 {len(trades):,}건\n")

    # ─── 전략별 판정 ──────────────────────────────────────────────
    #
    # 군집은 시간(주)으로 잡는다. 같은 주에 14쌍에서 열린 거래는 사실상 하나의
    # 시장 국면을 14번 센 것에 가깝다. 종목 수로 t 를 만들어내는 것이 그 착시다.

    out, pvals = [], {}
    for name, g in trades.groupby("strategy"):
        r = g["r"].to_numpy("float64")
        cl = time_clusters(pd.to_datetime(g["exit_time"]), freq="W")
        test = clustered_mean_test(r, cl)
        pvals[name] = test.p
        out.append({
            "전략": name,
            "거래": len(r),
            "기대값(R)": round(float(r.mean()), 4),
            "승률": f"{(r > 0).mean() * 100:.1f}%",
            "군집": test.n_clusters,
            "t(순진)": round(test.t_naive, 2),
            "t(군집)": round(test.t, 2),
            "p": round(test.p, 4),
            "판정": test.verdict,
        })

    table = pd.DataFrame(out).sort_values("기대값(R)", ascending=False)

    # 전략 6개를 동시에 보므로 다중검정 보정이 필요하다. 여섯 번 던져서
    # 하나가 p<0.05 로 나오는 것은 우연히도 흔하다.
    adj = holm(pvals)
    table["p(Holm)"] = table["전략"].map(lambda n: round(adj[n], 4))
    table["Holm 유의"] = table["p(Holm)"] < 0.05

    print(table.to_string(index=False))

    # ─── 전체 ────────────────────────────────────────────────────
    r_all = trades["r"].to_numpy("float64")
    cl_all = time_clusters(pd.to_datetime(trades["exit_time"]), freq="W")
    whole = clustered_mean_test(r_all, cl_all)
    print(f"\n전체 합산  기대값 {whole.mean:+.4f}R  "
          f"t(순진) {whole.t_naive:.2f} → t(군집) {whole.t:.2f}  "
          f"p={whole.p:.4f}  [{whole.verdict}]")
    print(f"  군집 {whole.n_clusters}개 · 유효표본 {whole.effective_n:.0f} "
          f"(실제 {whole.n:,}) · SE 를 {whole.inflation:.1f}배로 키웠다")

    if not table["Holm 유의"].any():
        print("\n판정: **Holm 보정 후 유의한 전략이 없다.**")
        print("  이 신호로 자동매매를 돌리면 비용만 내게 된다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
