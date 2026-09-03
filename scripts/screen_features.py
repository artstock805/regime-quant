"""
특징 예측력 스크리닝.

    python scripts/screen_features.py --interval 1h

전략을 새로 쓰기 전에 답해야 하는 질문: **이 특징들에 방향 정보가 있는가.**
없으면 어떤 전략을 짜도 안 된다.

정보계수(IC, Information Coefficient) = 특징값과 미래수익률의 스피어만 순위상관.
순위상관을 쓰는 이유는 이상치에 둔감하고, 특징의 스케일이 서로 달라도 비교되기 때문이다.

읽는 법
  |IC| < 0.01   잡음. 이걸로 전략을 짜면 안 된다.
  |IC| ~ 0.02   약하지만 실재할 수 있음. 비용을 넘길지는 별개.
  |IC| > 0.05   암호화폐 단기에서는 강한 편. 의심부터 해야 한다(미래참조 확인).

미래수익률은 평가에만 쓴다. 신호로 쓰면 그게 미래 참조다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                      # noqa: E402
from src.data.cache import get_klines                    # noqa: E402
from src.data.derivatives import (                       # noqa: E402
    ALL_FIELDS, align_to_bars, get_derivatives,
)
from src.indicators import all_specs, by_category, compute_all   # noqa: E402
from src.regime import KOREAN, Regime, classify          # noqa: E402

pd.set_option("display.width", 240)

# 가격 그 자체에 가까워 IC가 자동으로 높게 나오는 열은 제외한다.
# 이걸 남기면 '예측력'이 아니라 '가격과 같은 것'을 측정하게 된다.
LEVEL_LIKE = {
    "dc_upper", "dc_lower", "dc_mid", "kc_upper", "kc_lower",
    "bb_upper", "bb_lower", "st_line", "psar", "tenkan", "kijun",
    "obv", "atr", "momentum", "macd", "macd_signal", "awesome_osc", "kst",
}


def spearman_ic(feature: pd.Series, fwd: pd.Series) -> float:
    """스피어만 순위상관. 유효 표본이 적으면 NaN."""
    ok = feature.notna() & fwd.notna()
    if ok.sum() < 500:
        return np.nan
    a = feature[ok].rank()
    b = fwd[ok].rank()
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return np.nan
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--horizons", nargs="*", type=int, default=[1, 4, 12, 24])
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--derivatives", action="store_true",
                    help="미결제약정·펀딩·롱숏비율을 붙여서 함께 채점한다")
    ap.add_argument("--deriv-dir", default=None)
    args = ap.parse_args()
    if args.deriv_dir is None:
        args.deriv_dir = cfg.path("paths.derivatives_dir")

    kd = cfg.path("paths.klines_dir")
    start, end = cfg.require("backtest.start"), cfg.require("backtest.end")

    parts = []
    for sym in args.symbols:
        df = get_klines(sym, args.interval, start, end,
                        market=cfg.require("market"), klines_dir=kd)
        if args.derivatives:
            d = get_derivatives(sym, start, end, market=cfg.require("market"),
                                deriv_dir=args.deriv_dir)
            df = align_to_bars(df, d, args.interval)
            # **같은 행 위에서 채점해야 한다.** 파생은 BTC 2021-01, 나머지 2021-12
            # 부터라 그 이전 봉을 남겨 두면 캔들 특징만 더 긴 표본으로 채점돼
            # 비교가 성립하지 않는다.
            #
            # 게이트는 OI 로 잡는다. OI 는 전 구간 결측이 0% 인 유일한 필드다.
            # 모든 필드를 요구하면 2022년이 통째로 날아간다 — 그해 아카이브에
            # 상위트레이더 비율이 87% 비어 있기 때문이다(바이낸스 쪽 구멍).
            if "open_interest" in df.columns:
                df = df[df["open_interest"].notna()]
        feats = compute_all(df)
        feats = feats.drop(columns=[c for c in feats.columns if c in LEVEL_LIKE])
        reg = classify(compute_all(df))["regime"]
        for h in args.horizons:
            # 미래수익률. 신호로 쓰면 미래 참조지만, 평가 지표로는 이게 정답 라벨이다.
            fwd = df["close"].shift(-h) / df["close"] - 1.0
            block = feats.copy()
            block["__fwd"] = fwd
            block["__h"] = h
            block["__regime"] = reg
            block["__symbol"] = sym
            parts.append(block)
        print(f"  {sym} {len(df):,}봉", flush=True)

    data = pd.concat(parts)
    cols = [c for c in data.columns if not c.startswith("__")]
    print(f"\n특징 {len(cols)}개 · 지평 {args.horizons} · {len(args.symbols)}종목\n")

    # ── 전체 IC
    rows = []
    for h in args.horizons:
        sub = data[data["__h"] == h]
        for c in cols:
            rows.append({"특징": c, "지평": h, "IC": spearman_ic(sub[c], sub["__fwd"])})
    n_valid = {c: int((data[c].notna() & data["__fwd"].notna()).sum()) // len(args.horizons)
               for c in cols}
    ic = pd.DataFrame(rows).pivot(index="특징", columns="지평", values="IC")
    ic["절대최대"] = ic.abs().max(axis=1)
    ic = ic.sort_values("절대최대", ascending=False)

    print(f"━━━ 상위 {args.top}개 (|IC| 최대 기준) ━━━")
    print(ic.head(args.top).round(4).to_string())

    print(f"\n━━━ 분포 요약 ━━━")
    strong = ic[ic["절대최대"] >= 0.05]
    weak = ic[(ic["절대최대"] >= 0.02) & (ic["절대최대"] < 0.05)]
    noise = ic[ic["절대최대"] < 0.01]
    print(f"  |IC| >= 0.05 (강함, 미래참조 의심) : {len(strong)}개  {list(strong.index[:8])}")
    print(f"  0.02 ~ 0.05  (약하지만 실재 가능)  : {len(weak)}개  {list(weak.index[:8])}")
    print(f"  |IC| <  0.01 (잡음)               : {len(noise)}개 / {len(ic)}개")

    # ── 오더플로 특징만 따로
    flow = [c for c in cols if c in {
        "taker_imbalance", "taker_imbalance_z", "trades_zscore",
        "avg_trade_size_z", "vol_zscore", "cmf", "mfi", "obv_slope",
        "volume_ratio", "force_index", "ad_line", "pvt", "eom",
    }]
    print("\n━━━ 오더플로·거래량 계열 IC ━━━")
    print(ic.loc[[c for c in flow if c in ic.index]].round(4).to_string())

    # ── 파생 계열만 따로. 이게 --derivatives 실험의 본론이다.
    owners = {}
    for name in by_category().get("derivatives", []):
        for out_col in all_specs()[name].output_names:
            owners[out_col] = name
    deriv_cols = [c for c in ic.index if c in owners]
    if deriv_cols:
        print("\n━━━ 파생 시장 계열 IC (캔들 밖 정보원) ━━━")
        tbl = ic.loc[deriv_cols].round(4).copy()
        tbl["표본"] = [n_valid.get(c, 0) for c in tbl.index]
        print(tbl.to_string())

        kline_cols = [c for c in ic.index if c not in owners]
        best_d = float(ic.loc[deriv_cols, "절대최대"].max())
        best_k = float(ic.loc[kline_cols, "절대최대"].max())
        print(f"\n  파생 최고 |IC| {best_d:.4f}   캔들 최고 |IC| {best_k:.4f}")
        if best_d > best_k:
            print("  → 파생이 캔들을 넘었다. 캔들 밖에 정보가 있다는 뜻이다.")
        else:
            print("  → 파생이 캔들을 넘지 못했다. 이 방향으로는 우위가 안 나온다.")
        print("  주의: 2022년 아카이브에 상위트레이더 87%, 테이커 35% 가 비어 있다"
              " — 표본 열을 함께 볼 것.")

    # ── 레짐별 IC (가장 유망한 특징들만)
    top_names = list(ic.head(8).index)
    print(f"\n━━━ 레짐별 IC (상위 8개 특징, 지평 {args.horizons[1]}봉) ━━━")
    sub = data[data["__h"] == args.horizons[1]]
    reg_rows = []
    for regime in [str(r) for r in Regime]:
        block = sub[sub["__regime"] == regime]
        if len(block) < 1000:
            continue
        row = {"레짐": KOREAN[Regime(regime)], "표본": len(block)}
        for c in top_names:
            row[c] = spearman_ic(block[c], block["__fwd"])
        reg_rows.append(row)
    print(pd.DataFrame(reg_rows).round(4).to_string(index=False))

    print("\n판단: 상위 |IC| 가 0.02 를 못 넘으면 이 특징 집합에는 방향 정보가 거의 없다.")
    print("      전략을 더 짜는 대신 특징 자체를 바꿔야 한다(호가/청산/펀딩 등).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
