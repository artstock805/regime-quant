"""
MT5 종목(코인 제외)에서 지표 79종의 예측력을 잰다 — 전략을 좁히기 위한 첫 단계.

**왜 전략이 아니라 지표부터인가.**
    전략 × 국면 × 시간축을 전수로 돌리면 132 가설이 되고, 그중 최고를 고르면
    우연히 좋은 것을 고르게 된다. 반면 지표는 **전략보다 앞선 층**이다 —
    지표에 방향 정보가 없으면 그 지표로 짠 어떤 전략도 안 된다. 그래서 지표를
    먼저 걸러 탐색 공간을 줄인다.

    기존 `screen_features.py` 는 코인 클라인만 읽는다. 여기서는 MT5 봉을 쓴다.

**읽는 법** (기존 스크리너의 기준을 그대로 따른다)
    |IC| < 0.01   잡음. 이걸로 전략을 짜면 안 된다.
    |IC| ~ 0.02   약하지만 실재할 수 있음. 비용을 넘길지는 별개.
    |IC| > 0.05   의심부터 해야 한다 — 미래 참조나 가격 그 자체가 아닌지.

**지표 하나의 IC 만 보면 안 되는 이유가 둘.**
    1. 79개를 재면 우연히 |IC| 가 큰 것이 나온다. 그래서 **종목 간 부호
       일치**를 같이 본다 — 한 종목에서 큰 것보다 25종목에서 같은 부호인
       것이 훨씬 믿을 만하다.
    2. IC 가 커도 **이미 아는 것의 재표현**일 수 있다. 직전수익률(단기반전)과의
       상관을 같이 내서, 새 정보인지 확인한다. 오더플로 대용에서 이 함정에
       걸린 적이 있다(IC 는 컸는데 2/3 이 반전이었다).

사용법
    python scripts/screen_mt5_features.py --interval 1d
    python scripts/screen_mt5_features.py --interval 4h --horizons 1 3 5
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

from src.fx.mt5 import bars                                         # noqa: E402
from src.indicators import compute_all                              # noqa: E402
from src.indicators.core import zscore                              # noqa: E402
from src.regime import classify                                     # noqa: E402

pd.set_option("display.width", 240)

# 가격 그 자체에 가까워 IC 가 자동으로 높게 나오는 열. 남기면 '예측력'이 아니라
# '가격과 같은 것'을 재게 된다. 기존 스크리너와 같은 목록을 쓴다.
LEVEL_LIKE = {
    "dc_upper", "dc_lower", "dc_mid", "kc_upper", "kc_lower",
    "bb_upper", "bb_lower", "st_line", "psar", "tenkan", "kijun",
    "obv", "atr", "momentum", "macd", "macd_signal", "awesome_osc", "kst",
}

SYMBOLS = [
    "US500Cash", "US100Cash", "US30Cash", "GER40Cash", "UK100Cash",
    "JP225Cash", "FRA40Cash", "EU50Cash", "AUS200Cash", "HK50Cash",
    "IT40Cash", "NETH25Cash", "SWI20Cash", "ChinaHCash", "US2000Cash",
    "GOLD", "SILVER", "XAUEUR", "OILCash", "BRENTCash",
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "GBPJPY", "EURAUD", "CADJPY", "AUDNZD", "GBPAUD",
]

BARS = {"1h": 20000, "4h": 17000, "1d": 8000}


def ic(feature: pd.Series, fwd: pd.Series) -> float:
    """스피어만 순위상관. 표본이 적으면 NaN — 적은 표본의 큰 IC 는 잡음이다."""
    ok = feature.notna() & fwd.notna()
    if ok.sum() < 400:
        return np.nan
    a, b = feature[ok].rank(), fwd[ok].rank()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))


def screen_symbol(sym: str, interval: str, horizons: list[int]) -> pd.DataFrame:
    df = bars(sym, interval, BARS[interval])
    if len(df) < 800:
        return pd.DataFrame()
    feats = compute_all(df)
    feats = feats[[c for c in feats.columns if c not in LEVEL_LIKE]]
    # 단기반전 기준선. 새 정보인지 판단하려면 이것과 비교해야 한다.
    rev = zscore(df["close"].pct_change(), 100)
    regime = classify(feats)["regime"] if "atr" in feats.columns else None

    rows = []
    for h in horizons:
        fwd = df["close"].pct_change(h).shift(-h)
        for col in feats.columns:
            v = ic(feats[col], fwd)
            if not np.isfinite(v):
                continue
            rows.append({
                "지표": col, "지평": h, "종목": sym, "IC": v,
                "vs반전": feats[col].corr(rev, method="spearman"),
            })
    out = pd.DataFrame(rows)
    if not out.empty and regime is not None:
        out["고변동비율"] = float((regime == "high_vol").mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--horizons", nargs="*", type=int, default=[1, 3, 5])
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    frames = []
    for sym in SYMBOLS:
        try:
            r = screen_symbol(sym, args.interval, args.horizons)
        except Exception as exc:
            print(f"  {sym:14s} 실패 {type(exc).__name__}", flush=True)
            continue
        if r.empty:
            print(f"  {sym:14s} 건너뜀", flush=True)
            continue
        frames.append(r)
        print(f"  {sym:14s} 지표 {r['지표'].nunique()}개", flush=True)
    if not frames:
        return 1

    all_ = pd.concat(frames, ignore_index=True)
    n_sym = all_["종목"].nunique()

    print()
    print("=" * 96)
    print(f"지표 스크리닝 · {args.interval} · {n_sym}종목 · "
          f"지평 {args.horizons}")
    print("=" * 96)

    g = all_.groupby(["지표", "지평"]).agg(
        종목=("종목", "nunique"),
        IC평균=("IC", "mean"),
        IC중위=("IC", "median"),
        음수=("IC", lambda x: int((x < 0).sum())),
        vs반전=("vs반전", "mean"))
    g["부호일치"] = g[["종목", "음수"]].apply(
        lambda r: max(r["음수"], r["종목"] - r["음수"]) / r["종목"], axis=1)
    # 부호검정: 종목 간 부호가 한쪽으로 몰렸나. 한 종목의 큰 IC 보다 이게 낫다.
    g["부호p"] = g[["종목", "음수"]].apply(
        lambda r: st.binomtest(int(max(r["음수"], r["종목"] - r["음수"])),
                               int(r["종목"]), 0.5).pvalue, axis=1)
    g["|IC|"] = g["IC평균"].abs()
    g = g.reset_index()

    print(f"\n── |IC평균| 상위 {args.top} ──")
    top = g.sort_values("|IC|", ascending=False).head(args.top)
    cols = ["지표", "지평", "종목", "IC평균", "IC중위", "부호일치", "부호p",
            "vs반전"]
    print(top[cols].round(4).to_string(index=False))

    print("\n── 부호가 가장 일관된 것들 (종목 간 일치율 기준) ──")
    consistent = g[g["|IC|"] > 0.01].sort_values(
        ["부호p", "|IC|"], ascending=[True, False]).head(args.top)
    print(consistent[cols].round(4).to_string(index=False))

    # **새 정보인가.** 반전과 상관이 낮으면서 IC 가 있는 것이 값지다.
    fresh = g[(g["|IC|"] > 0.015) & (g["vs반전"].abs() < 0.25)]
    print(f"\n── 단기반전과 겹치지 않는 것들 (|vs반전|<0.25, |IC|>0.015) "
          f"— {len(fresh)}개 ──")
    if fresh.empty:
        print("  없음 — IC 가 있는 지표가 전부 반전의 재표현이라는 뜻이다")
    else:
        print(fresh.sort_values("|IC|", ascending=False)[cols]
              .round(4).to_string(index=False))

    m = len(g)
    print(f"\n가설 {m}개 (지표 × 지평). Bonferroni 문턱 p<{0.05/m:.6f}")
    print(f"  부호검정 통과: {int((g['부호p'] < 0.05/m).sum())}개")
    print()
    print("판단: |IC|<0.01 은 잡음이다. 부호가 종목 간 일관되고(부호p 작고),")
    print("      단기반전과 겹치지 않는 지표만 전략 후보로 삼는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
