"""
차트에서 그럴듯해 보이는 지표들을 '미래 확정' 기준으로 한 번에 거른다.

**거르는 기준은 하나다: 그 지표를 그리려면 오른쪽 봉이 필요한가.**

    피봇 고점은 "그 뒤 right 봉 동안 더 높은 값이 없었다"를 확인해야 피봇이 된다.
    그런데 차트는 그 표식을 **right 봉 전 캔들 위에** 그린다. 사람이 "여기서
    신호 떴네" 하고 보는 자리에서는, 실시간이었다면 아직 아무 표시도 없다.

    지그재그 · 프랙탈 · 다이버전스 · 하모닉 패턴이 전부 이 구조다.

**앞선 측정(probe_pivot_divergence.py)에서 빠졌던 것.**
    거기서는 같은 피봇 집합을 '피봇 자리'와 '확정 자리'에 각각 찍어 비교했다.
    즉 **신호 집합은 같고 진입 시점만 다르다.** 그러면 "미래를 봤기 때문"인지
    "그냥 늦게 들어가서"인지 가를 수 없다. 둘 다 같은 기전이긴 하지만, 반론은
    남는다 — "5봉 늦추면 어떤 신호든 무너지는 것 아니냐."

    그래서 여기서는 **인과적 대조군**을 같이 돌린다. 미리 볼 것이 없는 신호를
    똑같이 5봉 늦춰서, 지연 그 자체의 비용을 따로 잰다.

        reversal_z   직전 1봉 수익률 z-score. 이 프로젝트가 **실재한다고 확인한
                     거의 유일한 효과**다(오더플로 측정에서 "IC 의 2/3 이 반전의
                     재표현"). 효과가 있는 신호라 지연 비용을 잴 수 있다.
        rsi_cross    RSI 30 상향 / 70 하향 돌파. 봉이 닫히면 바로 안다.

    대조군의 지연 손실이 작은데 피봇 계열만 무너지면, 원인은 지연이 아니라
    **뒤늦게야 알 수 있는 극값을 미리 아는 척한 것**이다.

**확정 지연이 커질수록 괴리도 커져야 한다.** 그래서 right 를 2·5·10 으로 다르게
가져간 지표를 섞어 둔다. 괴리가 지연에 비례하면 기전이 확인된 것이다.

**방향차 = (강세 신호 뒤 수익률) − (약세 신호 뒤 수익률), 단위 bp.**
종목의 추세 편향은 양쪽에 똑같이 실리므로 빼기 한 번으로 상쇄된다.

**측정 결과 (2026-10-20, 1시간봉, FX 12 + 지수 10종목, 지평 5).**

    FX          확정지연   차트bp  실제bp   종목일치(차트→실제)
    ───────────────────────────────────────────────────────
    RSI돌파      0(인과)     0.12    0.12    8/12 → 8/12
    반전z        0(인과)     0.57    0.57    7/12 → 7/12
    프랙탈          2       20.56   -0.47   **12/12 → 1/12**
    다이버전스MACD   5       29.75    0.25   **12/12 → 6/12**
    다이버전스RSI    5       32.25    0.60   **12/12 → 6/12**
    지그재그        10       44.97    0.62   **12/12 → 8/12**

    **미래확정 지표 넷 모두 차트에서는 12/12 종목이 양수다.** 완벽한 일치.
    확정 자리로 옮기면 1~8/12 — 동전 던지기가 된다.

**결정적 증거 둘.**

    (1) 사라진 몫이 확정지연에 비례한다.  2봉→21bp · 5봉→30bp · 10봉→44bp.
        미리 보는 창이 넓을수록 좋아 보인다. 실재하는 신호에는 없는 성질이다.

    (2) **우위는 각 지표의 확정 경계에서 정확히 사라진다. 지연에 비례해 닳는
        것이 아니다.** 이게 '단순 지연 탓'이라는 반론을 끝낸다.

            지그재그(확정 10봉)  0봉 44.97 → 5봉 12.62 → 10봉 0.62
            프랙탈(확정 2봉)     0봉 20.56 → 2봉 -0.47 →  5봉 -0.12

        프랙탈은 **2봉만 늦춰도 이미 끝난다.** 거기서 3봉을 더 늦춰도 거의
        변하지 않는다(-0.47 → -0.12). 지연 자체가 원인이라면 더 늦출수록 더
        나빠져야 하는데 그러지 않는다. 지그재그도 제 확정 지연(10봉)에 닿는
        순간 0 이 된다. **손실은 '미리 보기를 멈추는 지점'에서 일어난다.**

    대조군은 지연 0 이라 사라진 몫도 0 이다. 다만 대조군의 우위 자체가
    0.1~0.6bp(유의하지 않음)라 '진짜 신호의 지연 비용'을 재지는 못했다 —
    이 시장에 그만한 인과적 신호가 없기 때문이다. 그래서 판단은 위의 (1)(2)에
    기댄다. 그쪽이 더 강한 증거다.

**덤으로 나온 것.** 정직한 시점에서는 오히려 살짝 거꾸로인 칸이 있었다 —
지수 다이버전스MACD 지평5 가 -3.79bp 에 0/9 종목(p=0.004), FX 프랙탈 지평5 가
-0.47bp 에 1/12(p=0.006). 다만 둘 다 FX 왕복 스프레드(1.6~3.3bp)보다 작아
뒤집어 거래할 수 있는 크기도 아니다.

사용법
    python scripts/probe_lookahead_screen.py
    python scripts/probe_lookahead_screen.py --interval 30m --horizons 1 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scipy import stats as st                                       # noqa: E402

from probe_pivot_divergence import pivot_flags                      # noqa: E402
from src.fx.mt5 import bars                                         # noqa: E402
from src.indicators.momentum import rsi                             # noqa: E402

pd.set_option("display.width", 240)

FAMILIES = {
    "지수": ["US500Cash", "US100Cash", "US30Cash", "GER40Cash", "UK100Cash",
             "JP225Cash", "FRA40Cash", "EU50Cash", "AUS200Cash", "HK50Cash"],
    "FX": ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
           "GBPJPY", "EURAUD", "CADJPY", "AUDNZD", "GBPAUD"],
}

BARS = {"30m": 20000, "1h": 20000, "4h": 17000, "1d": 8000}

RANGE_MIN, RANGE_MAX = 5, 60
DELAY_PROBE = 5          # 대조군에 똑같이 물리는 인위적 지연


def _macd_hist(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    return macd - macd.ewm(span=9, adjust=False).mean()


def _divergence(df: pd.DataFrame, osc: pd.Series, left: int,
                right: int) -> np.ndarray:
    """피봇 자리에 찍은 다이버전스. +1 강세 / -1 약세."""
    hi, lo = pivot_flags(osc, left, right)
    o = osc.to_numpy("float64")
    ph, pl = df["high"].to_numpy("float64"), df["low"].to_numpy("float64")
    out = np.zeros(len(o), dtype="float64")
    for mask, price, cmp_price, cmp_osc, val in (
        (hi, ph, np.greater, np.less, -1.0),
        (lo, pl, np.less, np.greater, +1.0),
    ):
        prev = -1
        for i in np.flatnonzero(mask):
            if prev >= 0 and RANGE_MIN <= i - prev <= RANGE_MAX:
                if cmp_price(price[i], price[prev]) and cmp_osc(o[i], o[prev]):
                    out[i] = val
            prev = i
    return out


def _swing(df: pd.DataFrame, left: int, right: int) -> np.ndarray:
    """지그재그·프랙탈의 공통 알맹이 — 가격 피봇 자리에 전환 표식."""
    hi, lo = pivot_flags(df["close"], left, right)
    out = np.zeros(len(df), dtype="float64")
    out[lo] = +1.0          # 저점 전환 = 상승 기대
    out[hi] = -1.0
    return out


def _reversal_z(df: pd.DataFrame) -> np.ndarray:
    """직전 1봉 수익률의 z-score. **인과적이다** — 봉이 닫히면 바로 안다."""
    r = df["close"].pct_change()
    z = (r - r.rolling(100).mean()) / r.rolling(100).std(ddof=0)
    out = np.zeros(len(df), dtype="float64")
    v = z.to_numpy("float64")
    out[v < -2.0] = +1.0     # 급락 뒤 반등 기대
    out[v > +2.0] = -1.0
    return out


def _rsi_cross(df: pd.DataFrame) -> np.ndarray:
    """RSI 30 상향 / 70 하향 돌파. **인과적이다.**"""
    v = rsi(df, 14).to_numpy("float64")
    out = np.zeros(len(v), dtype="float64")
    prev, cur = v[:-1], v[1:]
    out[1:][(prev < 30) & (cur >= 30)] = +1.0
    out[1:][(prev > 70) & (cur <= 70)] = -1.0
    return out


# (이름, 신호함수, 확정지연, 계열)
#   확정지연 0 = 인과적. 그 봉에서 바로 알 수 있다.
INDICATORS = [
    ("프랙탈(2)",        lambda d: _swing(d, 2, 2),                      2, "미래확정"),
    ("다이버전스RSI(5)", lambda d: _divergence(d, rsi(d, 14), 5, 5),     5, "미래확정"),
    ("다이버전스MACD(5)", lambda d: _divergence(d, _macd_hist(d), 5, 5), 5, "미래확정"),
    ("지그재그(10)",     lambda d: _swing(d, 10, 10),                   10, "미래확정"),
    ("반전z(대조군)",    _reversal_z,                                    0, "인과"),
    ("RSI돌파(대조군)",  _rsi_cross,                                     0, "인과"),
]


def shift_signal(sig: np.ndarray, by: int) -> np.ndarray:
    if by <= 0:
        return sig
    out = np.zeros_like(sig)
    out[by:] = sig[:-by]
    return out


def direction_gap(sig: np.ndarray, fwd: pd.Series) -> tuple[float, int, int]:
    """방향차와 양쪽 표본 수. 표본이 적으면 NaN."""
    ok = fwd.notna().to_numpy()
    f = fwd.to_numpy("float64")
    bull, bear = f[ok & (sig > 0)], f[ok & (sig < 0)]
    if len(bull) < 10 or len(bear) < 10:
        return float("nan"), len(bull), len(bear)
    return float(bull.mean() - bear.mean()), len(bull), len(bear)


def measure(sym: str, interval: str, horizons: list[int]) -> list[dict]:
    df = bars(sym, interval, BARS[interval])
    if len(df) < 1500:
        return []
    rows = []
    for name, fn, lag, kind in INDICATORS:
        base = fn(df)
        for h in horizons:
            fwd = df["close"].pct_change(h).shift(-h)
            chart, nb, nr = direction_gap(base, fwd)
            real, _, _ = direction_gap(shift_signal(base, lag), fwd)
            d5, _, _ = direction_gap(shift_signal(base, DELAY_PROBE), fwd)
            if not np.isfinite(chart):
                continue
            rows.append({
                "지표": name, "종류": kind, "확정지연": lag, "지평": h,
                "종목": sym, "강세": nb, "약세": nr,
                "차트": chart, "실제": real, f"+{DELAY_PROBE}봉지연": d5,
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--horizons", nargs="*", type=int, default=[1, 5])
    args = ap.parse_args()

    rows = []
    for fam, syms in FAMILIES.items():
        for sym in syms:
            try:
                r = measure(sym, args.interval, args.horizons)
            except Exception as exc:
                print(f"  {sym:12s} 실패 {type(exc).__name__}", flush=True)
                continue
            if not r:
                print(f"  {sym:12s} 건너뜀", flush=True)
                continue
            for x in r:
                x["계열"] = fam
            rows += r
            print(f"  {fam:4s} {sym:12s} 지표 {len({x['지표'] for x in r})}개", flush=True)
    if not rows:
        return 1

    all_ = pd.DataFrame(rows)
    d5 = f"+{DELAY_PROBE}봉지연"
    print()
    print("=" * 104)
    print(f"미래 확정 선별 · {args.interval} · 종목 {all_['종목'].nunique()}")
    print("=" * 104)
    print("\n방향차 bp = (강세 뒤 수익률) − (약세 뒤 수익률). 지표가 통하면 양수.")
    print(f"'차트' = 표식이 그려지는 자리 · '실제' = 확정되는 자리 · "
          f"'{d5}' = 모두에게 똑같이 물린 인위적 지연")

    for fam in FAMILIES:
        sub = all_[all_["계열"] == fam]
        if sub.empty:
            continue
        print(f"\n-- {fam} 계열 --")
        g = sub.groupby(["종류", "확정지연", "지표", "지평"]).agg(
            종목=("종목", "nunique"),
            차트bp=("차트", lambda x: round(x.mean() * 1e4, 2)),
            실제bp=("실제", lambda x: round(x.mean() * 1e4, 2)),
            지연5bp=(d5, lambda x: round(x.mean() * 1e4, 2)),
            차트일치=("차트", lambda x: int((x > 0).sum())),
            실제일치=("실제", lambda x: int((x > 0).sum())),
        ).reset_index()
        g["사라진몫"] = (g["차트bp"] - g["실제bp"]).round(2)
        g["남은비율"] = np.where(g["차트bp"].abs() > 1e-9,
                              (g["실제bp"] / g["차트bp"] * 100).round(0), np.nan)
        pv = []
        for _, r in g.iterrows():
            s = sub[(sub["지표"] == r["지표"]) & (sub["지평"] == r["지평"])]["실제"]
            pos, n = int((s > 0).sum()), len(s)
            pv.append(round(st.binomtest(max(pos, n - pos), n, 0.5).pvalue, 3) if n else np.nan)
        g["실제부호p"] = pv
        print(g.sort_values(["종류", "확정지연", "지표", "지평"]).to_string(index=False))

    print("\n-- 확정지연과 사라진 몫의 관계 (FX, 지평 5) --")
    fx = all_[(all_["계열"] == "FX") & (all_["지평"] == 5)]
    if not fx.empty:
        t = fx.groupby(["지표", "확정지연", "종류"]).agg(
            차트bp=("차트", lambda x: round(x.mean() * 1e4, 2)),
            실제bp=("실제", lambda x: round(x.mean() * 1e4, 2)),
            지연5bp=(d5, lambda x: round(x.mean() * 1e4, 2)),
        ).reset_index().sort_values("확정지연")
        t["사라진몫"] = (t["차트bp"] - t["실제bp"]).round(2)
        t["지연5만의손실"] = (t["차트bp"] - t["지연5bp"]).round(2)
        print(t.to_string(index=False))

    print()
    print("읽는 법")
    print("  · '인과' 지표의 **지연5만의손실** 이 지연 그 자체의 비용이다.")
    print("    그보다 '미래확정' 지표의 사라진 몫이 훨씬 크면, 원인은 지연이")
    print("    아니라 뒤늦게야 알 수 있는 극값을 미리 아는 척한 것이다.")
    print("  · 사라진 몫이 확정지연(2→5→10)에 따라 커지면 기전이 확인된 것이다.")
    print("  · FX 왕복 스프레드는 1.6~3.3bp 다. 실제bp 를 그것과 견줘라.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
