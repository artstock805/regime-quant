"""
피봇 RSI 다이버전스가 실제로 방향 정보를 갖는가 — 미래 참조를 걷어내고 잰다.

**왜 이걸 따로 재나.** 이 저장소에 이미 `rsi_divergence` 지표가 있는데 그건
회귀 기울기 차이를 쓴다. 주석에 이유가 적혀 있다 — "피봇 방식은 '확정된 피봇'을
알려면 미래 봉이 필요해서 미래 참조를 끌어들이기 쉽다."

그런데 트레이딩뷰에서 흔히 쓰는 다이버전스 지표(이번에 본 `패턴RSI` 포함)는
**바로 그 피봇 방식**이다. 차트에서는 고점·저점을 이은 선이 늘 그럴듯해 보인다.
그래서 질문이 둘로 갈린다.

    (1) 피봇 다이버전스에 방향 정보가 있는가?
    (2) 있어 보이는 것 중 얼마가 **미래 참조**인가?

**설계: 같은 규칙을 두 번 돌린다.**

    lag(정직)   피봇 i 는 i+right 봉에서야 확정된다. 신호도 그때 난다.
                실시간에 실제로 알 수 있는 시점이다.
    peek(차트)  피봇 i 에서 바로 신호가 난 것으로 친다. 차트에 그려지는 모습이고,
                **오른쪽 right 봉을 미리 본 것**이다.

    둘의 차이가 곧 '지나고 보니 맞아 보이는' 몫이다. 이걸 수치로 내는 것이
    이 스크립트의 요점이다.

**판정 기준.** 다이버전스가 쓸모 있으려면 강세 신호 뒤 수익률이 약세 신호 뒤보다
**높아야** 한다. 그 차이(방향차)를 재고, 종목 간 부호 일치로 검정한다.

**종목군을 반드시 가른다.** 이 프로젝트는 섞어 재다가 크게 틀린 적이 있다 —
지표 상위 |IC| 가 전부 '지수가 장기 우상향한다'는 사실 하나의 재표현이었고
FX 에서는 정확히 0 이었다(scripts/screen_mt5_features.py). 지수·FX·금속에너지를
갈라서 본다.

**측정 결과 (2026-10-20, 30종목, left=5 right=5). 결론: 전부 미래 참조였다.**

    방향차 bp = (강세 신호 뒤 수익률) − (약세 신호 뒤 수익률)

    계열        지평   lag(정직)   peek(차트)   미래참조 몫
    ─────────────────────────────────────────────────────
    1시간봉
    FX           1       0.01       17.73        17.72      ← 정확히 0
    FX           5       0.74       32.80        32.06
    지수         1       2.19       45.64        43.45
    지수         3      -0.22       70.28        70.51
    금속에너지   5       2.47      116.32       113.84
    30분봉 (차트에서 쓰던 봉)
    FX           1      -0.08       12.36        12.43      ← 음수
    FX           5       1.37       21.90        20.52
    지수         5       1.50       61.15        59.64
    금속에너지   1      -2.20       55.77        57.98

    peek 는 30/30 종목이 같은 부호에 p=0.0000 이다. 그런데 신호를 **실제로 알
    수 있는 시점으로 5봉 미루기만 하면** FX 부호 일치가 6/11(p=1.00) 로 떨어진다.
    두 시간축에서 독립적으로 같은 답이 나왔다.

**peek 가 양성 대조군이다.** "우위를 못 찾은 건 측정이 둔해서 아니냐"는 반론이
가능한데, 같은 측정이 peek 에서는 30/30 일치에 p=0.0000 을 **찾아냈다**. 검출력은
충분했다. 따라서 lag 의 0 은 '못 찾은 것'이 아니라 **없는 것**이다.

**비용과 견주면 더 분명하다.** FX 왕복 스프레드 실측이 EURUSD 1.6bp ·
USDCHF 2.8bp · AUDUSD 3.3bp 다. lag 방향차 최고치가 0.74~1.37bp 이므로 우위가
비용의 1/2~1/4 에 불과하다. 유의하지도 않지만 설령 진짜라 쳐도 스프레드를 못 넘는다.

**그래서 이 지표를 레지스트리에 등록하지 않았다.** 기존 `rsi_divergence` 가
피봇 방식을 피하고 회귀 기울기를 쓴 판단이 옳았다는 것이 수치로 확인됐다.
이 스크립트는 그 판단의 근거로 남긴다 — 차트에서 다이버전스가 그럴듯해 보일
때마다 다시 재지 않아도 되도록.

사용법
    python scripts/probe_pivot_divergence.py
    python scripts/probe_pivot_divergence.py --interval 4h --right 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scipy import stats as st                                       # noqa: E402

from src.fx.mt5 import bars                                         # noqa: E402
from src.indicators.momentum import rsi                             # noqa: E402

pd.set_option("display.width", 220)

FAMILIES = {
    "지수": ["US500Cash", "US100Cash", "US30Cash", "GER40Cash", "UK100Cash",
             "JP225Cash", "FRA40Cash", "EU50Cash", "AUS200Cash", "HK50Cash",
             "IT40Cash", "NETH25Cash", "SWI20Cash", "ChinaHCash", "US2000Cash"],
    "금속에너지": ["GOLD", "SILVER", "XAUEUR", "OILCash", "BRENTCash"],
    "FX": ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
           "GBPJPY", "EURAUD", "CADJPY", "AUDNZD", "GBPAUD"],
}

BARS = {"30m": 20000, "1h": 20000, "4h": 17000, "1d": 8000}

# 트레이딩뷰 기본 다이버전스 지표의 기본값을 그대로 쓴다. 우리가 고른 값이면
# 그 자체가 또 하나의 자유도가 된다.
LEFT, RIGHT = 5, 5
RANGE_MIN, RANGE_MAX = 5, 60


def pivot_flags(s: pd.Series, left: int, right: int) -> tuple[np.ndarray, np.ndarray]:
    """
    피봇 고점·저점 마스크. s[i] 가 [i-left, i+right] 구간의 유일한 최대(최소)면
    피봇이다.

    **이 함수 자체는 미래를 본다** — i+right 까지 읽기 때문이다. 미래 참조를
    막는 것은 호출하는 쪽의 책임이고, 아래 signals() 가 그 일을 한다.
    """
    v = s.to_numpy("float64")
    n = len(v)
    hi = np.zeros(n, dtype=bool)
    lo = np.zeros(n, dtype=bool)
    span = left + right + 1
    if n < span:
        return hi, lo
    # 창을 통째로 쌓아 한 번에 비교한다. 파이썬 루프로 돌면 지표 여러 개를
    # 30종목에 돌릴 때 너무 느리다. 결과는 루프판과 비트 단위로 같다(검증함).
    w = sliding_window_view(v, span)                 # (n-span+1, span)
    c = v[left: n - right]
    finite = np.isfinite(w).all(axis=1)
    uniq_hi = (w == c[:, None]).sum(axis=1) == 1
    hi[left: n - right] = finite & uniq_hi & (c == w.max(axis=1))
    lo[left: n - right] = finite & uniq_hi & (c == w.min(axis=1))
    return hi, lo


def signals(df: pd.DataFrame, osc: pd.Series, *, left: int, right: int,
            peek: bool) -> pd.Series:
    """
    +1 강세 다이버전스, -1 약세 다이버전스, 0 없음.

    강세  직전 피봇저점 대비 **가격은 더 낮은 저점, 오실레이터는 더 높은 저점**
    약세  직전 피봇고점 대비 **가격은 더 높은 고점, 오실레이터는 더 낮은 고점**

    `peek=False` 면 신호를 피봇에서 right 봉 뒤로 민다 — 그때가 피봇이 확정되는,
    즉 실시간에 알 수 있는 시점이다. `peek=True` 면 차트처럼 피봇 자리에 찍는다.
    """
    hi, lo = pivot_flags(osc, left, right)
    o = osc.to_numpy("float64")
    ph = df["high"].to_numpy("float64")
    pl = df["low"].to_numpy("float64")
    n = len(o)
    out = np.zeros(n, dtype="float64")

    def emit(idx: int, value: float) -> None:
        at = idx if peek else idx + right
        if 0 <= at < n:
            out[at] = value

    prev = -1
    for i in np.flatnonzero(hi):
        if prev >= 0 and RANGE_MIN <= i - prev <= RANGE_MAX:
            if ph[i] > ph[prev] and o[i] < o[prev]:
                emit(i, -1.0)
        prev = i

    prev = -1
    for i in np.flatnonzero(lo):
        if prev >= 0 and RANGE_MIN <= i - prev <= RANGE_MAX:
            if pl[i] < pl[prev] and o[i] > o[prev]:
                emit(i, +1.0)
        prev = i

    return pd.Series(out, index=df.index, name="div")


def measure(sym: str, interval: str, horizons: list[int],
            left: int, right: int) -> list[dict]:
    df = bars(sym, interval, BARS[interval])
    if len(df) < 1500:
        return []
    osc = rsi(df, 14)
    rows = []
    for peek in (False, True):
        sig = signals(df, osc, left=left, right=right, peek=peek)
        for h in horizons:
            fwd = df["close"].pct_change(h).shift(-h)
            ok = fwd.notna()
            bull = fwd[ok & (sig > 0)]
            bear = fwd[ok & (sig < 0)]
            if len(bull) < 10 or len(bear) < 10:
                continue
            base = float(fwd[ok].mean())
            # 방향차: 강세 신호 뒤 수익률 − 약세 신호 뒤 수익률. 다이버전스가
            # 통하면 양수여야 한다. 종목의 추세 편향은 양쪽에 똑같이 실리므로
            # 빼기 한 번으로 상쇄된다 — 지수의 우상향이 신호로 둔갑하는 것을
            # 막는 장치다.
            rows.append({
                "종목": sym, "모드": "peek(차트)" if peek else "lag(정직)",
                "지평": h,
                "강세건수": len(bull), "약세건수": len(bear),
                "강세초과": float(bull.mean()) - base,
                "약세초과": float(bear.mean()) - base,
                "방향차": float(bull.mean()) - float(bear.mean()),
            })
    return rows


def summarise(d: pd.DataFrame, label: str) -> None:
    print(f"\n-- {label} --")
    if d.empty:
        print("  데이터 없음")
        return
    g = d.groupby(["모드", "지평"]).agg(
        종목=("종목", "nunique"),
        강세=("강세건수", "sum"), 약세=("약세건수", "sum"),
        방향차평균=("방향차", "mean"), 방향차중위=("방향차", "median"),
        양수종목=("방향차", lambda x: int((x > 0).sum())),
    ).reset_index()
    # 종목 간 부호검정 — 한 종목의 큰 값보다 여러 종목의 같은 부호가 낫다
    pv = []
    for _, r in g.iterrows():
        sub = d[(d["모드"] == r["모드"]) & (d["지평"] == r["지평"])]["방향차"]
        pos, n = int((sub > 0).sum()), len(sub)
        pv.append(st.binomtest(max(pos, n - pos), n, 0.5).pvalue if n else np.nan)
    g["부호p"] = np.round(pv, 4)
    g["방향차bp"] = (g.pop("방향차평균") * 1e4).round(2)
    g["중위bp"] = (g.pop("방향차중위") * 1e4).round(2)
    print(g.to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--horizons", nargs="*", type=int, default=[1, 3, 5])
    ap.add_argument("--left", type=int, default=LEFT)
    ap.add_argument("--right", type=int, default=RIGHT)
    args = ap.parse_args()

    rows = []
    for fam, syms in FAMILIES.items():
        for sym in syms:
            try:
                r = measure(sym, args.interval, args.horizons, args.left, args.right)
            except Exception as exc:
                print(f"  {sym:12s} 실패 {type(exc).__name__}", flush=True)
                continue
            if not r:
                print(f"  {sym:12s} 건너뜀", flush=True)
                continue
            for x in r:
                x["계열"] = fam
            rows += r
            n = r[0]["강세건수"] + r[0]["약세건수"]
            print(f"  {fam:6s} {sym:12s} 신호 {n}", flush=True)
    if not rows:
        return 1

    all_ = pd.DataFrame(rows)
    print()
    print("=" * 100)
    print(f"피봇 RSI 다이버전스 · {args.interval} · left={args.left} right={args.right}"
          f" · 종목 {all_['종목'].nunique()}")
    print("=" * 100)
    print("\n방향차 = (강세 신호 뒤 수익률) − (약세 신호 뒤 수익률), 단위 bp.")
    print("다이버전스가 통하면 **양수**여야 한다.")

    summarise(all_, "전체 (계열을 섞은 것 — 참고만. 섞으면 추세 편향이 섞인다)")
    for fam in FAMILIES:
        summarise(all_[all_["계열"] == fam], f"{fam} 계열")

    # 미래 참조가 차지하는 몫
    print("\n-- 미래 참조가 차지하는 몫 (계열 x 지평, bp) --")
    piv = all_.pivot_table(index=["계열", "지평"], columns="모드",
                           values="방향차", aggfunc="mean") * 1e4
    if {"lag(정직)", "peek(차트)"}.issubset(set(piv.columns)):
        piv["미래참조몫"] = piv["peek(차트)"] - piv["lag(정직)"]
    print(piv.round(2).to_string())

    print()
    print("판단 기준")
    print("  · lag(정직) 의 방향차가 양수이고 부호검정이 유의해야 쓸 수 있다.")
    print("  · peek 만 좋고 lag 가 0 이면, 차트에서 보이던 것은 전부 미래 참조다.")
    print("  · FX 왕복 비용은 1R 의 10.9% 다. 방향차가 몇 bp 인지 비용과 견줘라.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
