"""
오더플로 **대용** 지표.

**이름에 '대용'이 붙은 이유를 먼저 읽어야 한다.**

진짜 오더플로는 세 가지를 안다 — 누가 공격자였는지(체결 방향), 얼마나 체결됐는지
(체결량), 호가창에 뭐가 쌓여 있는지(깊이). 리테일 FX 에는 그 세 가지가 **전부
없다.** 중앙 거래소가 없어서다. XM 데모(EURUSD)에서 실측한 결과:

    real_volume   300봉 전부 0
    market_book   market_book_add 실패 — 호가창을 제공하지 않는다
    틱의 last     180일 전까지 100% 0  (체결가가 아예 안 온다)
    틱의 volume   100% 0

즉 틱은 **호가(bid/ask) 갱신 기록**일 뿐이고 체결 기록이 아니다. 그래서 여기 있는
지표는 오더플로가 아니라 **오더플로처럼 읽히는 대용물**이다. 재료는 두 개뿐이다.

    volume(=tick_volume)  그 봉에서 호가가 몇 번 바뀌었나 — 활동량, 방향은 없음
    spread                그때 스프레드 — 유동성이 빠지면 벌어진다 (FX 만 있음)

방향은 **봉 안에서 종가가 어디에 붙었는지**로 추정한다. 고가 근처에 붙었으면 그
봉의 활동이 매수 쪽으로 쏠렸다고 본다. 이것이 A/D 계열이 쓰는 오래된 규칙이고,
체결 방향을 모를 때 쓸 수 있는 최선이다. **틀릴 수 있다** — 종가가 고가에 붙어도
실제로는 매도가 흡수된 것일 수 있다. 그 한계를 알고 쓰는 것과 모르고 쓰는 것은
다르므로 여기 적어 둔다.

`absorption`(노력 대비 결과)을 같이 두는 이유가 그것이다. 호가가 많이 바뀌었는데
가격이 안 움직였으면 한쪽이 흡수당한 것이고, 그때는 델타의 방향을 믿기 어렵다.

**측정 결과 (2026-09-07, 1시간봉 7개 통화쌍). 기대하고 오면 실망한다.**

    ofi_z      IC -0.054 (7/7 음수) — 이 프로젝트 단일 특징 중 최고
    cvd        IC -0.031 (7/7 음수)
    cvd_slope  IC -0.029 (6/7 음수)
    absorption IC -0.030 (5/7 음수) — 일관성이 약하다
    cvd_divergence IC +0.0005 — 사실상 0

그런데 **ofi_z 의 IC 중 약 2/3 은 이미 알려진 단기 반전이다.** 7쌍 평균으로
직전수익률 z 가 -0.035, 종가위치 z 가 -0.046, 거래량을 곱하면 -0.054 다.
그리고 `ofi_delta` 는 `거래량 × 종가위치` 인데 **ad_line·cmf 가 이미 같은
구성**이라, 이것으로 만든 전략은 flow_extension_fade 와 방향이 100% 일치했다.
즉 이 모듈의 대부분은 기존 volume 카테고리를 다시 만든 것이다.

틱으로 해상도를 올려도 나아지지 않았다 (`src/fx/ticks.py`): 1,117만 틱에서
봉당 상승/하락 틱 수를 센 불균형이 IC -0.018 로 더 약했다.

지표를 남겨 두는 이유는 **그 사실이 기록으로 남을 가치가 있어서**다. 다음에
누가 "오더플로를 넣어 보자"고 할 때 여기서부터 시작하면 된다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import zscore
from .registry import register


def _clv(df: pd.DataFrame) -> pd.Series:
    """
    종가 위치 (Close Location Value). -1(저가에 붙음) ~ +1(고가에 붙음).

    고가와 저가가 같은 봉(움직임이 없던 봉)은 0 이다. 0 으로 나누면 inf 가
    되어 뒤의 z-score 를 전부 오염시킨다.
    """
    rng = (df["high"] - df["low"]).astype("float64")
    clv = (2.0 * df["close"] - df["high"] - df["low"]) / rng.replace(0.0, np.nan)
    return clv.fillna(0.0).clip(-1.0, 1.0)


@register("ofi_delta", "orderflow", {}, needs=("high", "low", "close", "volume"))
def ofi_delta(df: pd.DataFrame) -> pd.Series:
    """
    봉별 델타 대용 = 활동량 × 종가 위치.

    체결 방향을 모르니 종가 위치로 방향을 준다. 양수면 그 봉의 활동이 매수
    쪽으로 쏠렸다는 추정이다.
    """
    return (df["volume"].astype("float64") * _clv(df)).rename("ofi_delta")


@register("ofi_z", "orderflow", {"period": 100},
          needs=("high", "low", "close", "volume"))
def ofi_z(df: pd.DataFrame, period: int = 100) -> pd.Series:
    """
    델타를 자기 최근 분포로 표준화한다.

    통화쌍마다 틱 수 규모가 다르고(EURUSD 와 EURGBP 가 몇 배 차이) 시간대마다
    또 다르다. 원값을 그대로 쓰면 문턱이 통화쌍·시간마다 다른 것을 요구한다.
    """
    return zscore(ofi_delta(df), period).rename("ofi_z")


@register("cvd", "orderflow", {"period": 100},
          needs=("high", "low", "close", "volume"))
def cvd(df: pd.DataFrame, period: int = 100) -> pd.Series:
    """
    누적 델타(CVD) 대용 — 단, **자기 이동평균을 뺀 값**이다.

    누적합을 그대로 쓰면 안 된다. 무한히 커지거나 작아지는 추세를 타서
    z-score 가 시간 흐름만 재게 되고, 최근 쏠림을 못 잡는다. 이동평균을 빼서
    "최근 흐름이 자기 평균보다 어느 쪽인가"만 남긴다.
    """
    raw = ofi_delta(df).cumsum()
    return (raw - raw.rolling(period, min_periods=period // 2).mean()).rename("cvd")


@register("cvd_slope", "orderflow", {"lookback": 20, "period": 100},
          needs=("high", "low", "close", "volume"))
def cvd_slope(df: pd.DataFrame, lookback: int = 20, period: int = 100) -> pd.Series:
    """
    CVD 의 기울기 — 쏠림이 **빨라지는 중인지** 본다.

    **이것은 흐름의 방향이 아니라 흐름의 가속이다.** cvd 가 추세를 제거한 값이라
    델타가 일정한 시장에서는 cvd 가 상수가 되고 기울기는 0 이 된다. 즉 "계속
    팔리고 있다"는 0 이고, "팔리는 속도가 빨라졌다"가 음수다.

    처음엔 헷갈렸고 테스트가 잡아냈다. 일정한 쏠림은 그 통화쌍의 평상시 상태에
    가깝고, 진입 근거가 되는 것은 평상시에서 벗어나는 순간이므로 이 성질이
    오히려 원하는 쪽이다.
    """
    c = cvd(df, period)
    return ((c - c.shift(lookback)) / float(lookback)).rename("cvd_slope")


@register("absorption", "orderflow", {"period": 100},
          needs=("high", "low", "close", "volume"))
def absorption(df: pd.DataFrame, period: int = 100) -> pd.Series:
    """
    노력 대비 결과 — 활동량은 많은데 가격이 안 움직였나.

    양수가 크면 **흡수**다. 호가가 바쁘게 바뀌었는데 가격이 제자리라는 것은
    한쪽 물량이 다른 쪽에 먹히고 있다는 뜻이고, 그때 델타의 방향은 믿기 어렵다.
    전략이 이걸 문지기로 쓴다.
    """
    rng = (df["high"] - df["low"]).astype("float64")
    # 통화쌍마다 가격 단위가 다르다(엔 100대, 유로 1대). 종가로 나눠 무차원화한다.
    rel_rng = (rng / df["close"].astype("float64")).replace(0.0, np.nan)
    effort = df["volume"].astype("float64") / rel_rng
    return zscore(effort.replace([np.inf, -np.inf], np.nan), period).rename("absorption")


@register("cvd_divergence", "orderflow", {"lookback": 20, "period": 100},
          needs=("high", "low", "close", "volume"))
def cvd_divergence(df: pd.DataFrame, lookback: int = 20,
                   period: int = 100) -> pd.Series:
    """
    가격과 누적델타가 **다른 말을 하는지**. +1 강세 / -1 약세 / 0 없음.

    가격은 신고가인데 델타는 못 따라오면 위로 밀 힘이 빠진 것이다(약세).
    반대로 가격은 신저가인데 델타가 안 따라오면 아래로 밀 힘이 빠진 것이다(강세).

    되돌림 신호로 오래 쓰인 형태지만 **자주 틀린다** — 힘이 빠져 보여도 그대로
    더 가는 경우가 많다. 그래서 이것만으로 방향을 정하지 않고 보조로만 쓴다.
    """
    close = df["close"].astype("float64")
    c = cvd(df, period)
    win = max(2, int(lookback))

    price_high = close >= close.rolling(win, min_periods=win).max()
    price_low = close <= close.rolling(win, min_periods=win).min()
    cvd_high = c >= c.rolling(win, min_periods=win).max()
    cvd_low = c <= c.rolling(win, min_periods=win).min()

    out = pd.Series(0.0, index=df.index, dtype="float64")
    out[price_high & ~cvd_high] = -1.0      # 가격 신고가, 델타는 아님 → 약세
    out[price_low & ~cvd_low] = 1.0         # 가격 신저가, 델타는 아님 → 강세
    return out.rename("cvd_divergence")


@register("spread_z", "orderflow", {"period": 100}, needs=("spread",))
def spread_z(df: pd.DataFrame, period: int = 100) -> pd.Series:
    """
    스프레드가 자기 평균보다 얼마나 벌어졌나. **FX 전용** — 암호화폐 봉에는
    spread 열이 없어서 compute_all 이 조용히 건너뛴다.

    스프레드가 벌어지는 것은 유동성이 빠지는 신호다. 그때 들어가면 비용이
    커지고, 이 프로젝트에서 비용은 우위를 먹어 치우는 쪽이었다(FX 실측 평균
    비용이 1R 의 10.9%).
    """
    return zscore(df["spread"].astype("float64"), period).rename("spread_z")
