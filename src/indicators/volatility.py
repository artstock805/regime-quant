"""변동성 계열 지표. 레짐 분류에서 '고변동' 판정의 근거가 여기서 나온다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import (
    atr as atr_fn, ema, highest, log_returns, lowest, percent_rank, rma, sma, stdev,
    true_range, typical_price,
)
from .registry import register


@register("atr", "volatility", {"period": 14}, needs=("high", "low", "close"))
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return atr_fn(df, period)


@register("atr_pct", "volatility", {"period": 14}, needs=("high", "low", "close"))
def atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR을 종가 대비 %로. 포지션 사이징이 실제로 쓰는 값이다."""
    return atr_fn(df, period) / df["close"].replace(0, np.nan) * 100.0


@register("atr_ratio", "volatility", {"short": 14, "long": 100}, needs=("high", "low", "close"))
def atr_ratio(df: pd.DataFrame, short: int = 14, long: int = 100) -> pd.Series:
    """단기 ATR / 장기 ATR. 1보다 크면 지금이 평소보다 시끄러운 국면이다."""
    return atr_fn(df, short) / atr_fn(df, long).replace(0, np.nan)


@register(
    "bollinger", "volatility", {"period": 20, "k": 2.0},
    outputs=("bb_upper", "bb_lower", "bb_width", "bb_pct_b"),
    needs=("close",),
)
def bollinger(df: pd.DataFrame, period: int = 20, k: float = 2.0) -> pd.DataFrame:
    basis = sma(df["close"], period)
    sd = stdev(df["close"], period)
    upper, lower = basis + k * sd, basis - k * sd
    span = (upper - lower).replace(0, np.nan)
    return pd.DataFrame({
        "bb_upper": upper, "bb_lower": lower,
        "bb_width": span / basis.replace(0, np.nan) * 100.0,
        "bb_pct_b": (df["close"] - lower) / span,
    })


@register("bbw_percentile", "volatility", {"period": 20, "lookback": 200}, needs=("close",))
def bbw_percentile(df: pd.DataFrame, period: int = 20, lookback: int = 200) -> pd.Series:
    """
    볼린저 폭이 지난 lookback 구간에서 몇 번째 백분위인지.
    변동성의 절대값은 종목마다 다르므로, 자기 과거와 비교해야 의미가 생긴다.
    """
    return percent_rank(bollinger(df, period)["bb_width"], lookback)


@register("squeeze_on", "volatility", {"period": 20, "bb_k": 2.0, "kc_mult": 1.5},
          needs=("high", "low", "close"))
def squeeze_on(df: pd.DataFrame, period: int = 20, bb_k: float = 2.0, kc_mult: float = 1.5) -> pd.Series:
    """
    볼린저가 켈트너 안으로 들어간 상태(=스퀴즈). 1이면 압축, 0이면 해제.
    압축 해제 순간이 돌파 전략의 주요 트리거다.
    """
    bb = bollinger(df, period, bb_k)
    basis = ema(df["close"], period)
    band = atr_fn(df, period) * kc_mult
    inside = (bb["bb_upper"] < basis + band) & (bb["bb_lower"] > basis - band)
    return inside.astype("float64").where(bb["bb_upper"].notna())


@register("hist_vol", "volatility", {"period": 20, "bars_per_year": 105120}, needs=("close",))
def hist_vol(df: pd.DataFrame, period: int = 20, bars_per_year: int = 105120) -> pd.Series:
    """
    로그수익률 표준편차를 연율화한 역사적 변동성(%).
    기본값 105120 = 5분봉의 연간 봉 수(288 × 365). 타임프레임을 바꾸면 함께 바꿔야 한다.
    """
    return stdev(log_returns(df["close"]), period) * np.sqrt(bars_per_year) * 100.0


@register("chaikin_volatility", "volatility", {"period": 10, "roc_period": 10},
          needs=("high", "low"))
def chaikin_volatility(df: pd.DataFrame, period: int = 10, roc_period: int = 10) -> pd.Series:
    span = ema(df["high"] - df["low"], period)
    return (span / span.shift(roc_period).replace(0, np.nan) - 1.0) * 100.0


@register("vix_fix", "volatility", {"period": 22}, needs=("high", "low", "close"))
def vix_fix(df: pd.DataFrame, period: int = 22) -> pd.Series:
    """
    Williams VixFix — 선물 VIX 없이도 '공포 정점'을 잡는 대용치.
    값이 급등하면 투매 국면이고, 되돌림 전략의 트리거로 쓴다.
    """
    hi_close = highest(df["close"], period)
    return 100.0 * (hi_close - df["low"]) / hi_close.replace(0, np.nan)


@register("ulcer_index", "volatility", {"period": 14}, needs=("close",))
def ulcer_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """구간 내 낙폭의 제곱평균 — 하방 변동성만 본다."""
    peak = highest(df["close"], period)
    dd = 100.0 * (df["close"] - peak) / peak.replace(0, np.nan)
    return np.sqrt((dd**2).rolling(period, min_periods=period).mean())


@register("mass_index", "volatility", {"fast": 9, "slow": 25}, needs=("high", "low"))
def mass_index(df: pd.DataFrame, fast: int = 9, slow: int = 25) -> pd.Series:
    span = df["high"] - df["low"]
    single = ema(span, fast)
    ratio = single / ema(single, fast).replace(0, np.nan)
    return ratio.rolling(slow, min_periods=slow).sum()


@register("range_pct", "volatility", {"period": 1}, needs=("high", "low", "close"))
def range_pct(df: pd.DataFrame, period: int = 1) -> pd.Series:
    """봉 고저 폭을 종가 대비 %로. 단일 봉의 크기 이상치를 잡는 데 쓴다."""
    hi = highest(df["high"], period) if period > 1 else df["high"]
    lo = lowest(df["low"], period) if period > 1 else df["low"]
    return (hi - lo) / df["close"].replace(0, np.nan) * 100.0


@register("tr_zscore", "volatility", {"period": 100}, needs=("high", "low", "close"))
def tr_zscore(df: pd.DataFrame, period: int = 100) -> pd.Series:
    """트루레인지의 z-점수. 급등락 봉을 표준화해서 감지한다."""
    tr = true_range(df)
    mu = tr.rolling(period, min_periods=period).mean()
    sd = tr.rolling(period, min_periods=period).std(ddof=0)
    return (tr - mu) / sd.replace(0, np.nan)


@register("rvi", "volatility", {"period": 14}, needs=("close",))
def rvi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Relative Volatility Index — 표준편차를 RSI처럼 상승분/하락분으로 나눠 본다.
    변동성이 '어느 방향으로' 커지는지를 구분한다.
    """
    delta = df["close"].diff()
    sd = stdev(df["close"], period)
    up = rma(sd.where(delta > 0, 0.0), period)
    down = rma(sd.where(delta < 0, 0.0), period)
    return 100.0 * up / (up + down).replace(0, np.nan)
