"""모멘텀 계열 지표."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import (
    _place, _windows, ema, highest, linreg_slope, lowest, roc as roc_fn, rma, sma,
    true_range, typical_price,
)
from .registry import register


@register("rsi", "momentum", {"period": 14}, needs=("close",))
def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain = rma(delta.clip(lower=0), period)
    loss = rma((-delta).clip(lower=0), period)
    rs = gain / loss.replace(0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # loss가 0이면 RSI는 정의상 100
    return out.mask((loss == 0) & gain.notna(), 100.0)


@register(
    "stoch", "momentum", {"k_period": 14, "d_period": 3, "smooth": 3},
    outputs=("stoch_k", "stoch_d"),
    needs=("high", "low", "close"),
)
def stoch(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth: int = 3) -> pd.DataFrame:
    hi, lo = highest(df["high"], k_period), lowest(df["low"], k_period)
    raw_k = 100.0 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    k = sma(raw_k, smooth)
    return pd.DataFrame({"stoch_k": k, "stoch_d": sma(k, d_period)})


@register(
    "stoch_rsi", "momentum", {"rsi_period": 14, "stoch_period": 14, "smooth": 3},
    outputs=("srsi_k", "srsi_d"),
    needs=("close",),
)
def stoch_rsi(df: pd.DataFrame, rsi_period: int = 14, stoch_period: int = 14, smooth: int = 3) -> pd.DataFrame:
    r = rsi(df, rsi_period)
    hi, lo = highest(r, stoch_period), lowest(r, stoch_period)
    raw = 100.0 * (r - lo) / (hi - lo).replace(0, np.nan)
    k = sma(raw, smooth)
    return pd.DataFrame({"srsi_k": k, "srsi_d": sma(k, smooth)})


@register(
    "macd", "momentum", {"fast": 12, "slow": 26, "signal": 9},
    outputs=("macd", "macd_signal", "macd_hist"),
    needs=("close",),
)
def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(df["close"], fast) - ema(df["close"], slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


@register("ppo", "momentum", {"fast": 12, "slow": 26}, needs=("close",))
def ppo(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    """MACD를 %로 정규화한 것 — 종목 간 비교가 가능해진다."""
    slow_line = ema(df["close"], slow)
    return (ema(df["close"], fast) - slow_line) / slow_line.replace(0, np.nan) * 100.0


@register("cci", "momentum", {"period": 20}, needs=("high", "low", "close"))
def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Commodity Channel Index.

    분모의 평균편차는 **그 윈도우의 단일 평균**으로부터의 절대편차 평균이다.
        MAD = mean_i( |TP_i - SMA(window)| )

    각 봉의 자기 이동평균과의 편차를 다시 평균내면(= (tp - ma).abs().rolling().mean())
    다른 값이 나온다. 이동평균이 윈도우 안에서 움직이기 때문이고, 타임프레임이
    짧을수록 오차가 커진다 — 실측으로 5분봉 18%, 1시간봉 21%, 일봉 1% 어긋났다.
    지표 대조(scripts/crosscheck_indicators.py)가 잡아낸 버그다.
    """
    tp = typical_price(df)
    ma = sma(tp, period)

    win = _windows(tp, period)
    if win is None:
        return pd.Series(np.nan, index=df.index, name="cci")
    mad_vals = np.abs(win - win.mean(axis=1, keepdims=True)).mean(axis=1)
    mad = _place(mad_vals, tp, period)

    return (tp - ma) / (0.015 * mad.replace(0, np.nan))


@register("roc", "momentum", {"period": 12}, needs=("close",))
def roc(df: pd.DataFrame, period: int = 12) -> pd.Series:
    return roc_fn(df["close"], period)


@register("momentum", "momentum", {"period": 10}, needs=("close",))
def momentum(df: pd.DataFrame, period: int = 10) -> pd.Series:
    return df["close"] - df["close"].shift(period)


@register("williams_r", "momentum", {"period": 14}, needs=("high", "low", "close"))
def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi, lo = highest(df["high"], period), lowest(df["low"], period)
    return -100.0 * (hi - df["close"]) / (hi - lo).replace(0, np.nan)


@register("tsi", "momentum", {"long": 25, "short": 13}, needs=("close",))
def tsi(df: pd.DataFrame, long: int = 25, short: int = 13) -> pd.Series:
    delta = df["close"].diff()
    smooth = ema(ema(delta, long), short)
    smooth_abs = ema(ema(delta.abs(), long), short)
    return 100.0 * smooth / smooth_abs.replace(0, np.nan)


@register("cmo", "momentum", {"period": 14}, needs=("close",))
def cmo(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Chande 모멘텀 오실레이터."""
    delta = df["close"].diff()
    up = delta.clip(lower=0).rolling(period, min_periods=period).sum()
    down = (-delta).clip(lower=0).rolling(period, min_periods=period).sum()
    total = (up + down).replace(0, np.nan)
    return 100.0 * (up - down) / total


@register("ultimate_osc", "momentum", {"p1": 7, "p2": 14, "p3": 28}, needs=("high", "low", "close"))
def ultimate_osc(df: pd.DataFrame, p1: int = 7, p2: int = 14, p3: int = 28) -> pd.Series:
    prev_close = df["close"].shift(1)
    true_low = pd.concat([df["low"], prev_close], axis=1).min(axis=1)
    bp = df["close"] - true_low
    tr = true_range(df)

    def _avg(n: int) -> pd.Series:
        return (
            bp.rolling(n, min_periods=n).sum()
            / tr.rolling(n, min_periods=n).sum().replace(0, np.nan)
        )

    return 100.0 * (4 * _avg(p1) + 2 * _avg(p2) + _avg(p3)) / 7.0


@register("awesome_osc", "momentum", {"fast": 5, "slow": 34}, needs=("high", "low"))
def awesome_osc(df: pd.DataFrame, fast: int = 5, slow: int = 34) -> pd.Series:
    mid = (df["high"] + df["low"]) / 2.0
    return sma(mid, fast) - sma(mid, slow)


@register("bop", "momentum", {"smooth": 14}, needs=("open", "high", "low", "close"))
def bop(df: pd.DataFrame, smooth: int = 14) -> pd.Series:
    """Balance of Power — 봉 안에서 매수·매도 중 누가 이겼는지."""
    span = (df["high"] - df["low"]).replace(0, np.nan)
    return sma((df["close"] - df["open"]) / span, smooth)


@register("kst", "momentum", {}, needs=("close",))
def kst(df: pd.DataFrame) -> pd.Series:
    """Know Sure Thing — 네 개 기간의 ROC를 가중 합산한 장기 모멘텀."""
    c = df["close"]
    parts = [
        sma(roc_fn(c, 10), 10) * 1,
        sma(roc_fn(c, 15), 10) * 2,
        sma(roc_fn(c, 20), 10) * 3,
        sma(roc_fn(c, 30), 15) * 4,
    ]
    return sum(parts)


@register(
    "rsi_divergence", "momentum", {"rsi_period": 14, "lookback": 20},
    needs=("close",),
)
def rsi_divergence(df: pd.DataFrame, rsi_period: int = 14, lookback: int = 20) -> pd.Series:
    """
    가격 기울기와 RSI 기울기의 부호 불일치를 하나의 값으로 만든다.

    +값 = 강세 다이버전스(가격은 내리는데 RSI는 오름), -값 = 약세 다이버전스.
    피봇을 찾아 짝짓는 고전적 방식 대신 회귀 기울기 차이를 쓴다. 피봇 방식은
    '확정된 피봇'을 알려면 미래 봉이 필요해서 미래 참조를 끌어들이기 쉽다.
    """
    price_slope = linreg_slope(df["close"], lookback)
    rsi_line = rsi(df, rsi_period)
    rsi_slope = rsi_line.diff(lookback) / lookback

    diverging = np.sign(price_slope) != np.sign(rsi_slope)
    magnitude = (rsi_slope.abs() + price_slope.abs()) / 2.0
    return (np.sign(rsi_slope) * magnitude).where(diverging, 0.0)
