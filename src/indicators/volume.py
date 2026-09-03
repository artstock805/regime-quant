"""
거래량 계열 지표.

taker_buy_base 를 쓰는 지표는 바이낸스가 봉마다 제공하는 '테이커 매수 체결량'에
의존한다. 거래소를 바꾸면 이 열이 없을 수 있어서 needs 에 명시해 두었고,
compute_all 이 조용히 건너뛴다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import ema, rma, sma, typical_price, zscore
from .registry import register


@register("obv", "volume", {}, needs=("close", "volume"))
def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()


@register("obv_slope", "volume", {"lookback": 20}, needs=("close", "volume"))
def obv_slope(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    OBV의 절대값은 시작점에 따라 달라져 비교 불가다. 변화량을 같은 구간
    거래량 합으로 정규화해서 -1~+1 범위로 만든다.
    """
    line = obv(df)
    vol_sum = df["volume"].rolling(lookback, min_periods=lookback).sum().replace(0, np.nan)
    return line.diff(lookback) / vol_sum


@register("mfi", "volume", {"period": 14}, needs=("high", "low", "close", "volume"))
def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = typical_price(df)
    flow = tp * df["volume"]
    up = flow.where(tp.diff() > 0, 0.0).rolling(period, min_periods=period).sum()
    down = flow.where(tp.diff() < 0, 0.0).rolling(period, min_periods=period).sum()
    return 100.0 * up / (up + down).replace(0, np.nan)


@register("cmf", "volume", {"period": 20}, needs=("high", "low", "close", "volume"))
def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow — 봉 안에서 종가가 고저 어디에 붙었는지를 거래량으로 가중."""
    span = (df["high"] - df["low"]).replace(0, np.nan)
    mfv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / span * df["volume"]
    vol_sum = df["volume"].rolling(period, min_periods=period).sum().replace(0, np.nan)
    return mfv.rolling(period, min_periods=period).sum() / vol_sum


@register("ad_line", "volume", {"lookback": 20}, needs=("high", "low", "close", "volume"))
def ad_line(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """누적/분산선의 정규화된 변화량."""
    span = (df["high"] - df["low"]).replace(0, np.nan)
    mfv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / span * df["volume"]
    line = mfv.cumsum()
    vol_sum = df["volume"].rolling(lookback, min_periods=lookback).sum().replace(0, np.nan)
    return line.diff(lookback) / vol_sum


@register("pvt", "volume", {"lookback": 20}, needs=("close", "volume"))
def pvt(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    ret = df["close"].pct_change()
    line = (ret * df["volume"]).cumsum()
    vol_sum = df["volume"].rolling(lookback, min_periods=lookback).sum().replace(0, np.nan)
    return line.diff(lookback) / vol_sum


@register("force_index", "volume", {"period": 13}, needs=("close", "volume"))
def force_index(df: pd.DataFrame, period: int = 13) -> pd.Series:
    """가격 변화 × 거래량을 평활. 종가 대비 %로 정규화해서 종목 간 비교 가능하게 한다."""
    raw = df["close"].diff() * df["volume"]
    scale = (df["close"] * df["volume"]).rolling(period, min_periods=period).mean()
    return ema(raw, period) / scale.replace(0, np.nan) * 100.0


@register("eom", "volume", {"period": 14}, needs=("high", "low", "volume"))
def eom(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Ease of Movement — 적은 거래량으로 크게 움직였는지."""
    mid = (df["high"] + df["low"]) / 2.0
    span = (df["high"] - df["low"]).replace(0, np.nan)
    box = df["volume"] / span
    return sma(mid.diff() / box.replace(0, np.nan), period)


@register("vol_zscore", "volume", {"period": 100}, needs=("volume",))
def vol_zscore(df: pd.DataFrame, period: int = 100) -> pd.Series:
    return zscore(df["volume"], period)


@register("volume_ratio", "volume", {"period": 20}, needs=("volume",))
def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """현재 거래량 / 평균 거래량. 돌파의 진위를 가리는 확인 지표로 쓴다."""
    return df["volume"] / sma(df["volume"], period).replace(0, np.nan)


@register("vwap_dev", "volume", {"period": 96}, needs=("high", "low", "close", "volume"))
def vwap_dev(df: pd.DataFrame, period: int = 96) -> pd.Series:
    """
    구간 VWAP 대비 종가의 이탈률(%).

    세션 기준 VWAP 대신 롤링 VWAP을 쓴다. 암호화폐는 24시간 거래라 '세션'의
    경계가 임의적이고, 롤링이면 타임프레임을 바꿔도 의미가 유지된다.
    """
    tp = typical_price(df)
    pv = (tp * df["volume"]).rolling(period, min_periods=period).sum()
    vol = df["volume"].rolling(period, min_periods=period).sum().replace(0, np.nan)
    vwap = pv / vol
    return (df["close"] / vwap - 1.0) * 100.0


@register("taker_imbalance", "volume", {"period": 20},
          needs=("volume", "taker_buy_base"))
def taker_imbalance(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    테이커 매수 비중 - 0.5 를 평활. +면 시장가 매수가 우세.

    바이낸스가 봉 단위로 주는 값이라 오더플로에 가장 가까운 정보다. 일반적인
    가격 기반 지표에 없는 정보를 더해 주기 때문에 앙상블에서 값이 크다.
    """
    share = df["taker_buy_base"] / df["volume"].replace(0, np.nan)
    return rma(share - 0.5, period)


@register("taker_imbalance_z", "volume", {"period": 100},
          needs=("volume", "taker_buy_base"))
def taker_imbalance_z(df: pd.DataFrame, period: int = 100) -> pd.Series:
    share = df["taker_buy_base"] / df["volume"].replace(0, np.nan)
    return zscore(share, period)


@register("trades_zscore", "volume", {"period": 100}, needs=("trades",))
def trades_zscore(df: pd.DataFrame, period: int = 100) -> pd.Series:
    """체결 건수의 z-점수. 거래량은 큰 주문 하나로도 뛰지만 건수는 참여자 수를 반영한다."""
    return zscore(df["trades"].astype("float64"), period)


@register("avg_trade_size_z", "volume", {"period": 100}, needs=("volume", "trades"))
def avg_trade_size_z(df: pd.DataFrame, period: int = 100) -> pd.Series:
    """평균 체결 크기의 z-점수. 급등하면 대형 주문이 들어온 국면이다."""
    size = df["volume"] / df["trades"].replace(0, np.nan)
    return zscore(size, period)
