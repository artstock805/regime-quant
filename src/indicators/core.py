"""
지표 프리미티브.

이 파일의 모든 함수는 **인과적(causal)** 이어야 한다. 즉 시점 t의 출력은 t 이하의
입력만으로 결정되어야 한다. center=True, shift(음수), bfill 은 이 파일에서 금지다.
tests/test_causality.py 가 지표 전체에 대해 이 성질을 자동으로 검사한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─── 이동평균 계열 ────────────────────────────────────────────────

def sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period, min_periods=period).mean()


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def rma(s: pd.Series, period: int) -> pd.Series:
    """Wilder 이동평균. RSI·ADX·ATR의 원 정의가 쓰는 평활이다."""
    return s.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def wma(s: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    return s.rolling(period, min_periods=period).apply(
        lambda w: float(np.dot(w, weights) / weights.sum()), raw=True
    )


def hma(s: pd.Series, period: int) -> pd.Series:
    """Hull 이동평균 — 지연이 짧은 추세선."""
    half, sqrt_p = max(1, period // 2), max(1, int(np.sqrt(period)))
    return wma(2 * wma(s, half) - wma(s, period), sqrt_p)


# ─── 변동성 프리미티브 ───────────────────────────────────────────

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return rma(true_range(df), period)


def stdev(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period, min_periods=period).std(ddof=0)


# ─── 형태 · 위치 ─────────────────────────────────────────────────

def highest(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period, min_periods=period).max()


def lowest(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period, min_periods=period).min()


def roc(s: pd.Series, period: int) -> pd.Series:
    """변화율(%). 0으로 나누는 구간은 NaN으로 둔다."""
    base = s.shift(period)
    return (s / base.replace(0, np.nan) - 1.0) * 100.0


def zscore(s: pd.Series, period: int) -> pd.Series:
    mu = s.rolling(period, min_periods=period).mean()
    sd = s.rolling(period, min_periods=period).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)


def _windows(s: pd.Series, period: int) -> np.ndarray | None:
    """
    끝점이 각 시점인 롤링 윈도우 뷰. shape (N-period+1, period).
    행 i 는 인덱스 i+period-1 에서 끝나는 윈도우 — 뒤를 보므로 인과적이다.
    뷰이므로 복사가 일어나지 않는다.
    """
    arr = s.to_numpy(dtype="float64")
    if len(arr) < period:
        return None
    return np.lib.stride_tricks.sliding_window_view(arr, period)


def _place(values: np.ndarray, s: pd.Series, period: int) -> pd.Series:
    """윈도우 계산 결과를 원래 인덱스에 맞춰 앞쪽을 NaN으로 채워 되돌린다."""
    out = np.full(len(s), np.nan)
    out[period - 1:] = values
    return pd.Series(out, index=s.index, name=s.name)


def _weighted_rolling_sum(arr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Σ_k weights[k] · y[t-period+1+k] 를 컨볼루션으로 계산한다.

    np.convolve(mode="valid") 는 커널을 뒤집어 적용하므로 미리 뒤집어 넣는다.
    N×period 배열을 만들지 않아 메모리가 기간에 비례해 늘지 않는다.
    """
    return np.convolve(arr, weights[::-1], mode="valid")


def percent_rank(s: pd.Series, period: int) -> pd.Series:
    """
    현재값이 지난 period 구간에서 몇 번째 백분위인지 (0~100).
    변동성이 '지금 높은지'를 절대값이 아니라 자기 과거와 비교해 판단할 때 쓴다.
    """
    win = _windows(s, period)
    if win is None:
        return pd.Series(np.nan, index=s.index, name=s.name)
    current = win[:, -1:]
    past = win[:, :-1]
    # 윈도우에 NaN이 있으면 결과도 NaN이어야 한다 (비교는 NaN을 False로 삼켜버린다)
    ratio = (past < current).mean(axis=1) * 100.0
    return _place(np.where(np.isnan(win).any(axis=1), np.nan, ratio), s, period)


def _linreg_parts(s: pd.Series, period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    구간 선형회귀를 닫힌 형태로 계산한다. (beta, ss_tot, last)

    x 가 0..period-1 로 고정이므로 Σ(x-x̄)² 이 상수이고,
    Σ(x-x̄)y 는 고정 가중치 컨볼루션 한 번으로 얻는다.
    """
    arr = s.to_numpy(dtype="float64")
    if len(arr) < period:
        return None
    x_dev = np.arange(period, dtype="float64")
    x_dev -= x_dev.mean()
    denom = float((x_dev**2).sum())

    sum_y = _weighted_rolling_sum(arr, np.ones(period))
    sum_y2 = _weighted_rolling_sum(arr * arr, np.ones(period))
    sum_xy = _weighted_rolling_sum(arr, x_dev)

    mean_y = sum_y / period
    beta = sum_xy / denom
    ss_tot = sum_y2 - period * mean_y**2
    # 부동소수 오차로 -1e-12 같은 값이 나올 수 있다
    ss_tot = np.where(ss_tot < 0, 0.0, ss_tot)
    return beta, ss_tot, arr[period - 1:]


def linreg_slope(s: pd.Series, period: int) -> pd.Series:
    """
    구간 선형회귀의 기울기를 마지막 값 대비 %로 정규화해 돌려준다.
    종목 가격대가 달라도 비교할 수 있게 만들기 위한 정규화다.
    """
    parts = _linreg_parts(s, period)
    if parts is None:
        return pd.Series(np.nan, index=s.index, name=s.name)
    beta, _, last = parts
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.where(last != 0, beta / last * 100.0, np.nan)
    return _place(vals, s, period)


def linreg_r2(s: pd.Series, period: int) -> pd.Series:
    """구간 선형회귀의 결정계수. 추세가 '깔끔한지'를 나타낸다."""
    parts = _linreg_parts(s, period)
    if parts is None:
        return pd.Series(np.nan, index=s.index, name=s.name)
    beta, ss_tot, _ = parts
    x_dev = np.arange(period, dtype="float64")
    x_dev -= x_dev.mean()
    ss_reg = beta**2 * float((x_dev**2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.where(ss_tot > 0, ss_reg / ss_tot, np.nan)
    return _place(vals, s, period)


# ─── 교차 ────────────────────────────────────────────────────────

def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """a가 b를 상향 돌파한 봉에서 True. 직전 봉을 보므로 인과적이다."""
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def bars_since(cond: pd.Series) -> pd.Series:
    """조건이 마지막으로 True였던 시점부터 몇 봉 지났는지."""
    idx = pd.Series(np.arange(len(cond)), index=cond.index)
    last = idx.where(cond.fillna(False)).ffill()
    return idx - last


# ─── 편의 ────────────────────────────────────────────────────────

def typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0


def hl2(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"]) / 2.0


def log_returns(s: pd.Series) -> pd.Series:
    return np.log(s / s.shift(1))
