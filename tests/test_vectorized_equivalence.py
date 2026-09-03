"""
벡터화한 프리미티브가 소박한 구현과 같은 값을 내는지 검증한다.

linreg_slope / linreg_r2 / percent_rank / aroon 은 rolling().apply() 파이썬 루프에서
스트라이드 뷰·컨볼루션 기반으로 바꿨다. 빠르지만 틀린 지표는 느린 지표보다 나쁘므로,
느린 참조 구현을 테스트 안에 남겨 두고 값이 일치하는지 비교한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_candles
from src.indicators.core import linreg_r2, linreg_slope, percent_rank
from src.indicators.trend import aroon


# ─── 참조(느린) 구현 ─────────────────────────────────────────────

def ref_percent_rank(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period, min_periods=period).apply(
        lambda w: float((w[:-1] < w[-1]).mean() * 100.0), raw=True
    )


def ref_linreg_slope(s: pd.Series, period: int) -> pd.Series:
    x = np.arange(period, dtype=float)
    x_dev = x - x.mean()
    denom = float((x_dev**2).sum())

    def _slope(w: np.ndarray) -> float:
        beta = float(np.dot(x_dev, w - w.mean()) / denom)
        last = w[-1]
        return beta / last * 100.0 if last else np.nan

    return s.rolling(period, min_periods=period).apply(_slope, raw=True)


def ref_linreg_r2(s: pd.Series, period: int) -> pd.Series:
    x = np.arange(period, dtype=float)
    x_dev = x - x.mean()

    def _r2(w: np.ndarray) -> float:
        y_dev = w - w.mean()
        ss_tot = float((y_dev**2).sum())
        if ss_tot == 0:
            return np.nan
        beta = float(np.dot(x_dev, y_dev) / (x_dev**2).sum())
        ss_res = float(((y_dev - beta * x_dev) ** 2).sum())
        return 1.0 - ss_res / ss_tot

    return s.rolling(period, min_periods=period).apply(_r2, raw=True)


def ref_aroon(df: pd.DataFrame, period: int) -> pd.DataFrame:
    win = period + 1
    up = df["high"].rolling(win, min_periods=win).apply(
        lambda w: float(np.argmax(w)) / period * 100.0, raw=True
    )
    down = df["low"].rolling(win, min_periods=win).apply(
        lambda w: float(np.argmin(w)) / period * 100.0, raw=True
    )
    return pd.DataFrame({"aroon_up": up, "aroon_down": down, "aroon_osc": up - down})


# ─── 비교 ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def close() -> pd.Series:
    return make_candles(600, "5min", seed=7)["close"]


@pytest.mark.parametrize("period", [5, 14, 50, 200])
def test_percent_rank_matches_reference(close: pd.Series, period: int):
    pd.testing.assert_series_equal(
        percent_rank(close, period), ref_percent_rank(close, period), rtol=1e-9
    )


@pytest.mark.parametrize("period", [5, 14, 50, 200])
def test_linreg_slope_matches_reference(close: pd.Series, period: int):
    pd.testing.assert_series_equal(
        linreg_slope(close, period), ref_linreg_slope(close, period), rtol=1e-7
    )


@pytest.mark.parametrize("period", [5, 14, 50, 200])
def test_linreg_r2_matches_reference(close: pd.Series, period: int):
    # 닫힌 형태는 ss_tot = Σy² - nȳ² 를 쓰므로 큰 값에서 상쇄오차가 조금 생긴다
    pd.testing.assert_series_equal(
        linreg_r2(close, period), ref_linreg_r2(close, period), rtol=1e-6, atol=1e-9
    )


@pytest.mark.parametrize("period", [7, 25])
def test_aroon_matches_reference(period: int):
    df = make_candles(400, "5min", seed=11)
    pd.testing.assert_frame_equal(aroon(df, period), ref_aroon(df, period), rtol=1e-9)


def test_nan_windows_stay_nan():
    """윈도우 안에 NaN이 하나라도 있으면 결과는 NaN이어야 한다."""
    s = make_candles(200, "5min", seed=3)["close"].copy()
    s.iloc[100] = np.nan

    for fn, ref in ((percent_rank, ref_percent_rank),
                    (linreg_slope, ref_linreg_slope),
                    (linreg_r2, ref_linreg_r2)):
        fast, slow = fn(s, 20), ref(s, 20)
        # NaN 위치가 같아야 한다
        assert fast.isna().equals(slow.isna()), f"{fn.__name__}: NaN 전파가 다릅니다"
        # NaN 을 지난 뒤 20봉 구간은 확실히 NaN
        assert fast.iloc[100:120].isna().all(), f"{fn.__name__}: NaN 윈도우를 값으로 채웠습니다"


def test_short_input_returns_all_nan():
    """기간보다 짧은 입력에서 터지지 않고 NaN을 돌려준다."""
    s = make_candles(10, "5min")["close"]
    for fn in (percent_rank, linreg_slope, linreg_r2):
        out = fn(s, 50)
        assert len(out) == 10 and out.isna().all()
