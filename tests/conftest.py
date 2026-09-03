"""테스트 공용 픽스처."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_candles(
    n: int, interval: str = "5min", start: str = "2026-01-01", seed: int = 0,
    derivatives: bool = True,
) -> pd.DataFrame:
    """
    검증용 합성 캔들. 값이 인덱스에서 결정론적으로 나오도록 만들어서
    집계 결과를 손으로 계산해 검증할 수 있게 한다.

    derivatives=True 면 파생 시장 열(미결제약정·롱숏비율·펀딩)도 붙인다.
    인과성 검사가 레지스트리 전체를 훑기 때문에, 이 열이 없으면 파생 지표 9종이
    검사에서 빠지는 게 아니라 입력 부족으로 **에러**가 난다. 기본을 True 로 둔 이유다.
    derivatives=False 는 '파생 데이터가 없을 때도 파이프라인이 도는가'를 볼 때 쓴다.
    """
    idx = pd.date_range(start, periods=n, freq=interval, tz="UTC", name="open_time")
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0, 0.3, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.3, n)
    df = pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": rng.uniform(1, 10, n),
            "quote_volume": rng.uniform(100, 1000, n),
            "trades": rng.integers(10, 100, n),
            "taker_buy_base": rng.uniform(0.5, 5, n),
        },
        index=idx,
    )
    if derivatives:
        df = df.join(_make_derivatives(idx, close, rng, interval))
    return df


def _make_derivatives(idx, close, rng, interval: str) -> pd.DataFrame:
    """
    파생 열의 합성판. 실제 데이터의 **구조**를 흉내 내는 것이 중요하다.

    특히 펀딩은 8시간마다 갱신되어 그 사이 봉에서는 같은 값이 반복된다. 이 계단
    구조가 없으면 롤링 표준편차가 실제보다 커져서 funding_zscore 가 실제와 전혀
    다르게 동작하고, 0으로 나누는 경로를 테스트가 밟지 못한다.
    """
    n = len(idx)
    oi = 10_000.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    step_hours = pd.Timedelta(pd.tseries.frequencies.to_offset(interval)).total_seconds() / 3600
    bars_per_funding = max(1, int(round(8 / step_hours))) if step_hours else 1
    n_blocks = n // bars_per_funding + 1
    funding = np.repeat(rng.normal(0.0001, 0.0002, n_blocks), bars_per_funding)[:n]
    return pd.DataFrame(
        {
            "open_interest": oi,
            "open_interest_value": oi * close,
            "top_trader_account_ls": np.exp(rng.normal(0.4, 0.15, n)),
            "top_trader_position_ls": np.exp(rng.normal(0.1, 0.2, n)),
            "global_account_ls": np.exp(rng.normal(0.6, 0.2, n)),
            "taker_ls_ratio": np.exp(rng.normal(0.0, 0.35, n)),
            "funding_rate": funding,
        },
        index=idx,
    )


@pytest.fixture
def candles_5m() -> pd.DataFrame:
    return make_candles(288, "5min")  # 하루치
