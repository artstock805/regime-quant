"""
바이낸스 로더 검증.

네트워크가 필요한 테스트는 network 마커를 달아 두었다.
  오프라인만:  pytest -m "not network"
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.data.binance import (
    _epoch_to_utc,
    _normalise,
    _read_dump_zip,
    _vision_url,
    fetch_klines,
    interval_timedelta,
)

# 실제 덤프 CSV 한 줄 (12열)
ROW_MS = "1751328000000,58605.4,58720.0,58540.0,58548.6,1234.5,1751331599999,72000000.0,145946,2341.332,137000000.0,0"
HEADER = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"


def _zip_of(lines: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("BTCUSDT-1h-2026-07.csv", "\n".join(lines) + "\n")
    return buf.getvalue()


# ─── 타임스탬프 단위 자동 판별 ────────────────────────────────────

def test_epoch_detects_milliseconds():
    out = _epoch_to_utc(pd.Series([1751328000000, 1751331600000]))
    assert out.iloc[0].year == 2025 or out.iloc[0].year == 2026
    assert out.iloc[0].tz is not None
    assert (out.iloc[1] - out.iloc[0]) == pd.Timedelta("1h")


def test_epoch_detects_microseconds():
    """바이낸스가 일부 선물 덤프의 단위를 마이크로초로 바꾼 적이 있다."""
    ms = _epoch_to_utc(pd.Series([1751328000000, 1751331600000]))
    us = _epoch_to_utc(pd.Series([1751328000000000, 1751331600000000]))
    assert us.iloc[0] == ms.iloc[0], "마이크로초를 밀리초와 같은 시각으로 읽어야 한다"


def test_epoch_detects_seconds():
    s = _epoch_to_utc(pd.Series([1751328000, 1751331600]))
    ms = _epoch_to_utc(pd.Series([1751328000000, 1751331600000]))
    assert s.iloc[0] == ms.iloc[0]


# ─── 헤더 줄 유무 자동 판별 ──────────────────────────────────────

def test_dump_without_header():
    df = _read_dump_zip(_zip_of([ROW_MS, ROW_MS]))
    assert len(df) == 2


def test_dump_with_header():
    """헤더가 있는 파일과 없는 파일이 섞여 있다 — 같은 결과가 나와야 한다."""
    with_h = _read_dump_zip(_zip_of([HEADER, ROW_MS, ROW_MS]))
    without_h = _read_dump_zip(_zip_of([ROW_MS, ROW_MS]))
    assert len(with_h) == len(without_h) == 2
    assert with_h.iat[0, 0] == without_h.iat[0, 0]


def test_normalise_schema():
    df = _normalise(_read_dump_zip(_zip_of([HEADER, ROW_MS])))
    assert df.index.name == "open_time"
    assert df.index.tz is not None
    assert list(df.columns) == [
        "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"
    ]
    assert df["close"].iloc[0] == pytest.approx(58548.6)
    assert df["trades"].iloc[0] == 145946


def test_normalise_drops_duplicate_bars():
    """같은 봉이 두 소스에서 겹쳐 들어오면 하나만 남아야 한다."""
    df = _normalise(_read_dump_zip(_zip_of([ROW_MS, ROW_MS, ROW_MS])))
    assert len(df) == 1


# ─── URL 조립 ────────────────────────────────────────────────────

def test_vision_url_shapes():
    monthly = _vision_url("futures_um", "BTCUSDT", "5m", "2026-07", daily=False)
    assert monthly.endswith("/futures/um/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-2026-07.zip")
    daily = _vision_url("spot", "ETHUSDT", "1h", "2026-07-15", daily=True)
    assert daily.endswith("/spot/daily/klines/ETHUSDT/1h/ETHUSDT-1h-2026-07-15.zip")


def test_interval_timedelta_rejects_unknown():
    assert interval_timedelta("15m") == timedelta(minutes=15)
    with pytest.raises(Exception, match="지원하지 않는 인터벌"):
        interval_timedelta("7m")


# ─── 네트워크 ────────────────────────────────────────────────────

@pytest.mark.network
def test_live_fetch_is_continuous_and_closed():
    now = datetime.now(timezone.utc)
    start = (now - timedelta(hours=8)).replace(minute=0, second=0, microsecond=0)
    df = fetch_klines("BTCUSDT", "5m", start, now)

    assert not df.empty
    gaps = df.index.to_series().diff().dropna().unique()
    assert list(gaps) == [pd.Timedelta("5min")], f"봉 간격에 구멍이 있다: {gaps}"
    # 마지막 봉은 이미 닫혀 있어야 한다
    assert df.index[-1] + timedelta(minutes=5) <= now
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
