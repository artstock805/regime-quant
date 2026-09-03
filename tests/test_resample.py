"""리샘플링 검증. 특히 미래 참조가 없는지를 본다."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import make_candles
from src.data.resample import align_higher, infer_interval, resample_ohlcv


def test_infer_interval():
    assert infer_interval(make_candles(20, "5min")) == pd.Timedelta("5min")
    assert infer_interval(make_candles(20, "1h")) == pd.Timedelta("1h")


def test_resample_aggregates_ohlc_correctly():
    src = make_candles(24, "5min")            # 2시간치
    out = resample_ohlcv(src, "1h")

    assert len(out) == 2
    first = src.iloc[:12]
    assert out["open"].iloc[0] == pytest.approx(first["open"].iloc[0])
    assert out["close"].iloc[0] == pytest.approx(first["close"].iloc[-1])
    assert out["high"].iloc[0] == pytest.approx(first["high"].max())
    assert out["low"].iloc[0] == pytest.approx(first["low"].min())
    assert out["volume"].iloc[0] == pytest.approx(first["volume"].sum())
    assert out["trades"].iloc[0] == first["trades"].sum()


def test_resample_labels_bars_by_open_time():
    """바이낸스와 같은 규칙 — 봉 라벨은 여는 시각이어야 한다."""
    src = make_candles(24, "5min", start="2026-01-01 00:00")
    out = resample_ohlcv(src, "1h")
    assert str(out.index[0]) == "2026-01-01 00:00:00+00:00"
    assert str(out.index[1]) == "2026-01-01 01:00:00+00:00"


def test_resample_drops_partial_last_bar():
    src = make_candles(20, "5min")            # 1시간 40분 → 1h 봉 하나 + 미완성 하나
    full = resample_ohlcv(src, "1h", drop_partial=True)
    partial = resample_ohlcv(src, "1h", drop_partial=False)

    assert len(full) == 1, "미완성 1h 봉이 남아있다"
    assert len(partial) == 2
    # 미완성 봉의 종가는 실제 1시간 종가가 아니다 — 이걸 신호에 쓰면 재현이 안 된다
    assert partial["close"].iloc[-1] == pytest.approx(src["close"].iloc[-1])


def test_resample_rejects_downsampling():
    with pytest.raises(ValueError, match="상위 → 하위"):
        resample_ohlcv(make_candles(10, "1h"), "5m")


def test_resample_rejects_non_multiple_interval():
    """3m → 5m 은 배수 관계가 아니다. 억지로 올리면 봉 경계가 어긋난다."""
    with pytest.raises(ValueError, match="배수"):
        resample_ohlcv(make_candles(30, "3min"), "5m")


def test_resample_15m_to_2h_is_allowed():
    out = resample_ohlcv(make_candles(48, "15min"), "2h")
    assert len(out) == 6


def test_align_higher_does_not_leak_future():
    """
    핵심 검증. 1h 봉 11:00 은 12:00 에 닫히므로, 11:05 의 5m 봉이 볼 수 있는
    최신 1h 종가는 10:00 봉의 종가여야 한다. 11:00 봉이 보이면 미래 참조다.
    """
    lower = make_candles(48, "5min", start="2026-01-01 08:00", seed=1)
    higher = resample_ohlcv(make_candles(576, "5min", start="2026-01-01 00:00", seed=1), "1h")

    aligned = align_higher(lower, higher, ["close"])

    at_1105 = pd.Timestamp("2026-01-01 11:05", tz="UTC")
    bar_1000 = pd.Timestamp("2026-01-01 10:00", tz="UTC")
    bar_1100 = pd.Timestamp("2026-01-01 11:00", tz="UTC")

    seen = aligned.loc[at_1105, "close"]
    assert seen == pytest.approx(higher.loc[bar_1000, "close"]), "닫힌 직전 1h 봉을 봐야 한다"
    assert seen != pytest.approx(higher.loc[bar_1100, "close"]), "아직 안 닫힌 1h 봉을 훔쳐봤다"


def test_align_higher_updates_right_after_close():
    """11:00 봉은 12:00 에 닫히므로 12:00 봉부터 보여야 한다."""
    lower = make_candles(96, "5min", start="2026-01-01 08:00", seed=1)   # 08:00~15:55
    higher = resample_ohlcv(make_candles(576, "5min", start="2026-01-01 00:00", seed=1), "1h")
    aligned = align_higher(lower, higher, ["close"])

    bar_1100 = pd.Timestamp("2026-01-01 11:00", tz="UTC")
    assert aligned.loc[pd.Timestamp("2026-01-01 11:55", tz="UTC"), "close"] != pytest.approx(
        higher.loc[bar_1100, "close"]
    )
    assert aligned.loc[pd.Timestamp("2026-01-01 12:00", tz="UTC"), "close"] == pytest.approx(
        higher.loc[bar_1100, "close"]
    )


def test_resample_skips_gaps_instead_of_inventing_bars():
    src = make_candles(24, "5min")
    holed = pd.concat([src.iloc[:6], src.iloc[18:]])   # 30분 구멍
    out = resample_ohlcv(holed, "1h", drop_partial=False)
    assert not out.isna().any().any(), "구멍 구간에 NaN 봉을 만들어냈다"
