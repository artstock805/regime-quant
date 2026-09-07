"""
틱 → 봉 집계.

**봉 라벨이 한 칸 밀리면 미래 참조가 된다.** 틱 특징이 다음 봉에 붙으면 그
봉의 종가를 알기 전에 알 수 없는 정보를 쓰는 것이고, 백테스트가 거짓이 된다.
그래서 정렬을 가장 먼저 잠근다.

캐시는 데이터 드라이브로 보낸다 (C 가 만성 부족). 진행 중인 구간을 파일로
굳혀 두면 다음에 그 조각을 그대로 읽어 최근 데이터가 영구히 빠지므로, 그
경계도 테스트한다.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.fx import ticks


def make_ticks(rows):
    """(초, bid, ask) 목록 → MT5 가 주는 모양의 틱 프레임."""
    base = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)   # 월요일 10:00
    return pd.DataFrame({
        "time_msc": [int((base + timedelta(seconds=s)).timestamp() * 1000)
                     for s, _, _ in rows],
        "bid": [b for _, b, _ in rows],
        "ask": [a for _, _, a in rows],
        "flags": [1122] * len(rows),
    })


# ─── 정렬 ────────────────────────────────────────────────────────

def test_bar_label_is_the_bar_open_not_the_close():
    """
    `mt5.bars()` 인덱스는 봉의 **시작 시각**이다. 여기서 마감 시각으로 라벨을
    달면 특징이 다음 봉에 붙어 미래 참조가 된다.
    """
    rows = [(0, 1.1000, 1.1002), (60, 1.1001, 1.1003), (120, 1.1002, 1.1004)]
    f = ticks.bar_features(make_ticks(rows), "1h")
    assert len(f) == 1
    assert f.index[0] == pd.Timestamp("2026-03-02 10:00", tz="UTC")


def test_ticks_are_binned_into_the_hour_they_belong_to():
    rows = [(0, 1.1000, 1.1002),        # 10시
            (3600, 1.1005, 1.1007),     # 11시
            (3660, 1.1006, 1.1008)]     # 11시
    f = ticks.bar_features(make_ticks(rows), "1h")
    assert list(f.index) == [pd.Timestamp("2026-03-02 10:00", tz="UTC"),
                            pd.Timestamp("2026-03-02 11:00", tz="UTC")]
    assert f["tick_count"].tolist() == [1, 2]


# ─── 불균형 계산 ─────────────────────────────────────────────────

def test_all_upticks_give_plus_one():
    rows = [(i, 1.1000 + i * 0.0001, 1.1002 + i * 0.0001) for i in range(5)]
    f = ticks.bar_features(make_ticks(rows), "1h")
    assert f["tick_imbalance"].iloc[0] == pytest.approx(1.0)
    assert f["tick_up"].iloc[0] == 4.0        # 첫 틱은 비교 대상이 없다
    assert f["tick_down"].iloc[0] == 0.0


def test_all_downticks_give_minus_one():
    rows = [(i, 1.1000 - i * 0.0001, 1.1002 - i * 0.0001) for i in range(5)]
    f = ticks.bar_features(make_ticks(rows), "1h")
    assert f["tick_imbalance"].iloc[0] == pytest.approx(-1.0)


def test_a_balanced_bar_gives_zero():
    rows = [(0, 1.1000, 1.1002), (1, 1.1001, 1.1003),
            (2, 1.1000, 1.1002), (3, 1.1001, 1.1003), (4, 1.1000, 1.1002)]
    f = ticks.bar_features(make_ticks(rows), "1h")
    assert f["tick_imbalance"].iloc[0] == pytest.approx(0.0)


def test_flat_ticks_are_not_counted_as_direction():
    """
    호가가 안 바뀐 틱은 방향이 없다. 상승이나 하락으로 세면 불균형이
    조용히 한쪽으로 기운다.
    """
    rows = [(0, 1.1000, 1.1002), (1, 1.1000, 1.1002), (2, 1.1000, 1.1002),
            (3, 1.1001, 1.1003)]
    f = ticks.bar_features(make_ticks(rows), "1h")
    assert f["tick_up"].iloc[0] == 1.0
    assert f["tick_down"].iloc[0] == 0.0
    assert f["tick_imbalance"].iloc[0] == pytest.approx(1.0)


def test_a_bar_where_nothing_moved_is_zero_not_nan():
    """
    움직인 틱이 하나도 없으면 0 으로 나누게 된다. NaN 이 지표로 흘러가면
    뒤의 z-score 가 통째로 죽는다.
    """
    rows = [(0, 1.1000, 1.1002), (1, 1.1000, 1.1002)]
    f = ticks.bar_features(make_ticks(rows), "1h")
    assert f["tick_imbalance"].iloc[0] == 0.0
    assert np.isfinite(f["tick_imbalance"]).all()


# ─── 스프레드·변동성 ─────────────────────────────────────────────

def test_spread_is_measured_from_bid_and_ask():
    rows = [(0, 1.1000, 1.1002), (1, 1.1000, 1.1005)]
    f = ticks.bar_features(make_ticks(rows), "1h")
    assert f["tick_spread_mean"].iloc[0] == pytest.approx(0.00035)
    assert f["tick_spread_max"].iloc[0] == pytest.approx(0.0005)


# ─── 빈 입력 ─────────────────────────────────────────────────────

def test_no_ticks_gives_an_empty_frame_not_a_crash():
    """주말 구간은 정상적으로 비어 있다. 예외가 아니라 빈 프레임이어야 한다."""
    empty = pd.DataFrame(columns=["time_msc", "bid", "ask", "flags"])
    assert ticks.bar_features(empty, "1h").empty


def test_an_unknown_interval_fails_loudly():
    from src.fx.mt5 import MT5Error
    rows = [(0, 1.1000, 1.1002)]
    with pytest.raises(MT5Error):
        ticks.bar_features(make_ticks(rows), "3h")


# ─── 봉에 붙이기 ─────────────────────────────────────────────────

def test_attach_leaves_gaps_as_nan_not_zero(monkeypatch):
    """
    **0 은 "쏠림 없음", NaN 은 "모름"이다.** 틱을 못 받은 구간을 0 으로 채우면
    중립 신호로 읽혀 지표가 조용히 희석된다.
    """
    idx = pd.date_range("2026-03-02 10:00", periods=3, freq="1h", tz="UTC")
    bars = pd.DataFrame({"close": [1.1, 1.1, 1.1]}, index=idx)

    rows = [(0, 1.1000, 1.1002), (60, 1.1001, 1.1003)]     # 10시만 있다
    monkeypatch.setattr(ticks, "load_ticks", lambda *a, **k: make_ticks(rows))

    out = ticks.attach(bars, "EURUSD", "1h", days=7)
    assert out.index.equals(bars.index)
    assert out["tick_imbalance"].notna().sum() == 1
    assert out["tick_imbalance"].isna().sum() == 2      # 0 으로 채우지 않았다


def test_attach_with_no_ticks_returns_an_aligned_empty_frame(monkeypatch):
    idx = pd.date_range("2026-03-02 10:00", periods=3, freq="1h", tz="UTC")
    bars = pd.DataFrame({"close": [1.1, 1.1, 1.1]}, index=idx)
    monkeypatch.setattr(
        ticks, "load_ticks",
        lambda *a, **k: pd.DataFrame(columns=["time_msc", "bid", "ask", "flags"]))
    out = ticks.attach(bars, "EURUSD", "1h", days=7)
    assert out.index.equals(bars.index)


# ─── 캐시 ────────────────────────────────────────────────────────

def test_cache_goes_to_the_data_drive_not_c():
    """
    C 는 만성적으로 부족하다 (이 PC 는 110GB SSD). 180일 EURUSD 틱만 313MB 라
    통화쌍마다 쌓이면 금방 문제가 된다.
    """
    root = ticks.cache_root()
    assert root.name == "fx-ticks"


def test_the_in_progress_window_is_not_cached(monkeypatch, tmp_path):
    """
    아직 틱이 더 들어올 구간을 파일로 굳히면 다음에 그 조각을 그대로 읽어
    **최근 데이터가 영구히 빠진다.** 조용히 낡은 값을 쓰는 쪽이 더 나쁘다.
    """
    monkeypatch.setattr(ticks, "cache_root", lambda: tmp_path)
    calls = {"n": 0}

    def fake_fetch(symbol, start, end):
        calls["n"] += 1
        return make_ticks([(0, 1.1000, 1.1002), (1, 1.1001, 1.1003)])

    monkeypatch.setattr(ticks, "_fetch_range", fake_fetch)
    ticks.load_ticks("EURUSD", days=3)
    written = list((tmp_path / "EURUSD").glob("*.parquet"))
    assert written == []          # 3일이면 전부 진행 중인 주다 — 캐시 안 함
    assert calls["n"] >= 1
