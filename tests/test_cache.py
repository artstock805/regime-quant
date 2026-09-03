"""캐시 왕복 검증. parquet을 거쳐도 시간대와 정렬이 살아있어야 한다."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import make_candles
from src.data.cache import cache_path, load_cached, save_cached


def test_roundtrip_preserves_utc_and_order(tmp_path):
    src = make_candles(100, "5min")
    p = save_cached(src, tmp_path / "BTCUSDT_5m.parquet")
    back = load_cached(p)

    assert back.index.tz is not None, "parquet 왕복에서 시간대가 사라졌다"
    assert back.index.name == "open_time"
    assert back.index.equals(src.index)
    pd.testing.assert_frame_equal(back, src, check_freq=False)


def test_save_sorts_and_dedupes():
    """저장 직전 정렬·중복제거를 놓치면 같은 신호를 두 번 내는 사고로 이어진다."""
    src = make_candles(20, "5min")
    scrambled = pd.concat([src.iloc[10:], src.iloc[:15]])   # 순서 뒤섞임 + 5봉 중복
    assert len(scrambled) == 25

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        back = load_cached(save_cached(scrambled, Path(d) / "x.parquet"))

    assert len(back) == 20
    assert back.index.is_monotonic_increasing
    assert not back.index.duplicated().any()


def test_load_missing_returns_empty(tmp_path):
    out = load_cached(tmp_path / "nope.parquet")
    assert out.empty
    assert "close" in out.columns


def test_cache_path_layout():
    p = cache_path("data/klines", "futures_um", "BTCUSDT", "5m")
    assert p.as_posix().endswith("data/klines/futures_um/BTCUSDT_5m.parquet")


def test_refresh_preserves_data_outside_the_requested_range(tmp_path, monkeypatch):
    """
    좁은 구간에 refresh 를 걸었다고 그 밖의 캐시가 날아가면 안 된다.
    실제로 120일 구간에 refresh 를 걸어 51,840봉이 2,855봉으로 잘린 사고가 있었다.
    """
    import src.data.cache as cache_mod
    from conftest import make_candles

    wide = make_candles(500, "1h", start="2026-01-01")
    save_cached(wide, cache_path(tmp_path, "m", "T", "1h"))

    # 뒤쪽 50봉만 새로 받아오는 상황을 흉내낸다
    narrow = wide.iloc[-50:].copy()
    narrow["close"] = narrow["close"] + 1.0          # 새로 받은 값은 달라야 한다
    monkeypatch.setattr(cache_mod, "fetch_klines",
                        lambda *a, **k: narrow)

    out = cache_mod.get_klines(
        "T", "1h", narrow.index[0], narrow.index[-1] + pd.Timedelta("1h"),
        market="m", klines_dir=tmp_path, refresh=True,
    )

    stored = load_cached(cache_path(tmp_path, "m", "T", "1h"))
    assert len(stored) == 500, f"구간 밖 캐시가 사라졌다 ({len(stored)}봉 남음)"
    assert stored.index[0] == wide.index[0]
    # 요청 구간 안은 새로 받은 값으로 갱신돼야 한다
    assert stored["close"].iloc[-1] == pytest.approx(narrow["close"].iloc[-1])
    assert len(out) == 50


def test_incremental_fetch_also_preserves_history(tmp_path, monkeypatch):
    """refresh 없이 꼬리만 이어 받을 때도 앞쪽이 온전해야 한다."""
    import src.data.cache as cache_mod
    from conftest import make_candles

    wide = make_candles(400, "1h", start="2026-01-01")
    save_cached(wide.iloc[:300], cache_path(tmp_path, "m", "T", "1h"))
    monkeypatch.setattr(cache_mod, "fetch_klines", lambda *a, **k: wide.iloc[300:])

    cache_mod.get_klines("T", "1h", wide.index[0], wide.index[-1] + pd.Timedelta("1h"),
                         market="m", klines_dir=tmp_path)
    stored = load_cached(cache_path(tmp_path, "m", "T", "1h"))
    assert len(stored) == 400
    assert stored.index[0] == wide.index[0] and stored.index[-1] == wide.index[-1]


def test_refresh_overwrites_values_inside_the_range(tmp_path, monkeypatch):
    """겹치는 구간은 새로 받은 값이 이겨야 한다 (잘못된 옛 데이터 교정 용도)."""
    import src.data.cache as cache_mod
    from conftest import make_candles

    wide = make_candles(200, "1h", start="2026-01-01")
    save_cached(wide, cache_path(tmp_path, "m", "T", "1h"))

    fixed = wide.iloc[100:150].copy()
    fixed["close"] = 99999.0
    monkeypatch.setattr(cache_mod, "fetch_klines", lambda *a, **k: fixed)

    cache_mod.get_klines("T", "1h", fixed.index[0], fixed.index[-1] + pd.Timedelta("1h"),
                         market="m", klines_dir=tmp_path, refresh=True)
    stored = load_cached(cache_path(tmp_path, "m", "T", "1h"))
    assert len(stored) == 200, "구간 밖이 보존돼야 한다"
    assert (stored.loc[fixed.index, "close"] == 99999.0).all(), "구간 안은 갱신돼야 한다"
    assert stored["close"].iloc[0] == pytest.approx(wide["close"].iloc[0]), "구간 밖은 그대로"


def test_overlapping_merge_is_deterministic(tmp_path, monkeypatch):
    """
    같은 타임스탬프가 겹칠 때 어느 값이 남는지가 확정적이어야 한다.

    sort_index() 기본값은 quicksort 로 **불안정 정렬**이라, 겹치는 봉의 원래 순서가
    보존되지 않고 keep="last" 가 임의의 행을 골랐다. 월간덤프·일간덤프·REST 가
    겹치는 구간에서 어느 소스가 이길지 비결정적이었다는 뜻이다.
    """
    import src.data.cache as cache_mod
    from conftest import make_candles

    wide = make_candles(200, "1h", start="2026-01-01")
    fixed = wide.iloc[100:150].copy()
    fixed["close"] = 99999.0

    for _ in range(5):          # 여러 번 돌려도 같은 답이어야 한다
        path = cache_path(tmp_path, "m", "T", "1h")
        if path.exists():
            path.unlink()
        save_cached(wide, path)
        monkeypatch.setattr(cache_mod, "fetch_klines", lambda *a, **k: fixed)
        cache_mod.get_klines("T", "1h", fixed.index[0],
                             fixed.index[-1] + pd.Timedelta("1h"),
                             market="m", klines_dir=tmp_path, refresh=True)
        stored = load_cached(path)
        assert len(stored) == 200, "구간 밖이 사라졌다"
        assert (stored.loc[fixed.index, "close"] == 99999.0).all(), (
            "겹치는 구간에서 새 값이 이기지 않았다 — 불안정 정렬"
        )


def test_save_cached_keeps_the_last_duplicate(tmp_path):
    """중복 제거는 '나중에 넣은 것'을 남겨야 한다 — 신선한 데이터가 이긴다."""
    from conftest import make_candles

    a = make_candles(50, "1h", start="2026-01-01")
    b = a.iloc[20:30].copy()
    b["close"] = -1.0
    stored = load_cached(save_cached(pd.concat([a, b]), tmp_path / "x.parquet"))
    assert len(stored) == 50
    assert (stored.loc[b.index, "close"] == -1.0).all()
