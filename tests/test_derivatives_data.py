"""
파생 데이터 정합성 — 봉에 붙이는 경계가 맞는가.

여기가 이 프로젝트에서 미래 참조가 새로 들어올 수 있는 유일한 통로다.
지표는 `test_causality.py` 가 자동으로 막지만, 그 검사는 **데이터가 이미 봉에
올바로 붙어 있다**는 전제 위에서 돈다. 붙이는 단계가 틀리면 모든 지표가
인과성 검사를 통과하면서 동시에 미래를 본다.

metrics 는 5분, 펀딩은 8시간 간격이라 봉보다 성기다. 성긴 값을 촘촘한 봉에
맞추는 방향(앞에서 끌어오는가, 뒤에서 당겨오는가)이 여기서 갈린다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_candles
from src.data.derivatives import (
    ALL_FIELDS, METRICS_COLUMNS, METRICS_FIELDS, STALENESS,
    align_to_bars, coverage, _normalise_funding, _normalise_metrics,
)


def _bars(n=48, interval="1h"):
    return make_candles(n, interval, start="2024-01-01", derivatives=False)


def _deriv(times, **cols):
    return pd.DataFrame(cols, index=pd.DatetimeIndex(times, tz="UTC", name="time"))


# ─── 붙이는 경계 ──────────────────────────────────────────────────

def test_uses_value_at_bar_close_not_after():
    """
    1시간봉 00:00 은 01:00 에 닫힌다. 01:00 까지의 값은 써도 되고,
    01:05 값은 **절대 쓰면 안 된다** — 그 봉이 닫힌 뒤에 나온 정보다.
    """
    bars = _bars(3, "1h")
    d = _deriv(
        ["2024-01-01 00:55", "2024-01-01 01:00", "2024-01-01 01:05"],
        open_interest=[100.0, 200.0, 999.0],
    )
    out = align_to_bars(bars, d, "1h")["open_interest"]
    assert out.iloc[0] == 200.0, "봉 마감 시각(01:00)의 값을 써야 한다"
    # 01:05 값은 01:00 봉(02:00 마감)부터 정당하게 쓸 수 있다. 앞당겨지면 안 된다.
    assert out.iloc[1] == 999.0
    first_use = out.index[out == 999.0].min()
    assert first_use == pd.Timestamp("2024-01-01 01:00", tz="UTC"), (
        "01:05 값이 그보다 이른 봉에 새어 들어갔다"
    )


def test_never_pulls_a_future_value_backward():
    """뒤의 값을 앞으로 당기면(bfill) 미래를 보게 된다."""
    bars = _bars(3, "1h")
    d = _deriv(["2024-01-01 02:30"], open_interest=[500.0])
    out = align_to_bars(bars, d, "1h")
    # 02:30 값은 03:00 에 닫히는 봉(=02:00)부터 쓸 수 있다
    assert pd.isna(out["open_interest"].iloc[0])
    assert pd.isna(out["open_interest"].iloc[1])
    assert out["open_interest"].iloc[2] == 500.0


def test_truncating_bars_does_not_change_earlier_values():
    """인과성의 핵심 — 뒤를 잘라내도 앞의 값이 바뀌면 안 된다."""
    bars = _bars(24, "1h")
    times = pd.date_range("2024-01-01", periods=200, freq="15min", tz="UTC")
    d = _deriv(times, open_interest=np.linspace(100, 300, 200))

    full = align_to_bars(bars, d, "1h")["open_interest"]
    cut = align_to_bars(bars.iloc[:10], d, "1h")["open_interest"]
    pd.testing.assert_series_equal(full.iloc[:10], cut, check_names=False)


# ─── 성긴 소스와 촘촘한 소스가 섞일 때 ─────────────────────────────

def test_sparse_and_dense_sources_both_survive_the_join():
    """
    실제로 겪은 버그. 5분 metrics 와 8시간 펀딩을 옆으로 합쳐 두면 대부분의 행에서
    펀딩이 NaN 이고, merge_asof 는 직전 '행'을 가져오므로 펀딩 커버리지가 2%로
    떨어졌다. 열마다 자기 타임라인으로 붙여야 한다.
    """
    bars = _bars(24, "1h")
    dense = _deriv(pd.date_range("2024-01-01", periods=300, freq="5min", tz="UTC"),
                   open_interest=np.arange(300, dtype=float))
    sparse = _deriv(pd.date_range("2024-01-01", periods=4, freq="8h", tz="UTC"),
                    funding_rate=[0.0001, 0.0002, -0.0001, 0.0003])
    merged = pd.concat([dense, sparse], axis=1, sort=True)

    out = align_to_bars(bars, merged, "1h")
    assert out["open_interest"].notna().mean() > 0.95
    assert out["funding_rate"].notna().mean() > 0.95, "성긴 열이 조인에서 사라졌다"


def test_stale_values_are_not_carried_past_tolerance():
    """
    데이터가 끊긴 구간에서 옛날 값을 조용히 재사용하는 것이 가장 위험한 실패다 —
    백테스트는 멀쩡히 돌아가고 결과만 거짓이 된다.
    """
    bars = _bars(48, "1h")
    d = _deriv(["2024-01-01 00:30"], open_interest=[100.0])
    out = align_to_bars(bars, d, "1h")
    limit = STALENESS["open_interest"]
    used = out["open_interest"].notna()
    last_used = out.index[used].max() + pd.Timedelta("1h")
    assert last_used - pd.Timestamp("2024-01-01 00:30", tz="UTC") <= limit


def test_funding_tolerance_is_wider_than_metrics():
    """펀딩은 8시간마다 오므로 metrics 와 같은 잣대를 쓰면 대부분 버려진다."""
    assert STALENESS["funding_rate"] > max(STALENESS[f] for f in METRICS_FIELDS)
    assert STALENESS["funding_rate"] >= pd.Timedelta("8h")


# ─── 원본 파싱 ────────────────────────────────────────────────────

def test_metrics_column_names_are_renamed():
    raw = pd.DataFrame({
        "create_time": ["2024-01-01 00:00:00", "2024-01-01 00:05:00"],
        "symbol": ["BTCUSDT", "BTCUSDT"],
        **{k: ["1.0", "2.0"] for k in METRICS_COLUMNS},
    })
    out = _normalise_metrics(raw)
    assert list(out.columns) == list(METRICS_FIELDS)
    assert out.index.tz is not None, "UTC 로 정규화돼야 한다"


def test_metrics_duplicate_timestamps_keep_the_last():
    raw = pd.DataFrame({
        "create_time": ["2024-01-01 00:00:00"] * 2,
        **{k: ["1.0", "9.0"] for k in METRICS_COLUMNS},
    })
    out = _normalise_metrics(raw)
    assert len(out) == 1
    assert out["open_interest"].iloc[0] == 9.0


def test_funding_accepts_either_column_spelling():
    """아카이브와 REST 가 열 이름을 다르게 쓴다. 둘 다 읽혀야 한다."""
    a = _normalise_funding(pd.DataFrame({"calc_time": ["1704067200000"],
                                         "last_funding_rate": ["0.0001"]}))
    b = _normalise_funding(pd.DataFrame({"fundingTime": ["1704067200000"],
                                         "fundingRate": ["0.0001"]}))
    assert a.index[0] == b.index[0]
    assert float(a["funding_rate"].iloc[0]) == 0.0001


# ─── 없을 때도 죽지 않아야 한다 ────────────────────────────────────

def test_empty_derivatives_returns_bars_unchanged():
    bars = _bars(10, "1h")
    out = align_to_bars(bars, pd.DataFrame(), "1h")
    pd.testing.assert_frame_equal(out, bars)


def test_coverage_reports_zero_for_missing_fields():
    cov = coverage(_bars(10, "1h")).set_index("필드")
    assert (cov["커버리지_%"] == 0).all()
    assert set(cov.index) == set(ALL_FIELDS)


@pytest.mark.parametrize("interval", ["5m", "15m", "1h", "4h"])
def test_alignment_works_across_timeframes(interval):
    freq = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}[interval]
    bars = make_candles(30, freq, start="2024-01-01", derivatives=False)
    d = _deriv(pd.date_range("2024-01-01", periods=2000, freq="5min", tz="UTC"),
               open_interest=np.arange(2000, dtype=float))
    out = align_to_bars(bars, d, interval)
    assert out["open_interest"].notna().sum() >= len(bars) - 1
    # 값이 단조 증가해야 한다 — 시간 순서가 뒤섞이지 않았다는 뜻
    v = out["open_interest"].dropna()
    assert (v.diff().dropna() > 0).all()


# ─── 재시도와 헛요청 방지 ─────────────────────────────────────────

def test_transient_5xx_is_retried_then_succeeds(monkeypatch):
    """
    아카이브 서버는 간헐적으로 503 을 낸다. 재시도가 없으면 한 장의 실패가
    129종목 수집 전체를 날린다 — 실제로 4시간봉 검증이 BATUSDT 에서 죽었다.
    """
    import io, zipfile
    import requests
    from src.data import derivatives as dv

    good = io.BytesIO()
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("x.csv", "create_time,sum_open_interest\n2024-01-01 00:00:00,100\n")
    payload = good.getvalue()

    calls = {"n": 0}

    class Resp:
        def __init__(self, status, content=b""):
            self.status_code, self.content = status, content
        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(str(self.status_code), response=self)

    class Sess:
        def get(self, url, timeout=None):
            calls["n"] += 1
            return Resp(503) if calls["n"] < 3 else Resp(200, payload)

    monkeypatch.setattr(dv.time, "sleep", lambda s: None)
    out = dv._fetch_zip(Sess(), "http://x/y.zip")
    assert calls["n"] == 3, f"재시도가 일어나지 않았다 (호출 {calls['n']}회)"
    assert out is not None and len(out) == 1


def test_permanent_error_is_not_retried(monkeypatch):
    """403 같은 영구 오류를 재시도하면 시간만 버리고 원인을 가린다."""
    import requests
    from src.data import derivatives as dv

    calls = {"n": 0}

    class Resp:
        status_code, content = 403, b""
        def raise_for_status(self):
            raise requests.HTTPError("403", response=self)

    class Sess:
        def get(self, url, timeout=None):
            calls["n"] += 1
            return Resp()

    monkeypatch.setattr(dv.time, "sleep", lambda s: None)
    with pytest.raises(requests.HTTPError):
        dv._fetch_zip(Sess(), "http://x/y.zip")
    assert calls["n"] == 1


def test_404_is_not_an_error():
    """상장 이전 날짜는 정상적인 '없음'이다."""
    from src.data import derivatives as dv

    class Resp:
        status_code, content = 404, b""
        def raise_for_status(self):
            raise AssertionError("404 에 raise_for_status 를 부르면 안 된다")

    class Sess:
        def get(self, url, timeout=None):
            return Resp()

    assert dv._fetch_zip(Sess(), "http://x/y.zip") is None


def test_probe_sidecar_stops_refetching_absent_ranges(tmp_path):
    """
    캐시가 2021-12 부터인데 시작이 2020-10 이면, 없는 430일을 실행마다 다시 훑는다.
    404 는 조용히 넘어가므로 눈에 안 띄고 시간만 먹는다.
    """
    from datetime import datetime, timezone
    from src.data.derivatives import _load_probe, _save_probe

    cache = tmp_path / "BTCUSDT-metrics.parquet"
    assert _load_probe(cache) is None
    when = datetime(2020, 10, 1, tzinfo=timezone.utc)
    _save_probe(cache, when)
    assert _load_probe(cache) == when


def test_corrupt_probe_sidecar_is_ignored(tmp_path):
    """사이드카가 깨져도 죽지 않아야 한다 — 다시 훑으면 되는 일이다."""
    from src.data.derivatives import _load_probe, _probe_path

    cache = tmp_path / "X-metrics.parquet"
    _probe_path(cache).write_text("{ not json", encoding="utf-8")
    assert _load_probe(cache) is None
