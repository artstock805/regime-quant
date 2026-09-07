"""
FX 엔진 스케줄러.

**이게 없어서 "시작"을 눌러도 아무 일이 없었다.** `run_once()` 주석은
"스케줄러가 봉 마감마다 부른다"고 하는데 FX 쪽에는 그 스케줄러가 없었고,
`start()` 는 running 플래그만 세웠다. 부르는 곳은 화면의 「한 바퀴」 버튼뿐이었다.

**시계 계산으로 봉 마감을 맞추지 않는다.** MT5 는 서버 시간을 주고(GMT+2/+3,
서머타임으로 바뀐다) 주말에는 장이 닫힌다. "다음 봉 마감까지 몇 초"를 계산하면
그 셋을 다 맞춰야 하고 하나만 틀려도 같은 봉을 두 번 보거나 건너뛴다.
**새 봉이 실제로 나타났는지 보는 쪽**이 그 전부를 원천 회피한다.

여기서 지키는 것
    · 시작하면 루프가 실제로 돈다 (플래그만 세우지 않는다)
    · 새 봉이 떴을 때만 한 바퀴 돈다 — 같은 봉으로 두 번 주문하지 않는다
    · **시작 직후 첫 봉으로는 진입하지 않는다** (이미 지나간 봉이다)
    · 한 바퀴가 터져도 루프는 살아 있다
    · 주말(새 봉 없음)에는 조용히 아무것도 하지 않는다
    · 정지·킬이 스레드를 실제로 세운다
"""

from __future__ import annotations

import threading
import time

import pytest

from src.fx import engine as eng


class FakeEngine(eng.FxTrader):
    """MT5 도 네트워크도 타지 않는 엔진. 루프 동작만 본다."""

    def __init__(self):
        # 부모 __init__ 은 설정을 읽으므로 필요한 것만 손으로 세운다.
        self.state = eng.EngineState()
        self._lock = threading.Lock()
        self._known = {}
        self._stop = threading.Event()
        self._thread = None
        self.journal = _FakeJournal()
        self.bar = "2026-09-07T05:00:00+00:00"
        self.ran = []
        self.explode = False

    def _newest_bar(self):
        return self.bar

    def run_once(self):
        if self.explode:
            raise RuntimeError("한 바퀴 실패")
        self.ran.append(self.bar)
        return {}


class _FakeJournal:
    def __init__(self):
        self.rows = []

    def write(self, kind, **fields):
        self.rows.append({"kind": kind, **fields})
        return {}


@pytest.fixture
def fast_poll(monkeypatch):
    """폴링을 짧게 줄여 테스트가 초 단위로 끝나게 한다."""
    monkeypatch.setattr(eng, "POLL_SECONDS", 0.05)


def wait_for(fn, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if fn():
            return True
        time.sleep(0.01)
    return False


# ─── 루프가 실제로 돈다 ──────────────────────────────────────────

def test_the_loop_actually_runs(fast_poll):
    """
    **플래그만 세우면 안 된다.** 예전에는 running=True 만 켜고 아무것도
    부르지 않아서, 시작을 누르고 나가면 매매가 한 건도 나가지 않았다.
    """
    e = FakeEngine()
    e._spawn()
    try:
        assert wait_for(lambda: e.state.checks >= 2), "루프가 돌지 않는다"
        assert e._thread is not None and e._thread.is_alive()
    finally:
        e._stop.set()


def test_spawning_twice_does_not_make_two_loops(fast_poll):
    """
    루프가 겹치면 같은 신호로 두 번 주문할 수 있다. MT5 는 스레드 안전하지도
    않다.
    """
    e = FakeEngine()
    e._spawn()
    first = e._thread
    e._spawn()
    try:
        assert e._thread is first
    finally:
        e._stop.set()


# ─── 새 봉이 떴을 때만 ───────────────────────────────────────────

def test_the_first_bar_is_not_traded(fast_poll):
    """
    **시작 시점의 마지막 봉은 이미 지나간 봉이다.** 그걸로 진입하면 백테스트가
    가정한 "봉 마감 직후 체결"보다 한참 늦은 가격에 들어간다.
    """
    e = FakeEngine()
    e._spawn()
    try:
        assert wait_for(lambda: e.state.checks >= 3)
        assert e.ran == [], "시작 직후 첫 봉으로 진입했다"
        assert e.state.last_bar == e.bar      # 기준으로 삼기는 한다
        assert e.state.cycles == 0
    finally:
        e._stop.set()


def test_a_new_bar_triggers_exactly_one_cycle(fast_poll):
    """같은 봉으로 두 번 돌면 같은 신호로 두 번 주문한다."""
    e = FakeEngine()
    e._spawn()
    try:
        assert wait_for(lambda: e.state.last_bar == e.bar)
        e.bar = "2026-09-07T06:00:00+00:00"          # 새 봉 마감
        assert wait_for(lambda: e.state.cycles >= 1)
        assert wait_for(lambda: e.state.checks >= 6)  # 더 돌려 본다
        assert e.state.cycles == 1, "같은 봉으로 여러 번 돌았다"
        assert e.ran == ["2026-09-07T06:00:00+00:00"]
    finally:
        e._stop.set()


def test_several_new_bars_each_get_one_cycle(fast_poll):
    e = FakeEngine()
    e._spawn()
    try:
        assert wait_for(lambda: e.state.last_bar == e.bar)
        for h in ("06", "07", "08"):
            e.bar = f"2026-09-07T{h}:00:00+00:00"
            assert wait_for(lambda h=h: e.state.last_bar.startswith(
                f"2026-09-07T{h}"))
        assert wait_for(lambda: e.state.cycles == 3)
        assert len(e.ran) == 3
    finally:
        e._stop.set()


# ─── 주말 ────────────────────────────────────────────────────────

def test_a_closed_market_is_quiet(fast_poll):
    """
    주말에는 새 봉이 안 뜬다. 오류를 장부에 쌓으면 진짜 문제가 묻힌다.
    """
    e = FakeEngine()
    e._spawn()
    try:
        assert wait_for(lambda: e.state.checks >= 5)
        assert e.state.cycles == 0
        assert [r for r in e.journal.rows if r["kind"] == "error"] == []
    finally:
        e._stop.set()


def test_bars_that_cannot_be_read_do_not_crash_the_loop(fast_poll):
    """MT5 가 잠깐 끊겨 None 이 와도 루프는 살아 있어야 한다."""
    e = FakeEngine()
    e.bar = None
    e._spawn()
    try:
        assert wait_for(lambda: e.state.checks >= 4)
        assert e._thread.is_alive()
        assert e.state.cycles == 0
    finally:
        e._stop.set()


# ─── 한 바퀴가 터져도 죽지 않는다 ────────────────────────────────

def test_a_failed_cycle_does_not_kill_the_loop(fast_poll):
    """
    한 종목이 말썽인 것으로 매매가 통째로 멈추면 열린 포지션을 관리할 주체가
    사라진다. 손절은 브로커에 걸려 있어 지켜지지만 재평가가 멈춘다.
    """
    e = FakeEngine()
    e._spawn()
    try:
        assert wait_for(lambda: e.state.last_bar == e.bar)
        e.explode = True
        e.bar = "2026-09-07T06:00:00+00:00"
        assert wait_for(lambda: e.state.last_error is not None)
        assert "한 바퀴 실패" in e.state.last_error
        assert e._thread.is_alive(), "한 번 실패로 루프가 죽었다"

        # 다음 봉에서는 정상 동작해야 한다
        e.explode = False
        e.bar = "2026-09-07T07:00:00+00:00"
        assert wait_for(lambda: e.ran == ["2026-09-07T07:00:00+00:00"])
    finally:
        e._stop.set()


def test_a_failed_cycle_is_journalled(fast_poll):
    """조용히 실패하면 왜 매매가 없었는지 나중에 알 수 없다."""
    e = FakeEngine()
    e.explode = True
    e._spawn()
    try:
        assert wait_for(lambda: e.state.last_bar == e.bar)
        e.bar = "2026-09-07T06:00:00+00:00"
        assert wait_for(lambda: any(r["kind"] == "error"
                                    for r in e.journal.rows))
        row = next(r for r in e.journal.rows if r["kind"] == "error")
        assert row["where"] == "loop"
    finally:
        e._stop.set()


# ─── 정지 ────────────────────────────────────────────────────────

def test_stop_ends_the_thread(fast_poll):
    e = FakeEngine()
    e.state.running = True
    e._spawn()
    assert wait_for(lambda: e.state.checks >= 2)
    e._stop.set()
    assert wait_for(lambda: not e._thread.is_alive()), "스레드가 안 멈춘다"


def test_the_loop_stops_promptly_not_after_a_full_poll(monkeypatch):
    """
    `time.sleep(POLL)` 로 기다리면 정지 요청이 최대 한 주기 늦게 먹는다.
    1시간봉이면 30초지만, 킬 스위치가 30초 늦게 먹는 것은 좋지 않다.
    Event.wait 를 써야 즉시 깬다.
    """
    monkeypatch.setattr(eng, "POLL_SECONDS", 30.0)     # 긴 주기로 둔다
    e = FakeEngine()
    e._spawn()
    assert wait_for(lambda: e.state.checks >= 1)
    t0 = time.time()
    e._stop.set()
    assert wait_for(lambda: not e._thread.is_alive(), timeout=3.0)
    assert time.time() - t0 < 2.0, "정지가 한 주기를 기다렸다"
