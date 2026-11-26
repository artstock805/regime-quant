"""
엔진이 도는 동안 PC 가 절전에 들어가지 않게 한다 (src/fx/keepawake.py).

**실제로 있었던 일이다.** 엔진을 켜 두고 자리를 비웠더니 PC 가 절전에 들어가,
일주일 동안 엔진이 판단한 시간이 13시간쯤뿐이었다.

지키려는 것:

    · 루프가 시작할 때 절전 방지를 요청하고, 끝날 때 푼다
    · 요청과 해제는 **루프 스레드에서** 한다 — Windows 에서 이 요청은 부른 스레드에
      묶인다. 다른 스레드(웹 요청 스레드)에서 걸면 그 스레드가 끝날 때 조용히 풀린다
    · 루프가 터져서 끝나도 푼다
    · 요청이 거부되면 됐다고 말하지 않는다

Windows 호출은 가짜다. 테스트가 이 PC 의 전원 상태에 손대면 안 된다.
"""

from __future__ import annotations

import threading

import pytest

from src.fx import engine as eng
from src.fx import keepawake
from test_fx_scheduler import FakeEngine, wait_for

HOLD = 0x80000001          # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
RELEASE = 0x80000000       # ES_CONTINUOUS


@pytest.fixture
def calls(monkeypatch):
    """Windows 에 무엇을, 어느 스레드에서 요청했는지. 두 번째 값으로 거부를 흉내 낸다."""
    seen: list[tuple[int, str]] = []
    granted = [True]

    def fake(flags: int) -> bool:
        seen.append((flags, threading.current_thread().name))
        return granted[0]

    monkeypatch.setattr(keepawake, "_set", fake)
    monkeypatch.setattr(eng, "POLL_SECONDS", 0.05)
    return seen, granted


def test_the_loop_keeps_the_pc_awake_while_it_runs_and_lets_go_when_it_stops(calls):
    seen, _ = calls
    engine = FakeEngine()
    engine.state.running = True
    engine._spawn()
    assert wait_for(lambda: engine._awake)
    assert seen == [(HOLD, "fx-engine-main")], "루프 스레드에서 걸어야 한다"

    engine._stop.set()
    engine._thread.join(timeout=3)
    assert seen[-1] == (RELEASE, "fx-engine-main"), "건 스레드가 풀어야 한다"
    assert engine._awake is False


def test_the_hold_is_released_even_when_the_loop_dies(calls, monkeypatch):
    seen, _ = calls
    engine = FakeEngine()

    def boom():
        raise RuntimeError("루프가 죽었다")

    monkeypatch.setattr(engine, "_poll", boom)
    with pytest.raises(RuntimeError):
        engine._loop()
    assert [flags for flags, _ in seen] == [HOLD, RELEASE]
    assert engine._awake is False


def test_a_refused_request_is_not_reported_as_held(calls):
    """Windows 가 아니거나 요청이 거부됐다. 화면이 '켜짐'이라고 하면 안 된다."""
    seen, granted = calls
    granted[0] = False
    engine = FakeEngine()
    engine.state.running = True
    engine._spawn()
    assert wait_for(lambda: len(seen) >= 1)
    assert engine._awake is False
    engine._stop.set()
    engine._thread.join(timeout=3)


def test_hold_and_release_ask_for_the_right_thing(calls):
    seen, _ = calls
    assert keepawake.hold() and keepawake.release()
    assert [flags for flags, _ in seen] == [HOLD, RELEASE]
    # 화면을 켜 두라고는 하지 않는다 (ES_DISPLAY_REQUIRED = 0x2) — 모니터는 꺼져도 된다
    assert not HOLD & 0x00000002
