"""
자동 감시 — 켜 둬도 안전한가.

감시는 "켜 두고 잊는" 기능이라 조용한 실패가 가장 위험하다. 세 가지를 고정한다.

**1) 다른 작업을 막지 않는다.** TaskRunner 는 한 번에 하나만 돌린다. 감시가 그
   슬롯을 차지하면 데이터 받기·검증 버튼이 전부 막힌다. 감시는 자기 슬롯에서 돌되,
   무거운 작업이 도는 동안에는 그 주기를 건너뛴다.

**2) 같은 신호를 반복해서 알리지 않는다.** 매 주기 같은 '롱진입'을 다시 알리면
   알림이 무의미해진다. 상태가 **바뀐** 것만 알린다.

**3) 켜자마자 쏟아내지 않는다.** 첫 사이클은 기준을 세우는 것이다. 이미 보유
   중이던 종목이 전부 '진입'으로 뜨면 사용자는 그게 지금 생긴 신호라고 읽는다.

그리고 어떤 경우에도 주문은 나가지 않는다 — 그 사실도 코드에서 고정한다.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from src.server import watcher as watcher_mod
from src.server.watcher import ACTIONABLE, INTERVALS, SignalWatcher


def _sig(symbol, action, score=0.9):
    return {"symbol": symbol, "action": action, "score": score, "close": 100.0,
            "regime_korean": "상승추세", "stop_price": 96.0, "target_price": 106.0,
            "bar_time": "2026-09-04T10:00:00+00:00"}


class _FakeTask:
    def __init__(self, running):
        self.running = running


class _FakeRunner:
    def __init__(self, running=False):
        self.current = _FakeTask(running) if running else None


# ─── 주문 경로가 없다 ─────────────────────────────────────────────

def test_watcher_has_no_order_execution():
    src = Path(watcher_mod.__file__).read_text(encoding="utf-8").lower()
    banned = ["create_order", "place_order", "new_order", "submit_order",
              "cancel_order", "api_secret", "api_key", "/fapi/v1/order"]
    hits = [b for b in banned if b in src]
    assert not hits, f"감시에 주문/인증 코드가 들어왔습니다: {hits}"


def test_status_declares_no_order_execution():
    assert SignalWatcher().status()["order_execution"] is False


# ─── 다른 작업을 막지 않는가 ──────────────────────────────────────

def test_cycle_is_skipped_while_a_heavy_task_runs(monkeypatch):
    """
    **핵심.** 감시와 무거운 작업이 겹치면 둘 다 느려지고 사용자는 무엇이 멈춘
    건지 알 수 없다. 무거운 작업이 도는 동안에는 조회하지 않는다.
    """
    w = SignalWatcher(runner=_FakeRunner(running=True))
    called = []
    monkeypatch.setattr(w, "_run_scan", lambda: called.append(1))
    w._cycle()
    assert called == [], "무거운 작업 중에 조회했다"
    # 건너뛴 주기는 '조회'가 아니다. 화면도 "조회 N회 · 건너뜀 M회"로 나눠 보여준다.
    assert w.skipped == 1 and w.cycles == 0


def test_cycle_runs_when_nothing_else_is_running(monkeypatch):
    w = SignalWatcher(runner=_FakeRunner(running=False))
    called = []
    monkeypatch.setattr(w, "_run_scan", lambda: called.append(1))
    w._cycle()
    assert called == [1] and w.skipped == 0


def test_a_failing_cycle_does_not_kill_the_watcher(monkeypatch):
    """한 주기 실패로 감시가 멈추면, 켜 둔 사람은 그 사실을 모른다."""
    w = SignalWatcher(runner=_FakeRunner())

    def boom():
        raise RuntimeError("네트워크 오류")

    monkeypatch.setattr(w, "_run_scan", boom)
    w._cycle()
    assert "네트워크 오류" in w.last_error
    assert w.cycles == 1
    # 다음 주기는 정상 동작해야 한다
    monkeypatch.setattr(w, "_run_scan", lambda: None)
    w._cycle()
    assert w.cycles == 2


# ─── 알림 규칙 ────────────────────────────────────────────────────

def test_first_cycle_sets_a_baseline_without_alerting():
    """
    켜자마자 보유 중이던 종목이 전부 '진입'으로 쏟아지면 사용자는 그게 지금
    생긴 신호라고 읽는다. 첫 사이클은 기준만 세운다.
    """
    w = SignalWatcher()
    w._record_new([_sig("BTCUSDT", "롱진입"), _sig("ETHUSDT", "숏진입")])
    assert list(w.alerts) == [], "첫 사이클에 알렸다"
    assert w._seen == {"BTCUSDT": "롱진입", "ETHUSDT": "숏진입"}


def test_only_changed_actions_are_alerted():
    w = SignalWatcher()
    w._record_new([_sig("BTCUSDT", "관망"), _sig("ETHUSDT", "관망")])   # 기준
    w._record_new([_sig("BTCUSDT", "롱진입"), _sig("ETHUSDT", "관망")])
    assert [a["symbol"] for a in w.alerts] == ["BTCUSDT"]

    # 같은 상태가 이어지면 다시 알리지 않는다
    w._record_new([_sig("BTCUSDT", "롱진입"), _sig("ETHUSDT", "관망")])
    assert len(w.alerts) == 1, "같은 신호를 반복해서 알렸다"

    # 청산으로 바뀌면 알린다
    w._record_new([_sig("BTCUSDT", "청산"), _sig("ETHUSDT", "관망")])
    assert [a["action"] for a in w.alerts] == ["롱진입", "청산"]


def test_hold_and_wait_are_never_alerted():
    w = SignalWatcher()
    w._record_new([_sig("BTCUSDT", "롱진입")])          # 기준
    w._record_new([_sig("BTCUSDT", "유지")])
    w._record_new([_sig("BTCUSDT", "관망")])
    assert list(w.alerts) == [], f"'유지/관망'을 알렸다: {list(w.alerts)}"
    assert ACTIONABLE == {"롱진입", "숏진입", "청산"}


def test_alerts_are_capped_so_memory_does_not_grow():
    w = SignalWatcher()
    w._record_new([_sig("A", "관망")])
    for i in range(200):
        w._record_new([_sig("A", "롱진입" if i % 2 else "청산")])
    assert len(w.alerts) == watcher_mod.MAX_ALERTS


# ─── 제어 ─────────────────────────────────────────────────────────

def test_rejects_an_interval_that_is_not_offered():
    ok, msg = SignalWatcher().start(7)
    assert not ok and "주기" in msg


def test_does_not_start_twice(monkeypatch):
    w = SignalWatcher(runner=_FakeRunner())
    monkeypatch.setattr(w, "_run_scan", lambda: time.sleep(0.05))
    assert w.start(60)[0] is True
    ok, msg = w.start(60)
    assert not ok and "이미" in msg
    w.stop()


def test_stop_is_prompt(monkeypatch):
    """
    Event.wait 를 써야 멈추라고 했을 때 바로 멈춘다. sleep 이면 다음 주기까지
    최대 1시간 기다린다.
    """
    w = SignalWatcher(runner=_FakeRunner())
    monkeypatch.setattr(w, "_run_scan", lambda: None)
    w.start(3600)
    time.sleep(0.2)
    t0 = time.time()
    w.stop()
    w._thread.join(timeout=3)
    assert not w.running, "멈추지 않았다"
    assert time.time() - t0 < 3, "멈추는 데 너무 오래 걸린다"


def test_status_is_json_serialisable():
    json.dumps(SignalWatcher().status(), ensure_ascii=False)


def test_intervals_are_at_least_one_minute():
    """1시간봉 전략이라 그보다 자주 물어볼 이유가 없다 — 아카이브만 두드리게 된다."""
    assert min(INTERVALS) >= 60


# ─── 조회 실패 처리 ───────────────────────────────────────────────

class _FakeProc:
    """subprocess.Popen 대역. 중단 가능해야 해서 Popen 을 쓴다 — run 이 아니다."""

    def __init__(self, returncode=0, out=""):
        self.returncode, self._out = returncode, out
        self.terminated = False

    def communicate(self, timeout=None):
        return (self._out, None)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True


def test_scan_failure_is_reported_not_swallowed(monkeypatch):
    proc = _FakeProc(returncode=1, out="line1\nBTCUSDT: 실패 ConnectionError\n")
    monkeypatch.setattr(watcher_mod.subprocess, "Popen", lambda *a, **k: proc)
    w = SignalWatcher(runner=_FakeRunner())
    w._run_scan()
    assert w.last_error and "실패" in w.last_error


def test_successful_scan_reads_the_json_and_clears_the_error(monkeypatch, tmp_path):
    payload = {"created": "2026-09-04T10:00:00+00:00", "interval": "1h",
               "signals": [_sig("BTCUSDT", "관망"), _sig("ETHUSDT", "롱진입")]}
    path = tmp_path / "signals.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(watcher_mod, "SIGNALS_JSON", path)
    monkeypatch.setattr(watcher_mod.subprocess, "Popen",
                        lambda *a, **k: _FakeProc(returncode=0))
    w = SignalWatcher(runner=_FakeRunner())
    w.last_error = "예전 오류"
    w._run_scan()
    assert w.last_count == 2 and w.last_error is None


def test_scan_never_calls_subprocess_run():
    """
    Popen 으로 바꾼 이유는 중단 가능해야 하기 때문이다. run 으로 되돌아가면
    '멈춤'을 눌러도 조회가 끝날 때까지 계속 돈다. 게다가 이 파일의 테스트가
    run 을 가로채고 있으면 **실제 조회가 돌아 버린다** — 실제로 그랬고,
    테스트 한 번에 2분이 걸렸다.
    """
    src = Path(watcher_mod.__file__).read_text(encoding="utf-8")
    assert "subprocess.run(" not in src, "subprocess.run 은 중단할 수 없다"
    assert "subprocess.Popen(" in src


# ─── 끄면 즉시 꺼진 것으로 보이는가 ───────────────────────────────

def test_stop_immediately_reports_off_even_while_a_scan_finishes(monkeypatch):
    """
    조회 한 번이 1분 넘게 걸린다. 그동안 '실행 중'으로 두면 껐는데 안 꺼진
    것처럼 보이고, 다시 누르면 "이미 감시 중"이 나온다 — 실제로 그랬다.
    """
    w = SignalWatcher(runner=_FakeRunner())
    slow = threading.Event()
    monkeypatch.setattr(w, "_run_scan", lambda: slow.wait(5))

    w.start(3600)
    time.sleep(0.2)
    assert w.running is True

    ok, _ = w.stop()
    assert ok
    assert w.running is False, "껐는데 아직 '실행 중'으로 보인다"
    assert w.stopping is True, "정리 중이라는 사실은 남아야 한다"
    assert w.alive is True

    slow.set()
    w._thread.join(timeout=5)
    assert not w.alive and not w.stopping


def test_restart_waits_for_the_previous_thread(monkeypatch):
    """정리 중에 다시 켜면 스레드가 둘이 되어 조회가 겹친다."""
    w = SignalWatcher(runner=_FakeRunner())
    slow = threading.Event()
    monkeypatch.setattr(w, "_run_scan", lambda: slow.wait(30))

    w.start(3600)
    time.sleep(0.2)
    w.stop()
    ok, msg = w.start(3600)         # 아직 정리 중
    assert not ok and "정리" in msg

    slow.set()
    w._thread.join(timeout=5)


def test_stop_during_a_scan_is_not_reported_as_a_failure(monkeypatch, tmp_path):
    """사용자가 끈 것을 '조회 실패'로 표시하면 오해한다."""
    class P:
        returncode = -15                       # terminate 된 프로세스
        def communicate(self, timeout=None):
            return ("중단됨", None)
        def poll(self):
            return None
        def terminate(self):
            pass

    monkeypatch.setattr(watcher_mod.subprocess, "Popen", lambda *a, **k: P())
    w = SignalWatcher(runner=_FakeRunner())
    w._stop.set()                              # 끈 상태
    w._run_scan()
    assert w.last_error is None, f"중단을 실패로 기록했다: {w.last_error}"
