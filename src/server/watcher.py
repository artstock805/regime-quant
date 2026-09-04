"""
자동 감시 — 켜 두면 주기마다 신호를 다시 조회한다. **주문은 여전히 내지 않는다.**

이 파일이 별도로 있는 이유가 두 가지다.

**1) 슬롯을 따로 쓴다.** `TaskRunner` 는 한 번에 하나만 돌린다 — 무거운 작업이
겹치면 둘 다 느려지고 사용자는 무엇이 멈춘 건지 알 수 없기 때문이다. 그런데 감시는
켜 두면 계속 도는 것이라, 그 슬롯을 차지하면 다른 버튼이 전부 막힌다.
그래서 감시는 자기 슬롯에서 돌되, **다른 작업이 실행 중이면 그 주기를 건너뛴다.**

**2) 같은 스크립트를 부른다.** 웹용으로 신호 계산을 다시 쓰지 않는다.
`scripts/signals.py` 를 그대로 subprocess 로 돌리고 결과 JSON 을 읽는다.
버튼으로 한 번 돌린 것과 감시가 돌린 것이 **글자 하나까지 같은 경로**여야
화면의 숫자를 믿을 수 있다.

감시가 하는 일은 '알려 주는 것'까지다. 그 신호로 돈을 걸지는 사람이 정한다 —
검증 1/5, Deflated Sharpe 0.865(기준 0.95) 상태다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SIGNALS_JSON = ROOT / "runs" / "signals.json"

# 사람이 고를 수 있는 주기. 1분보다 짧게 두지 않는다 — 1시간봉 전략이라
# 그보다 자주 물어봐야 할 이유가 없고, 바이낸스 아카이브만 두드리게 된다.
INTERVALS = {60: "1분", 300: "5분", 900: "15분", 1800: "30분", 3600: "1시간"}
DEFAULT_INTERVAL = 300
MAX_ALERTS = 60
ACTIONABLE = {"롱진입", "숏진입", "청산"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SignalWatcher:
    """주기마다 신호를 다시 조회하고, **새로 생긴** 진입·청산만 알린다."""

    def __init__(self, runner=None) -> None:
        self._runner = runner                  # 무거운 작업과 겹치지 않게 물어본다
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None

        self.interval: int = DEFAULT_INTERVAL
        self.started_at: str | None = None
        self.last_run: str | None = None
        self.next_run: str | None = None
        self.cycles: int = 0
        self.skipped: int = 0
        self.last_error: str | None = None
        self.last_count: int = 0
        self.alerts: deque = deque(maxlen=MAX_ALERTS)
        # 직전 사이클의 (종목 → 동작). 이게 있어야 '새로 생긴' 것을 가릴 수 있다.
        self._seen: dict[str, str] = {}

    # ─── 제어 ────────────────────────────────────────────────────

    @property
    def alive(self) -> bool:
        """스레드가 아직 살아 있는가. 조회 한 번이 1분 넘게 걸릴 수 있다."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def stopping(self) -> bool:
        """끄라고 했는데 아직 정리 중."""
        return self.alive and self._stop.is_set()

    @property
    def running(self) -> bool:
        """
        화면에 보여 줄 상태. **끄라고 한 순간 꺼진 것으로 본다.**
        스레드가 조회를 마칠 때까지 '실행 중'으로 두면, 껐는데 안 꺼진 것처럼
        보이고 다시 누르면 "이미 감시 중"이 나온다 — 실제로 그랬다.
        """
        return self.alive and not self._stop.is_set()

    def start(self, interval: int = DEFAULT_INTERVAL) -> tuple[bool, str]:
        with self._lock:
            if self.running:
                return False, "이미 감시 중입니다."
            if interval not in INTERVALS:
                return False, f"주기는 {', '.join(INTERVALS.values())} 중에서 고르세요."
            # 직전 스레드가 정리 중이면 잠깐 기다린다. 안 기다리면 스레드가 둘이 된다.
            if self.alive:
                self._thread.join(timeout=5)
                if self.alive:
                    return False, "직전 조회를 정리하는 중입니다. 잠시 뒤 다시 시도하세요."
            self.interval = interval
            self._stop.clear()
            self.started_at = _now().isoformat()
            self.last_error = None
            self._seen = {}
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return True, f"{INTERVALS[interval]}마다 조회합니다."

    def stop(self) -> tuple[bool, str]:
        if not self.alive:
            return False, "감시 중이 아닙니다."
        self._stop.set()
        self.next_run = None
        # 조회 중이면 그 서브프로세스도 끊는다. 안 그러면 "멈춤"을 눌러도
        # 최대 1분 넘게 계속 도는 것처럼 보인다.
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        return True, "감시를 멈췄습니다."

    def status(self) -> dict:
        return {
            "running": self.running,
            "stopping": self.stopping,
            "interval": self.interval,
            "interval_label": INTERVALS.get(self.interval, f"{self.interval}초"),
            "choices": [{"seconds": k, "label": v} for k, v in INTERVALS.items()],
            "started_at": self.started_at,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "cycles": self.cycles,
            "skipped": self.skipped,
            "last_error": self.last_error,
            "last_count": self.last_count,
            "alerts": list(self.alerts)[::-1],      # 최신이 위로
            "order_execution": False,
        }

    def clear_alerts(self) -> None:
        self.alerts.clear()

    # ─── 본체 ────────────────────────────────────────────────────

    def _loop(self) -> None:
        # 켜자마자 한 번 돈다. "켰는데 5분간 아무 일도 없다"가 가장 흔한 오해다.
        while not self._stop.is_set():
            self._cycle()
            self.next_run = None if self._stop.is_set() else (
                _now().timestamp() + self.interval)
            if self._stop.wait(self.interval):
                break
        self.next_run = None

    def _cycle(self) -> None:
        # 무거운 작업이 돌고 있으면 이번 주기는 건너뛴다. 둘이 CPU 를 두고 싸우면
        # 둘 다 느려지고, 사용자는 무엇이 멈춘 건지 알 수 없게 된다.
        cur = getattr(self._runner, "current", None) if self._runner else None
        if cur is not None and getattr(cur, "running", False):
            self.skipped += 1
            return

        try:
            self._run_scan()
        except Exception as exc:                    # 한 주기 실패가 감시를 죽이면 안 된다
            self.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            self.cycles += 1
            self.last_run = _now().isoformat()

    def _run_scan(self) -> None:
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
               "PYTHONUNBUFFERED": "1", "NO_COLOR": "1"}
        proc = subprocess.Popen(
            [sys.executable, "scripts/signals.py", "--json", str(SIGNALS_JSON)],
            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._proc = proc
        try:
            out, _ = proc.communicate(timeout=max(600, self.interval * 3))
        except subprocess.TimeoutExpired:
            proc.kill()
            self.last_error = "조회가 너무 오래 걸려 중단했습니다"
            return
        finally:
            self._proc = None

        if self._stop.is_set():
            return                       # 사용자가 끈 것이지 실패가 아니다
        if proc.returncode != 0:
            tail = (out or "").strip().splitlines()[-3:]
            self.last_error = "조회 실패: " + " / ".join(tail) if tail else "조회 실패"
            return

        data = json.loads(SIGNALS_JSON.read_text(encoding="utf-8"))
        signals = data.get("signals", [])
        self.last_count = len(signals)
        self.last_error = None
        self._record_new(signals)

    def _record_new(self, signals: list[dict]) -> None:
        """
        **새로 생긴** 진입·청산만 알린다.

        매 주기 같은 '롱진입'을 다시 알리면 알림이 무의미해진다. 종목별 직전 동작을
        기억해 두고, 동작이 바뀐 것만 남긴다. 첫 사이클은 기준을 세우는 것이므로
        알리지 않는다 — 켜자마자 보유 중인 종목이 전부 '진입'으로 쏟아지면 안 된다.
        """
        first = not self._seen
        now = _now().isoformat()
        for s in signals:
            sym, act = s.get("symbol"), s.get("action")
            if not sym:
                continue
            prev = self._seen.get(sym)
            self._seen[sym] = act
            if first or act not in ACTIONABLE or act == prev:
                continue
            self.alerts.append({
                "at": now, "symbol": sym, "action": act,
                "score": s.get("score"), "close": s.get("close"),
                "regime": s.get("regime_korean"),
                "stop_price": s.get("stop_price"),
                "target_price": s.get("target_price"),
                "bar_time": s.get("bar_time"),
            })


watcher = SignalWatcher()
