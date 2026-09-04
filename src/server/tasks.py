"""
백그라운드 작업 실행기 — 웹 화면이 CLI 스크립트를 그대로 돌리게 한다.

**왜 스크립트를 subprocess 로 부르는가.** 데이터 수집·검증·성적표는 이미
`scripts/` 에 있고 테스트도 그쪽에 붙어 있다. 웹용으로 같은 로직을 다시 쓰면
두 벌이 되고 언젠가 갈라진다. 화면은 스크립트를 부르고 출력만 흘려보낸다 —
터미널에서 돌린 것과 **글자 하나까지 같은 결과**가 나온다.

한 번에 하나만 돌린다. 이 작업들은 CPU 와 디스크를 크게 쓰고, 두 개가 겹치면
둘 다 느려지면서 사용자는 무엇이 멈춘 건지 알 수 없게 된다.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MAX_LINES = 4000          # 링 버퍼. 수집 로그가 수만 줄이라 전부 들고 있으면 샌다

# 터미널 색상 코드. pytest 는 파이프로 보내도 색을 넣는 경우가 있고, 그대로
# 브라우저에 넘기면 깨진 글자로 보인다.
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


@dataclass
class Task:
    name: str
    label: str
    argv: list[str]
    started: str
    lines: deque = field(default_factory=lambda: deque(maxlen=MAX_LINES))
    returncode: int | None = None
    error: str | None = None
    _proc: subprocess.Popen | None = None

    @property
    def running(self) -> bool:
        return self.returncode is None and self.error is None

    def as_dict(self, since: int = 0) -> dict:
        lines = list(self.lines)
        return {
            "name": self.name, "label": self.label,
            "started": self.started, "running": self.running,
            "returncode": self.returncode, "error": self.error,
            "total_lines": len(lines),
            "lines": lines[since:],
        }


class TaskRunner:
    """한 번에 하나. 실행 중이면 새 요청을 거절한다."""

    def __init__(self) -> None:
        self._current: Task | None = None
        self._history: dict[str, Task] = {}
        self._lock = threading.Lock()

    @property
    def current(self) -> Task | None:
        return self._current

    def get(self, name: str) -> Task | None:
        if self._current and self._current.name == name:
            return self._current
        return self._history.get(name)

    def start(self, name: str, label: str, argv: list[str]) -> tuple[bool, str]:
        with self._lock:
            if self._current and self._current.running:
                return False, f"'{self._current.label}' 이(가) 실행 중입니다. 끝나면 다시 시도하세요."
            task = Task(name=name, label=label, argv=list(argv),
                        started=datetime.now(timezone.utc).isoformat())
            self._current = task
            self._history[name] = task

        threading.Thread(target=self._run, args=(task,), daemon=True).start()
        return True, "시작했습니다."

    def stop(self) -> bool:
        t = self._current
        if t and t.running and t._proc:
            t._proc.terminate()
            t.lines.append("[중단 요청]")
            return True
        return False

    def _run(self, task: Task) -> None:
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
               "PYTHONUNBUFFERED": "1",
               # 색상 코드를 애초에 만들지 않게 한다. 그래도 새는 것은 ANSI 로 지운다.
               "NO_COLOR": "1", "TERM": "dumb", "PY_COLORS": "0"}
        try:
            proc = subprocess.Popen(
                [sys.executable, *task.argv], cwd=ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                # 콘솔 창이 새로 뜨지 않게 한다 — 웹에서 돌리는데 창이 튀어나오면 놀란다
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as exc:
            task.error = f"{type(exc).__name__}: {exc}"
            return

        task._proc = proc
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                task.lines.append(ANSI.sub("", line).rstrip("\n"))
            proc.wait()
            task.returncode = proc.returncode
        except Exception as exc:
            task.error = f"{type(exc).__name__}: {exc}"
        finally:
            task._proc = None


# ─── 화면이 부를 수 있는 작업 목록 ────────────────────────────────
#
# 여기 없는 명령은 실행되지 않는다. 웹에서 임의의 인자를 받아 실행하면
# 그건 원격 코드 실행이 된다 — 로컬 전용이어도 열어 둘 이유가 없다.

def catalog(universe_path: str = "runs/universe_final.json") -> dict[str, dict]:
    return {
        "signals": {
            "label": "실시간 신호",
            "desc": "지금 진입·청산 신호가 있는지 조회합니다 (주문 안 함)",
            "argv": ["scripts/signals.py", "--json", "runs/signals.json"],
            "minutes": "1~3분",
        },
        "fetch_small": {
            "label": "데이터 받기 · 기본 8종목",
            "desc": "1시간봉 + 1분봉 + 파생. 상황판과 검증에 필요합니다",
            "argv": ["scripts/fetch_data.py", "--intervals", "1h", "1m"],
            "minutes": "약 20분 · 1GB",
        },
        "fetch_deriv": {
            "label": "파생 데이터 받기",
            "desc": "미결제약정·펀딩·롱숏비율. 기본 전략이 이걸 씁니다",
            "argv": ["scripts/fetch_derivatives.py"],
            "minutes": "약 10분",
        },
        "fetch_full": {
            "label": "데이터 받기 · 129종목 전체",
            "desc": "검증을 그대로 재현하려면 필요합니다. 용량을 먼저 확인하세요",
            "argv": ["scripts/fetch_universe.py", "--workers", "5"],
            "minutes": "약 3시간 · 16GB",
            "heavy": True,
        },
        "validate": {
            "label": "백테스트 검증",
            "desc": "5항목 판정을 직접 재현합니다",
            "argv": ["scripts/validate.py", "--derivatives", "--calibrate-strength",
                     "--quick"],
            "minutes": "몇 분 ~ 수십 분",
        },
        "scoreboard": {
            "label": "지표 성적표",
            "desc": "어떤 보조지표가 값을 하는지 + 그 순위를 믿을 수 있는지",
            "argv": ["scripts/scoreboard.py", "--derivatives"],
            "minutes": "약 10분",
        },
        "selftest": {
            "label": "자체 점검",
            "desc": "테스트 전체를 실행합니다",
            "argv": ["-m", "pytest", "-q"],
            "minutes": "약 1분",
        },
    }


runner = TaskRunner()
