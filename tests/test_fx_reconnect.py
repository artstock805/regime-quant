"""
MT5 연결이 죽으면 다시 붙는다.

**실제로 있었던 일이다.** PC 가 절전에 들어갔다 나오자 터미널과의 연결이 죽었다.
죽은 연결은 예외를 던지지 않고 모든 호출에 None 을 돌려준다. `initialise()` 는
"한 번 붙었다"는 플래그만 보고 그대로 돌아갔고, 엔진은 종목 목록(None)을 돌다가
`TypeError: 'NoneType' object is not iterable` 을 30초마다 찍으며 며칠을 멈춰 있었다.
앱을 다시 켜야만 풀렸다.

지키려는 것:

    · 붙어 있다고 믿고 있어도 주기적으로 터미널이 응답하는지 본다
    · 응답이 없으면 닫고 다시 붙는다 — 사람이 앱을 다시 켜지 않아도
    · 연결이 수상하면(계정·종목이 None) 주기를 기다리지 않고 바로 다시 본다
    · 죽은 연결은 TypeError 가 아니라 MT5Error 로 말한다
    · 멀쩡한 연결을 매번 다시 붙이지 않는다

터미널은 가짜다. 테스트가 실제 MT5 에 닿으면 안 된다.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from src.fx import mt5 as m


class Terminal:
    """가짜 MT5. alive 가 False 면 죽은 연결처럼 모든 조회에 None 을 준다."""

    def __init__(self) -> None:
        self.alive = False          # 지금 이 연결이 살아 있는가
        self.reachable = True       # 터미널이 떠 있어 다시 붙을 수 있는가
        self.inits = self.shutdowns = self.health_checks = 0

    def initialize(self) -> bool:
        self.inits += 1
        self.alive = self.reachable
        return self.reachable

    def shutdown(self) -> None:
        self.shutdowns += 1
        self.alive = False

    def last_error(self):
        return -10004, "No IPC connection"

    def terminal_info(self):
        self.health_checks += 1
        return SimpleNamespace(trade_allowed=True) if self.alive else None

    def account_info(self):
        if not self.alive:
            return None
        return SimpleNamespace(login=1, server="Demo", currency="USD", balance=10_000.0,
                               equity=10_000.0, margin_free=9_000.0, trade_mode=0,
                               leverage=100)

    def symbols_get(self):
        if not self.alive:
            return None
        return (SimpleNamespace(name="EURUSD", path="Forex\\Majors\\EURUSD"),)


@pytest.fixture
def desk(monkeypatch):
    terminal, clock = Terminal(), [1000.0]
    monkeypatch.setattr(m, "_mt5", lambda: terminal)
    monkeypatch.setattr(m, "_initialised", False)
    monkeypatch.setattr(m, "_checked_at", float("-inf"))
    monkeypatch.setattr(m, "time", SimpleNamespace(monotonic=lambda: clock[0],
                                                   time=time.time))

    def later(seconds: float) -> None:
        clock[0] += seconds

    return terminal, later


def sleep_and_wake(terminal: Terminal) -> None:
    """절전에서 깨어났다. 터미널은 떠 있지만 우리가 쥔 연결은 죽었다."""
    terminal.alive = False


def test_a_dead_connection_is_reopened_without_restarting_the_app(desk):
    terminal, later = desk
    assert m.account().login == 1
    sleep_and_wake(terminal)
    later(m._HEALTH_EVERY + 1)

    assert m.account().login == 1, "다시 붙어서 계정을 읽어야 한다"
    assert terminal.inits == 2 and terminal.shutdowns == 1


def test_the_symbol_list_says_disconnected_instead_of_crashing(desk):
    """엔진 루프가 실제로 죽던 자리다. None 을 돌다가 TypeError 가 났다."""
    terminal, _ = desk
    assert m.available_symbols() == ["EURUSD"]
    sleep_and_wake(terminal)

    with pytest.raises(m.MT5Error, match="종목 목록"):
        m.available_symbols()


def test_a_suspicious_answer_is_rechecked_on_the_very_next_call(desk):
    """확인 주기가 안 지났어도, None 을 받았으면 다음 호출이 바로 다시 붙는다."""
    terminal, _ = desk
    m.available_symbols()
    sleep_and_wake(terminal)
    with pytest.raises(m.MT5Error):
        m.available_symbols()

    assert m.available_symbols() == ["EURUSD"]
    assert terminal.inits == 2


def test_a_failed_account_read_also_triggers_the_recheck(desk):
    terminal, _ = desk
    m.account()
    sleep_and_wake(terminal)
    with pytest.raises(m.MT5Error, match="계정 정보"):
        m.account()
    assert m.account().login == 1


def test_it_keeps_trying_until_the_terminal_is_back(desk):
    """터미널 자체가 꺼져 있으면 붙지 못한다. 켜지면 그때 붙는다."""
    terminal, later = desk
    m.account()
    sleep_and_wake(terminal)
    terminal.reachable = False
    later(m._HEALTH_EVERY + 1)
    for _ in range(3):
        with pytest.raises(m.MT5Error, match="붙지 못했습니다"):
            m.account()

    terminal.reachable = True
    assert m.account().login == 1


def test_a_healthy_connection_is_not_reopened(desk):
    terminal, later = desk
    for _ in range(50):
        m.available_symbols()       # (account 는 자기 일로 terminal_info 를 따로 부른다)
    assert terminal.inits == 1 and terminal.health_checks == 0, \
        "주기 안에서는 확인조차 하지 않는다 — 봉 수천 개를 받는 바퀴가 느려진다"

    later(m._HEALTH_EVERY + 1)
    for _ in range(50):
        m.available_symbols()
    assert terminal.inits == 1 and terminal.health_checks == 1, "주기마다 한 번만 확인한다"
    assert terminal.shutdowns == 0
