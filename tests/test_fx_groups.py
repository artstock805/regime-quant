"""
투표 군 — 기본 군과 단기 군.

1시간봉 미만의 유튜브 전략은 기본 군에 넣을 수 없다(1시간봉으로는 15분봉 조건을
계산할 수 없다). 그래서 따로 투표하고 따로 진입·청산하는 단기 군을 둔다.
여기서 지키려는 것은 **두 군이 한 계좌에서 서로를 건드리지 않는다**는 것이다.

    · 포지션의 주인은 주문 태그에서 읽는다. 재시작해도 남의 군 것을 내 것으로 세지 않는다
    · 한 군의 청산 신호가 다른 군의 포지션을 닫지 않는다
    · 한 군이 들고 있어도 다른 군은 같은 종목에 들어갈 수 있다
    · 동시 보유 한도는 군마다 따로 센다
    · 킬 스위치·일일 손실은 계좌의 것이라 두 군이 함께 쓴다
    · 단기 군에는 1시간봉으로 검증한 기존 전략이 들어가지 않는다
    · 1분봉 전략은 어느 군에도 들어가지 않는다

MT5 와 broker 는 전부 가짜다. 테스트가 돈을 움직이면 안 된다.
"""

from __future__ import annotations

import pytest

from src.config import Config, load_config
from src.execution.journal import Journal
from src.fx import broker
from src.fx import engine as eng
from src.fx.broker import Fill
from src.youtube import short_interval, short_strategies, store
from src.youtube.strategy import strategy_group
from test_fx_engine import DEMO, INFO, PAIR, Signal, fake_bars
from test_youtube import URL, raw_spec

BAR = "2025-06-01T12:00:00+00:00"


# ─── 태그에서 주인을 읽는다 ──────────────────────────────────────

def test_the_tag_carries_the_group():
    main = broker.tag("EURUSD", BAR)
    short = broker.tag("EURUSD", BAR, broker.SHORT_CODE)
    assert main == "rqEURUSD2506011200"                 # 기존 형식은 그대로다
    assert short == "rqs.EURUSD2506011200" and len(short) <= 31
    assert broker.group_of({"comment": main}) == "main"
    assert broker.group_of({"comment": short}) == "short"


@pytest.mark.parametrize("pos", [
    {}, {"comment": ""}, {"comment": "[sl 1.0950]"},     # 표식이 없거나 브로커가 바꿨다
    {"comment": "rqSGDJPY2506011200"},                   # S 로 시작하는 종목 — 대문자다
])
def test_a_position_without_a_marker_belongs_to_the_main_group(pos):
    """주인 없는 포지션으로 두면 아무도 청산 판단을 하지 않는다."""
    assert broker.group_of(pos) == "main"


# ─── 어느 전략이 어느 군인가 ─────────────────────────────────────

@pytest.mark.parametrize("timeframe, group", [
    ("1m", None),                                   # MT5 가 6개월치 1분봉을 주지 않는다
    ("5m", "short"), ("15m", "short"), ("30m", "short"),
    ("1h", "main"), ("4h", "main"), ("1d", "main"), (None, "main"),
])
def test_strategies_shorter_than_the_engine_bar_go_to_the_short_group(timeframe, group):
    assert strategy_group(timeframe, "1h") == group


def test_the_split_follows_the_engine_bar_not_a_fixed_hour():
    assert strategy_group("1h", "4h") == "short"
    assert strategy_group("5m", "5m") == "main"


def base_raw(tmp_path) -> dict:
    raw = dict(load_config(use_local=False).raw)
    raw["youtube"] = {"dir": str(tmp_path / "yt")}
    raw["fx"] = {**raw["fx"], "interval": "1h",
                 "watchlist_file": str(tmp_path / "wl.json")}
    return raw


def enable(cfg, name: str, timeframe: str | None) -> None:
    store.save(raw_spec(strategy_name=name, timeframe=timeframe), URL, cfg=cfg)
    store.set_enabled(name, True, cfg)


def test_the_short_group_runs_on_its_shortest_enabled_timeframe(tmp_path):
    cfg = Config(base_raw(tmp_path))
    assert short_interval(cfg) is None and short_strategies(cfg) == []

    enable(cfg, "half_hour", "30m")
    enable(cfg, "quarter", "15m")
    enable(cfg, "one_minute", "1m")        # 어느 군에도 안 들어간다
    enable(cfg, "hourly", "1h")            # 기본 군
    assert short_interval(cfg) == "15m"
    assert {s.name for s in short_strategies(cfg)} == {"yt_half_hour", "yt_quarter"}

    enable(cfg, "five", "5m")
    assert short_interval(cfg) == "5m"
    store.set_enabled("five", False, cfg)
    assert short_interval(cfg) == "15m"    # 끄면 다시 올라간다


# ─── 두 군이 도는 계좌 ───────────────────────────────────────────

class Book:
    """가짜 브로커의 포지션 장부. 두 엔진이 같은 것을 본다 — 계좌는 하나다."""

    def __init__(self) -> None:
        self.live: dict[str, dict] = {}
        self.sent: list[str] = []
        self.closed: list[int] = []
        self._next = 100

    def open(self, symbol, side, lots, sl, tp=None, *, tag_value="",
             allow_live=False, **kw):
        self._next += 1
        self.live[str(self._next)] = {
            "ticket": self._next, "symbol": symbol, "side": side, "volume": lots,
            "comment": tag_value, "magic": broker.MAGIC}
        self.sent.append(tag_value)
        return Fill(ticket=self._next, symbol=symbol, side=side, volume=lots,
                    price=1.1002, stop_loss=sl, take_profit=tp, tag=tag_value,
                    retcode=10009, comment="Done")

    def close(self, ticket, *, allow_live=False, reason="", **kw):
        self.live.pop(str(ticket), None)
        self.closed.append(int(ticket))
        return {"ticket": ticket, "closed": True, "symbol": "EURUSD",
                "side": "LONG", "volume": 0.1, "price": 1.1, "profit": 0.0}

    def reconcile(self, known):
        return {"open": dict(self.live),
                "closed_elsewhere": [k for k in known if k not in self.live],
                "unknown_to_us": [], "not_ours": [], "at": "now"}

    def groups(self) -> list[str]:
        return sorted(broker.group_of(p) for p in self.live.values())


@pytest.fixture
def desk(tmp_path, monkeypatch):
    """기본 군과 단기 군 엔진, 그리고 둘이 함께 쓰는 가짜 계좌."""
    book = Book()
    monkeypatch.setattr(eng, "account", lambda: DEMO)
    monkeypatch.setattr(eng, "symbol_info", lambda name: INFO)
    monkeypatch.setattr(eng, "bars", lambda *a, **k: fake_bars())
    monkeypatch.setattr(eng, "positions", lambda: {})
    monkeypatch.setattr(eng.broker, "open_position", book.open)
    monkeypatch.setattr(eng.broker, "close_position", book.close)
    monkeypatch.setattr(eng.broker, "reconcile", book.reconcile)
    monkeypatch.setattr(eng.broker, "lots_for_risk", lambda *a, **k: (0.10, ""))

    cfg = Config(base_raw(tmp_path))
    enable(cfg, "quarter", "15m")
    main = eng.FxTrader(cfg=cfg, journal=Journal(tmp_path / "j.jsonl"))
    short = eng.FxTrader(cfg=cfg, journal=main.journal, guard=main.guard,
                         group=eng.SHORT)
    for t in (main, short):
        monkeypatch.setattr(t, "watchlist", lambda: [PAIR])
        t.state.running = True

    def drive(trader, sig):
        monkeypatch.setattr(eng, "signal_for", lambda *a, **k: sig)
        return trader.run_once()

    return main, short, book, drive


def test_the_short_group_votes_only_with_short_youtube_strategies(desk):
    main, short, _, _ = desk
    assert [s.name for s in short.strategies()] == ["yt_quarter"]
    assert "yt_quarter" not in {s.name for s in main.strategies()}
    assert len(main.strategies()) > 1, "기본 군의 기존 전략이 사라지면 안 된다"
    assert short.interval == "15m" and main.interval == "1h"


def test_both_groups_can_hold_the_same_pair(desk):
    main, short, book, drive = desk
    drive(main, Signal("롱진입", 1))
    drive(short, Signal("롱진입", 1))
    assert book.sent == ["rqEURUSD2506011200", "rqs.EURUSD2506011200"]
    assert book.groups() == ["main", "short"]


def test_each_group_counts_only_its_own_positions(desk):
    """재시작 직후와 같다 — 엔진은 기억이 없고 브로커의 comment 만 본다."""
    main, short, book, drive = desk
    drive(main, Signal("롱진입", 1))
    drive(short, Signal("숏진입", -1, stop=1.1050, target=1.0900))
    main._known.clear()
    short._known.clear()

    drive(main, Signal("유지", 1))
    drive(short, Signal("유지", -1, stop=1.1050, target=1.0900))
    assert [broker.group_of(p) for p in main._known.values()] == ["main"]
    assert [broker.group_of(p) for p in short._known.values()] == ["short"]
    assert len(book.sent) == 2, "이미 들고 있는데 또 들어가면 안 된다"


def test_an_exit_in_one_group_leaves_the_other_alone(desk):
    main, short, book, drive = desk
    drive(main, Signal("롱진입", 1))
    drive(short, Signal("롱진입", 1))

    drive(short, Signal("청산", 0, stop=None))
    assert book.groups() == ["main"], "단기 군의 청산이 기본 군 포지션을 닫았다"

    drive(main, Signal("청산", 0, stop=None))
    assert book.live == {}


def test_an_opposite_signal_flips_only_its_own_group(desk):
    """단기 군이 숏으로 돌아서도 기본 군의 롱은 그대로다 (헤지 계좌)."""
    main, short, book, drive = desk
    drive(main, Signal("롱진입", 1))
    drive(short, Signal("롱진입", 1))
    drive(short, Signal("숏진입", -1, stop=1.1050, target=1.0900,
                        bar_time="2025-06-01T12:15:00+00:00"))
    sides = sorted((broker.group_of(p), p["side"]) for p in book.live.values())
    assert sides == [("main", "LONG"), ("short", "SHORT")]


def test_the_position_limit_is_counted_per_group(desk):
    main, short, book, drive = desk
    limit = main.guard.limits.max_positions
    for i in range(limit):                       # 기본 군이 한도를 다 채웠다
        book.open(f"PAIR{i}", "LONG", 0.1, 1.0, tag_value=f"rqPAIR{i}0000000000")
    drive(main, Signal("롱진입", 1))
    assert len(book.sent) == limit, "기본 군은 한도에 걸려야 한다"

    drive(short, Signal("롱진입", 1))
    assert book.sent[-1].startswith("rqs."), "단기 군은 자기 한도를 따로 센다"


def test_the_kill_switch_stops_both_groups(desk):
    """가드는 계좌의 것이다. 한 군에서 멈추라고 했는데 다른 군이 진입하면 킬 스위치가 아니다."""
    main, short, book, drive = desk
    assert short.guard is main.guard
    main.kill("테스트")
    drive(short, Signal("롱진입", 1))
    assert book.sent == []


def test_the_journal_says_which_group_acted(desk, tmp_path):
    main, short, _, drive = desk
    drive(main, Signal("롱진입", 1))
    drive(short, Signal("롱진입", 1))
    orders = [r for r in main.journal.tail(50) if r["kind"] == "order"]
    assert sorted(r.get("group", "main") for r in orders) == ["main", "short"]


def test_a_short_group_with_nothing_enabled_does_nothing(desk, monkeypatch):
    main, short, book, drive = desk
    store.set_enabled("quarter", False, short.cfg)
    monkeypatch.setattr(eng, "bars", lambda *a, **k: pytest.fail("돌 것이 없는데 봉을 받았다"))
    monkeypatch.setattr(eng, "signal_for", lambda *a, **k: pytest.fail("신호를 계산했다"))

    assert short.interval is None and short._newest_bar() is None
    status = short.run_once()
    assert status["scanned"] == 0 and status["interval"] is None
    assert book.sent == []


def test_the_status_names_the_group(desk):
    main, short, _, _ = desk
    assert main.status()["group"] == "main"
    assert short.status()["group"] == "short" and short.status()["interval"] == "15m"


# ─── API ─────────────────────────────────────────────────────────

@pytest.fixture
def client(desk, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.server import fx_api

    main, short, book, _ = desk
    monkeypatch.setattr(eng, "_traders", {"main": main, "short": short})
    monkeypatch.setattr(main, "_spawn", lambda: None)      # 루프 스레드는 띄우지 않는다
    monkeypatch.setattr(short, "_spawn", lambda: None)
    for t in (main, short):
        t.state.running = False
    app = FastAPI()
    app.include_router(fx_api.router)
    return TestClient(app), main, short


def test_each_group_is_started_and_stopped_on_its_own(client):
    c, main, short = client
    assert c.get("/api/fx/engine").json()["group"] == "main"
    assert c.get("/api/fx/engine?group=short").json()["group"] == "short"

    assert c.post("/api/fx/engine/start?group=short").json()["running"] is True
    assert short.state.running and not main.state.running

    c.post("/api/fx/engine/start")
    assert c.post("/api/fx/engine/stop?group=short").json()["running"] is False
    assert main.state.running and not short.state.running


def test_the_kill_endpoint_stops_every_group(client):
    c, main, short = client
    c.post("/api/fx/engine/start")
    c.post("/api/fx/engine/start?group=short")
    c.post("/api/fx/engine/kill")
    assert not main.state.running and not short.state.running
    assert main.guard.status(0.0)["kill_switch"] is True


def test_an_unknown_group_is_a_404(client):
    c, _, _ = client
    assert c.get("/api/fx/engine?group=nope").status_code == 404
    assert c.post("/api/fx/engine/start?group=nope").status_code == 404


def test_the_trader_registry_shares_the_guard_and_journal(monkeypatch, tmp_path):
    monkeypatch.setattr(eng, "_traders", {})
    monkeypatch.setattr(eng, "Journal", lambda *a, **k: Journal(tmp_path / "j.jsonl"))
    main, short = eng.trader(), eng.trader("short")
    assert main is eng.trader("main") and short is eng.trader("short")
    assert short.guard is main.guard and short.group is eng.SHORT
    assert {t.group.name for t in eng.traders()} == {"main", "short"}
    with pytest.raises(KeyError):
        eng.trader("nope")
