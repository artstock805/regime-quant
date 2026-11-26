"""
역행 대응 — 마팅게일 추가 진입과 급변동 시 양방 대응 (src/fx/defense.py).

지키려는 것:

    · 첫 진입의 수량은 그대로고 손절만 (추가 횟수 + 1) 단계 밖에 건다
    · 손절 거리 1배마다, 봉 마감에, 한 번에 한 단계씩, 직전의 2배로 추가한다
    · 추가는 최대 3회. 그 뒤는 브로커에 걸린 최종 손절이 맡는다
    · 추가할 때마다 묶음 전체의 목표가를 "처음 노렸던 수익"이 나는 가격으로 옮긴다
    · 급변동으로 밀렸으면 추가하지 않고 같은 수량의 반대 포지션으로 묶는다
    · 묶여 있는 동안은 추가하지 않는다. 묶음이 풀리면 다시 대응한다
    · 재시작해도 묶음을 잃지 않는다 — 역할은 주문 태그에 있다
    · 표식 없는 예전 포지션은 건드리지 않는다
    · 킬 스위치·정지·동시 보유 한도가 대응 주문에도 그대로 듣는다

MT5 와 broker 는 전부 가짜다. 테스트가 돈을 움직이면 안 된다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config, load_config
from src.execution.journal import Journal
from src.fx import broker, defense
from src.fx import engine as eng
from src.fx.broker import Fill
from test_fx_engine import DEMO, INFO, PAIR, Signal, fake_bars

STEP = 0.0050                      # Signal 기본값: 종가 1.1000, 손절 1.0950
BASE_TAG = "rqEURUSD2506011200p500m0"    # p500 = 단계 간격 500포인트 (0.0050)


class Book:
    """가짜 브로커. 포지션에 진입가·손절·목표가가 있어야 묶음을 읽을 수 있다."""

    def __init__(self) -> None:
        self.live: dict[str, dict] = {}
        self.price = 1.1000        # 지금 체결되는 가격
        self._next = 100

    def open(self, symbol, side, lots, sl, tp=None, *, tag_value="",
             allow_live=False, **kw):
        self._next += 1
        self.live[str(self._next)] = {
            "ticket": self._next, "symbol": symbol, "side": side, "volume": lots,
            "entry_price": self.price, "stop_loss": sl, "take_profit": tp,
            "comment": tag_value, "magic": broker.MAGIC}
        return Fill(ticket=self._next, symbol=symbol, side=side, volume=lots,
                    price=self.price, stop_loss=sl, take_profit=tp, tag=tag_value,
                    retcode=10009, comment="Done")

    def close(self, ticket, *, allow_live=False, reason="", **kw):
        self.live.pop(str(ticket), None)
        return {"ticket": ticket, "closed": True, "symbol": "EURUSD", "side": "LONG",
                "volume": 0.1, "price": self.price, "profit": 0.0,
                "reason": reason, "retcode": 10009}

    def move(self, ticket, stop_loss, take_profit=None):
        self.live[str(ticket)].update(stop_loss=stop_loss, take_profit=take_profit)
        return {"ticket": ticket, "stop_loss": stop_loss, "take_profit": take_profit}

    def reconcile(self, known):
        return {"open": {k: dict(v) for k, v in self.live.items()},
                "closed_elsewhere": [k for k in known if k not in self.live],
                "unknown_to_us": [], "not_ours": [], "at": "now"}

    def by_tag(self) -> dict[str, dict]:
        return {p["comment"][len(BASE_TAG) - 2:]: p for p in self.live.values()}


@pytest.fixture
def desk(tmp_path, monkeypatch):
    """최종 손절을 거는 설정."""
    return make_desk(tmp_path, monkeypatch, final_stop=True)


@pytest.fixture
def nostop(tmp_path, monkeypatch):
    """최종 손절을 꺼 둔 설정 — 손절 없이 진입한다."""
    return make_desk(tmp_path, monkeypatch, final_stop=False)


@pytest.fixture
def live_rules(tmp_path, monkeypatch):
    """지금 쓰는 설정 — 손절 없이, 2회 추가부터 본전 청산, 계좌 손실 한도 20%."""
    return make_desk(tmp_path, monkeypatch, final_stop=False,
                     breakeven_after=2, seed_stop=0.20)


def make_desk(tmp_path, monkeypatch, *, final_stop: bool, **more):
    book = Book()
    monkeypatch.setattr(eng, "account", lambda: DEMO)
    monkeypatch.setattr(eng, "symbol_info", lambda name: INFO)
    monkeypatch.setattr(eng, "bars", lambda *a, **k: fake_bars())
    monkeypatch.setattr(eng, "positions", lambda: {})
    monkeypatch.setattr(eng.broker, "open_position", book.open)
    monkeypatch.setattr(eng.broker, "close_position", book.close)
    monkeypatch.setattr(eng.broker, "move_stop", book.move)
    monkeypatch.setattr(eng.broker, "reconcile", book.reconcile)
    monkeypatch.setattr(eng.broker, "lots_for_risk", lambda *a, **k: (0.10, ""))

    raw = dict(load_config(use_local=False).raw)
    raw["youtube"] = {"dir": str(tmp_path / "yt")}
    raw["fx"] = {**raw["fx"], "interval": "1h",
                 "watchlist_file": str(tmp_path / "wl.json"),
                 "defense": {"enabled": True, "max_adds": 3, "multiplier": 2.0,
                             "shock_mult": 2.0, "final_stop": final_stop, **more}}
    trader = eng.FxTrader(cfg=Config(raw), journal=Journal(tmp_path / "j.jsonl"))
    monkeypatch.setattr(trader, "watchlist", lambda: [PAIR])
    trader.state.running = True

    hour = [12]

    def drive(price=None, sig=None, same_bar=False):
        """
        가격이 여기서 봉을 마감했다고 치고 한 바퀴 돌린다. 부를 때마다 다음 봉이다 —
        same_bar 면 방금 그 봉에서 한 번 더 돈다(화면의 "한 바퀴" 버튼).
        """
        if price is not None:
            book.price = price
        if not same_bar:
            hour[0] += 1
        sig = sig or Signal("유지", 1, close=book.price,
                            bar_time=f"2025-06-01T{hour[0]:02d}:00:00+00:00")
        monkeypatch.setattr(eng, "signal_for", lambda *a, **k: sig)
        return trader.run_once()

    return trader, book, drive


def enter(drive) -> None:
    drive(sig=Signal("롱진입", 1))


# ─── 첫 진입 ─────────────────────────────────────────────────────

def test_the_first_entry_keeps_its_size_and_parks_the_stop_four_steps_away(desk):
    _, book, drive = desk
    enter(drive)
    (pos,) = book.live.values()
    assert pos["comment"] == BASE_TAG and pos["volume"] == 0.10
    assert pos["stop_loss"] == pytest.approx(1.1000 - 4 * STEP)
    assert pos["take_profit"] == pytest.approx(1.1100)


# ─── 마팅게일 ────────────────────────────────────────────────────

def test_nothing_is_added_before_the_price_reaches_the_step(desk):
    _, book, drive = desk
    enter(drive)
    drive(1.0951)
    assert len(book.live) == 1


def test_one_step_against_adds_double_the_size(desk):
    _, book, drive = desk
    enter(drive)
    drive(1.0950)
    legs = book.by_tag()
    assert set(legs) == {"m0", "m1"}
    assert legs["m1"]["side"] == "LONG" and legs["m1"]["volume"] == pytest.approx(0.20)
    assert legs["m1"]["stop_loss"] == pytest.approx(1.0800), "추가분도 최종 손절을 함께 쓴다"


def test_adds_stop_after_three(desk):
    _, book, drive = desk
    enter(drive)
    for price in (1.0950, 1.0900, 1.0850, 1.0810, 1.0805):
        drive(price)
    legs = book.by_tag()
    assert set(legs) == {"m0", "m1", "m2", "m3"}
    assert [legs[k]["volume"] for k in ("m0", "m1", "m2", "m3")] == \
        pytest.approx([0.10, 0.20, 0.40, 0.80])


def test_a_bar_that_skips_two_steps_adds_only_one(desk):
    """봉 마감에 한 번 판단한다. 한 번에 4배를 싣지 않는다."""
    _, book, drive = desk
    enter(drive)
    drive(1.0890)
    assert set(book.by_tag()) == {"m0", "m1"}
    drive(1.0890, same_bar=True)
    assert set(book.by_tag()) == {"m0", "m1"}, "같은 봉에서 한 바퀴 더 돌려도 또 싣지 않는다"
    drive(1.0890)
    assert set(book.by_tag()) == {"m0", "m1", "m2"}, "다음 봉이 마감하면 다음 단계를 본다"


def test_a_missing_middle_add_does_not_stall_the_defense(desk):
    """1차가 사라지고 2차만 남았다. 다음은 3차다 — 이미 있는 2차 태그로 또 내지 않는다."""
    _, book, drive = desk
    enter(drive)
    drive(1.0950)
    drive(1.0900)
    del book.live[str(book.by_tag()["m1"]["ticket"])]
    drive(1.0850)
    legs = book.by_tag()
    assert set(legs) == {"m0", "m2", "m3"}
    assert legs["m3"]["volume"] == pytest.approx(0.80)


def test_every_add_pulls_the_whole_baskets_target_in(desk):
    """평균 단가에서, 처음 노렸던 수익 금액이 나는 가격으로."""
    trader, book, drive = desk
    enter(drive)
    drive(1.0950)
    reward = float(trader.cfg.require("risk.reward_ratio"))
    average = (0.10 * 1.1000 + 0.20 * 1.0950) / 0.30
    expected = average + reward * STEP * 0.10 / 0.30
    targets = [p["take_profit"] for p in book.live.values()]
    assert targets == pytest.approx([expected, expected])
    assert expected < 1.1100, "목표가는 처음보다 가까워야 한다"
    # 그 가격에서 묶음의 이익 = 첫 진입이 노렸던 이익
    profit = sum(p["volume"] * (expected - p["entry_price"]) for p in book.live.values())
    assert profit == pytest.approx(0.10 * reward * STEP)


def test_a_short_basket_is_defended_upwards(desk):
    _, book, drive = desk
    drive(sig=Signal("숏진입", -1, stop=1.1050, target=1.0900))
    (base,) = book.live.values()
    assert base["stop_loss"] == pytest.approx(1.1000 + 4 * STEP)
    drive(1.1050, Signal("유지", -1, stop=1.1100, target=1.0900, close=1.1050))
    legs = book.by_tag()
    assert legs["m1"]["side"] == "SHORT" and legs["m1"]["volume"] == pytest.approx(0.20)


# ─── 양방 대응 ───────────────────────────────────────────────────

def test_a_shock_bar_is_hedged_instead_of_doubled(desk, monkeypatch):
    _, book, drive = desk
    enter(drive)
    monkeypatch.setattr(eng.fx_defense, "shock", lambda df, mult: -1)
    drive(1.0940)
    legs = book.by_tag()
    assert set(legs) == {"m0", "h"}, "급변동에는 물량을 늘리지 않는다"
    hedge = legs["h"]
    assert hedge["side"] == "SHORT" and hedge["volume"] == pytest.approx(0.10)
    assert hedge["stop_loss"] == pytest.approx(1.0940 + STEP), "한 단계 되돌리면 풀린다"
    assert hedge["take_profit"] == pytest.approx(1.0800), "묶음의 최종 손절에서 이익을 챙긴다"


def test_the_hedge_matches_the_whole_basket(desk, monkeypatch):
    _, book, drive = desk
    enter(drive)
    drive(1.0950)                                    # 조용히 밀려 한 번 추가
    monkeypatch.setattr(eng.fx_defense, "shock", lambda df, mult: -1)
    drive(1.0890)                                    # 다음 단계는 급변동으로 왔다
    assert book.by_tag()["h"]["volume"] == pytest.approx(0.30)


def test_a_locked_basket_is_left_alone(desk, monkeypatch):
    _, book, drive = desk
    enter(drive)
    monkeypatch.setattr(eng.fx_defense, "shock", lambda df, mult: -1)
    drive(1.0940)
    monkeypatch.setattr(eng.fx_defense, "shock", lambda df, mult: 0)
    drive(1.0880)
    assert set(book.by_tag()) == {"m0", "h"}


def test_defense_resumes_once_the_hedge_is_gone(desk, monkeypatch):
    """되돌림에 양방 포지션이 손절됐다. 다시 밀리면 묶음을 다시 돌본다."""
    _, book, drive = desk
    enter(drive)
    monkeypatch.setattr(eng.fx_defense, "shock", lambda df, mult: -1)
    drive(1.0940)
    hedge = book.by_tag()["h"]
    del book.live[str(hedge["ticket"])]
    monkeypatch.setattr(eng.fx_defense, "shock", lambda df, mult: 0)
    drive(1.0945)
    assert set(book.by_tag()) == {"m0", "m1"}


def test_a_shock_in_our_favour_is_not_a_reason_to_hedge(desk, monkeypatch):
    _, book, drive = desk
    enter(drive)
    monkeypatch.setattr(eng.fx_defense, "shock", lambda df, mult: 1)
    drive(1.0950)
    assert set(book.by_tag()) == {"m0", "m1"}


def test_shock_reads_the_last_bar_against_the_ones_before():
    idx = pd.date_range("2025-01-01", periods=30, freq="1h", tz="UTC")
    calm = pd.DataFrame({"open": 1.1, "high": 1.1010, "low": 1.0990, "close": 1.1},
                        index=idx)
    assert defense.shock(calm, 2.0) == 0
    crash = calm.copy()
    crash.iloc[-1] = [1.1000, 1.1002, 1.0950, 1.0955]
    assert defense.shock(crash, 2.0) == -1
    spike = calm.copy()
    spike.iloc[-1] = [1.1000, 1.1050, 1.0998, 1.1045]
    assert defense.shock(spike, 2.0) == 1
    assert defense.shock(calm.iloc[:5], 2.0) == 0, "봉이 모자라면 판단하지 않는다"


# ─── 청산 ────────────────────────────────────────────────────────

def test_an_opposite_signal_closes_every_leg_and_the_hedge(desk, monkeypatch):
    _, book, drive = desk
    enter(drive)
    drive(1.0950)
    monkeypatch.setattr(eng.fx_defense, "shock", lambda df, mult: -1)
    drive(1.0890)
    assert len(book.live) == 3
    drive(1.0890, Signal("청산", 0, stop=None, close=1.0890))
    assert book.live == {}


def test_no_new_entry_while_a_leg_refuses_to_close(desk, monkeypatch):
    """묶음을 닫다가 하나가 실패했다. 남은 것을 두고 반대로 새로 들어가지 않는다."""
    _, book, drive = desk
    enter(drive)
    drive(1.0950)
    stuck = book.by_tag()["m1"]["ticket"]
    real_close = book.close

    def close(ticket, **kw):
        if ticket == stuck:
            raise eng.MT5Error("청산 거부")
        return real_close(ticket, **kw)

    monkeypatch.setattr(eng.broker, "close_position", close)
    drive(1.0950, Signal("숏진입", -1, close=1.0950, stop=1.1000, target=1.0850,
                         bar_time="2025-06-01T20:00:00+00:00"))
    assert [p["side"] for p in book.live.values()] == ["LONG"], "남은 롱 옆에 숏을 열면 안 된다"


def test_the_baskets_side_is_the_first_entrys_even_with_a_hedge_on(desk, monkeypatch):
    """반대 방향 포지션이 섞여 있어도 같은 방향 신호를 '반대 신호'로 읽지 않는다."""
    _, book, drive = desk
    enter(drive)
    monkeypatch.setattr(eng.fx_defense, "shock", lambda df, mult: -1)
    drive(1.0940)
    monkeypatch.setattr(eng.fx_defense, "shock", lambda df, mult: 0)
    drive(1.0940, Signal("롱진입", 1, close=1.0940, stop=1.0890,
                         bar_time="2025-06-01T15:00:00+00:00"))
    assert set(book.by_tag()) == {"m0", "h"}


# ─── 가드와 재시작 ───────────────────────────────────────────────

def test_a_basket_takes_one_slot_however_many_legs_it_has(desk):
    trader, book, drive = desk
    trader.guard.limits.max_positions = 1
    enter(drive)
    drive(1.0950)
    drive(1.0900)
    assert set(book.by_tag()) == {"m0", "m1", "m2"}, "자기 자리 때문에 추가가 막히면 안 된다"


def test_the_kill_switch_stops_adds_too(desk):
    trader, book, drive = desk
    enter(drive)
    trader.guard.engage_kill("사용자")
    drive(1.0950)
    assert len(book.live) == 1
    assert "킬 스위치" in trader.journal.tail(5, kinds=["blocked"])[-1]["reason"]


def test_a_stopped_engine_does_not_add(desk):
    trader, book, drive = desk
    enter(drive)
    trader.state.running = False
    drive(1.0950)
    assert len(book.live) == 1


def test_the_basket_survives_a_restart(desk):
    """엔진은 기억이 없다. 역할은 브로커의 comment 에서 다시 읽는다."""
    trader, book, drive = desk
    enter(drive)
    drive(1.0950)
    trader._known.clear()
    drive(1.0900)
    assert set(book.by_tag()) == {"m0", "m1", "m2"}


def test_a_position_from_before_this_feature_is_not_defended(desk):
    _, book, drive = desk
    book.open("EURUSD", "LONG", 0.10, 1.0950, 1.1100, tag_value="rqEURUSD2506011200")
    drive(1.0900)
    assert len(book.live) == 1


def test_defense_orders_are_journalled_with_what_they_answer(desk, monkeypatch):
    trader, _, drive = desk
    enter(drive)
    drive(1.0950)
    monkeypatch.setattr(eng.fx_defense, "shock", lambda df, mult: -1)
    drive(1.0890)
    rows = trader.journal.tail(10, kinds=["order"])[::-1]           # 오래된 것부터
    assert [(r.get("defense"), r.get("level")) for r in rows] == \
        [(None, None), ("martingale", 1), ("hedge", 1)]
    assert rows[1]["basket"] == BASE_TAG and rows[1]["strategies"] == []


def test_the_status_shows_the_rules(desk):
    trader, _, _ = desk
    assert trader.status()["defense"] == {"enabled": True, "max_adds": 3,
                                          "multiplier": 2.0, "shock_mult": 2.0,
                                          "final_stop": True, "seed_stop": 0.0,
                                          "breakeven_after": 0}


# ─── 2회 이상 추가한 묶음은 본전에서 닫는다 ──────────────────────

def average(book) -> float:
    legs = list(book.live.values())
    return sum(p["volume"] * p["entry_price"] for p in legs) / sum(p["volume"] for p in legs)


def test_the_first_add_still_aims_for_the_original_profit(live_rules):
    trader, book, drive = live_rules
    enter(drive)
    drive(1.0950)
    reward = float(trader.cfg.require("risk.reward_ratio"))
    expected = average(book) + reward * STEP * 0.10 / 0.30
    assert [p["take_profit"] for p in book.live.values()] == pytest.approx([expected] * 2)
    assert expected > average(book), "한 번 추가까지는 수익을 노린다"


def test_from_the_second_add_the_basket_closes_at_break_even(live_rules):
    _, book, drive = live_rules
    enter(drive)
    drive(1.0950)
    drive(1.0900)
    assert set(book.by_tag()) == {"m0", "m1", "m2"}
    assert [p["take_profit"] for p in book.live.values()] == pytest.approx([average(book)] * 3), \
        "깊이 물린 묶음은 평균 단가에서 빠져나온다"

    drive(1.0850)                                       # 3회째에도 본전이다
    assert [p["take_profit"] for p in book.live.values()] == pytest.approx([average(book)] * 4)
    assert all(p["stop_loss"] is None for p in book.live.values())


def test_break_even_is_off_unless_configured(nostop):
    trader, book, drive = nostop
    enter(drive)
    for price in (1.0950, 1.0900):
        drive(price)
    assert all(p["take_profit"] > average(book) for p in book.live.values())


# ─── 계좌 손실 한도 ──────────────────────────────────────────────

def with_account(monkeypatch, book, *, equity: float, seed: float = 10_000.0):
    """입금액과 평가 자산을 정하고, 전량 청산이 가짜 계좌를 비우게 한다."""
    from dataclasses import replace

    def close_all(*, allow_live=False, reason=""):
        out = [book.close(int(ticket), reason=reason) for ticket in list(book.live)]
        return out

    monkeypatch.setattr(eng, "account", lambda: replace(DEMO, equity=equity))
    monkeypatch.setattr(eng, "cash_flows", lambda *a, **k: {"net_funding": seed})
    monkeypatch.setattr(eng.FxTrader, "_seed_cache", None)
    monkeypatch.setattr(eng.broker, "close_all", close_all)


def test_nothing_happens_while_equity_is_above_the_floor(live_rules, monkeypatch):
    trader, book, drive = live_rules
    enter(drive)
    with_account(monkeypatch, book, equity=8_000.01)
    assert trader._seed_stop() is False
    assert len(book.live) == 1 and trader.guard.kill_switch is False


def test_twenty_percent_down_closes_everything_and_stops_the_engine(live_rules, monkeypatch):
    trader, book, drive = live_rules
    enter(drive)
    drive(1.0950)
    assert len(book.live) == 2
    with_account(monkeypatch, book, equity=8_000.00)

    assert trader._seed_stop() is True
    assert book.live == {}, "앱이 연 포지션을 전부 닫는다"
    assert trader.guard.kill_switch is True and trader.state.running is False
    (row,) = [r for r in trader.journal.tail(20, kinds=["engine"])
              if r.get("event") == "seed_stop"]
    assert row["seed"] == 10_000.0 and row["floor"] == pytest.approx(8_000.0)
    assert len(trader.journal.tail(20, kinds=["close"])) == 2, "닫은 것도 장부에 남는다"

    trader.state.running = True                         # 사람이 킬을 풀기 전에는 못 들어간다
    drive(1.1000, Signal("롱진입", 1, bar_time="2025-06-02T09:00:00+00:00"))
    assert book.live == {}
    assert trader._seed_stop() is False, "이미 멈췄으면 다시 닫으려 들지 않는다"


def test_the_floor_follows_the_deposit_not_the_balance(live_rules, monkeypatch):
    """입금액이 5만이면 4만 아래에서 닫는다. 지금 잔고가 얼마든 기준은 넣은 돈이다."""
    trader, book, drive = live_rules
    enter(drive)
    with_account(monkeypatch, book, equity=40_500.0, seed=50_000.0)
    assert trader._seed_stop() is False
    with_account(monkeypatch, book, equity=39_999.0, seed=50_000.0)
    assert trader._seed_stop() is True


def test_an_unknown_deposit_leaves_the_limit_off_and_the_status_says_so(live_rules, monkeypatch):
    trader, book, drive = live_rules
    enter(drive)
    with_account(monkeypatch, book, equity=100.0, seed=0.0)
    assert trader._seed_stop() is False and len(book.live) == 1
    assert trader.status()["seed_stop"] == {"limit": 0.20, "seed": 0.0, "floor": None}


def test_the_status_shows_the_floor(live_rules, monkeypatch):
    trader, book, _ = live_rules
    with_account(monkeypatch, book, equity=10_000.0)
    assert trader.status()["seed_stop"] == {"limit": 0.20, "seed": 10_000.0, "floor": 8_000.0}


def test_the_limit_is_off_unless_configured(nostop, monkeypatch):
    trader, book, drive = nostop
    enter(drive)
    with_account(monkeypatch, book, equity=1.0)
    assert trader._seed_stop() is False and len(book.live) == 1


def test_the_deposit_is_read_once_in_a_while_not_every_poll(live_rules, monkeypatch):
    trader, book, _ = live_rules
    with_account(monkeypatch, book, equity=10_000.0)
    calls = []
    monkeypatch.setattr(eng, "cash_flows",
                        lambda *a, **k: calls.append(1) or {"net_funding": 10_000.0})
    for _ in range(20):
        trader._seed_stop()
    assert len(calls) == 1, "거래 내역 전체를 30초마다 훑지 않는다"


def test_the_engine_loop_checks_the_limit_between_bars(live_rules, monkeypatch):
    """봉 마감을 기다리지 않는다 — 루프가 돌 때마다 본다."""
    trader, book, drive = live_rules
    enter(drive)
    with_account(monkeypatch, book, equity=7_500.0)
    monkeypatch.setattr(eng, "POLL_SECONDS", 0.05)
    monkeypatch.setattr(trader, "_newest_bar", lambda: None)
    trader._spawn()
    trader._thread.join(timeout=5)
    assert not trader._thread.is_alive(), "한도에 닿았으면 루프가 선다"
    assert book.live == {} and trader.guard.kill_switch is True


# ─── 최종 손절은 선택 사항이다 ───────────────────────────────────

def test_without_the_final_stop_the_entry_carries_no_stop(nostop):
    _, book, drive = nostop
    enter(drive)
    (pos,) = book.live.values()
    assert pos["stop_loss"] is None, "손절을 꺼 두었으면 걸지 않는다"
    assert pos["comment"] == BASE_TAG, "단계 간격은 태그에 적는다 — 손절에서 읽을 수 없으니"
    assert pos["take_profit"] == pytest.approx(1.1100) and pos["volume"] == 0.10


def test_martingale_keeps_working_without_a_stop(nostop):
    trader, book, drive = nostop
    enter(drive)
    for price in (1.0950, 1.0900, 1.0850):
        drive(price)
    legs = book.by_tag()
    assert set(legs) == {"m0", "m1", "m2", "m3"}
    assert [legs[k]["volume"] for k in ("m0", "m1", "m2", "m3")] == \
        pytest.approx([0.10, 0.20, 0.40, 0.80])
    assert all(p["stop_loss"] is None for p in book.live.values()), \
        "추가분에도, 목표가를 옮길 때도 손절을 붙이지 않는다"
    targets = {round(p["take_profit"], 6) for p in book.live.values()}
    assert len(targets) == 1, "묶음 전체가 같은 목표가를 쓴다"


def test_without_a_stop_nothing_closes_the_basket_after_the_last_add(nostop):
    """마지막 추가 뒤에도 계속 밀린다. 더 싣지도 닫지도 않는다 — 그게 이 설정의 뜻이다."""
    _, book, drive = nostop
    enter(drive)
    for price in (1.0950, 1.0900, 1.0850, 1.0790, 1.0700, 1.0500):
        drive(price)
    assert set(book.by_tag()) == {"m0", "m1", "m2", "m3"}


def test_the_step_is_read_from_the_tag_after_a_restart(nostop):
    """손절이 없으니 간격을 손절에서 읽을 수 없다. 태그에 있어야 재시작을 견딘다."""
    trader, book, drive = nostop
    enter(drive)
    trader._known.clear()
    drive(1.0951)
    assert len(book.live) == 1, "간격(0.0050)에 닿기 전에는 추가하지 않는다"
    drive(1.0950)
    assert set(book.by_tag()) == {"m0", "m1"}


def test_the_hedge_has_no_target_when_there_is_no_final_stop(nostop, monkeypatch):
    _, book, drive = nostop
    enter(drive)
    monkeypatch.setattr(eng.fx_defense, "shock", lambda df, mult: -1)
    drive(1.0940)
    hedge = book.by_tag()["h"]
    assert hedge["side"] == "SHORT" and hedge["volume"] == pytest.approx(0.10)
    assert hedge["take_profit"] is None, "이익을 챙길 최종 손절 자리가 없다"
    assert hedge["stop_loss"] == pytest.approx(1.0940 + STEP), "한 단계 되돌리면 풀린다"


def test_a_basket_opened_with_a_stop_keeps_it_after_the_option_is_switched_off(nostop):
    """
    설정을 끄기 전에 연 묶음. 태그에 간격이 없고 손절이 걸려 있다 — 간격은 그
    손절에서 읽고, 추가분도 그 손절을 함께 쓴다. 설정이 열린 포지션의 손절을
    걷어 가지 않는다.
    """
    _, book, drive = nostop
    book.open("EURUSD", "LONG", 0.10, 1.0800, 1.1100, tag_value="rqEURUSD2506011200m0")
    drive(1.0950)
    by_role = {p["comment"][-2:]: p for p in book.live.values()}
    assert set(by_role) == {"m0", "m1"}
    assert by_role["m1"]["stop_loss"] == pytest.approx(1.0800)
    assert by_role["m0"]["stop_loss"] == pytest.approx(1.0800)


def test_the_r_multiple_comes_from_the_recorded_risk_not_the_stop():
    """손절을 멀리 걸거나 안 걸면 손절 가격으로는 1R 을 알 수 없다."""
    from src.server.fx_api import _r_multiple
    trade = {"exit_price": 1.1050, "side": "LONG", "entry_price": 1.1000}
    assert _r_multiple(trade, {"price": 1.1000, "stop_loss": None,
                               "risk_per_unit": 0.0050}) == 1.0
    assert _r_multiple(trade, {"price": 1.1000, "stop_loss": 1.0800,
                               "risk_per_unit": 0.0050}) == 1.0
    assert _r_multiple({**trade, "side": "SHORT"},
                       {"price": 1.1000, "stop_loss": None, "risk_per_unit": 0.0050}) == -1.0
    # 예전 기록(1R 이 없다)은 손절 가격으로 계산한다
    assert _r_multiple(trade, {"price": 1.1000, "stop_loss": 1.0950}) == 1.0
    assert _r_multiple(trade, {"price": 1.1000, "stop_loss": None}) is None


def test_the_entry_is_journalled_with_its_risk(nostop):
    trader, _, drive = nostop
    enter(drive)
    drive(1.0950)
    rows = trader.journal.tail(5, kinds=["order"])[::-1]
    assert [r["risk_per_unit"] for r in rows] == pytest.approx([STEP, STEP])
    assert rows[0]["stop_loss"] is None


def test_the_final_stop_is_on_unless_switched_off():
    assert defense.rules(Config({"fx": {"defense": {"enabled": True}}})).final_stop is True
    off = defense.rules(Config({"fx": {"defense": {"final_stop": False}}}))
    assert off.final_stop is False


def test_the_step_travels_in_the_tag():
    assert defense.base_suffix(0.0050, 0.00001) == "p500m0"
    assert defense.base_suffix(1.5, 0.001) == "p1500m0"            # 엔 통화쌍
    assert defense.base_suffix(0.0050, 0.0) == "m0", "포인트를 모르면 적지 않는다"
    tag = broker.tag("EURUSD", "2025-06-01T12:00:00+00:00", broker.SHORT_CODE) + "p99999m3"
    assert len(tag) <= 31, "간격을 적어도 comment 한도 안이다"
    assert defense.role({"comment": "rqEURUSD2506011200p500m0"}) == ("base", 0)
    assert defense.role({"comment": "rqEURUSD2506011200p500m2"}) == ("add", 2)
    assert defense.role({"comment": "rqEURUSD2506011200p500h"}) == ("hedge", 0)


# ─── 규칙 읽기 ───────────────────────────────────────────────────

@pytest.mark.parametrize("comment, expected", [
    ("rqEURUSD2506011200m0", ("base", 0)),
    ("rqs.EURUSD2506011200m2", ("add", 2)),
    ("rqEURUSD2506011200h", ("hedge", 0)),
    ("rqEURUSD2506011200", None),
    ("", None),
])
def test_the_role_is_read_from_the_tag(comment, expected):
    assert defense.role({"comment": comment}) == expected


def test_defense_tags_keep_the_group_and_fit_the_comment_limit():
    tag = broker.tag("EURUSD", "2025-06-01T12:00:00+00:00", broker.SHORT_CODE) + "m3"
    assert len(tag) <= 31 and broker.group_of({"comment": tag}) == "short"


def test_the_rules_default_to_off_and_are_clamped():
    assert defense.rules(Config({})).enabled is False
    wild = defense.rules(Config({"fx": {"defense": {"max_adds": 40, "multiplier": 0.5}}}))
    assert wild.max_adds == 9 and wild.multiplier == 1.0


def test_volumes_are_rounded_down_to_the_lot_step():
    assert defense.round_step(0.279, 0.01) == pytest.approx(0.27)
    assert np.isclose(defense.round_step(0.2, 0.01), 0.2)
