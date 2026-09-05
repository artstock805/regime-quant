"""
MT5 주문 계층 — 계좌에 실제로 손을 대는 곳.

여기서 고정하는 것은 **사고를 막는 성질**이다. 전략이 맞았는지 틀렸는지는
백테스트가 볼 일이고, 이 파일은 "틀렸을 때 얼마나 크게 틀리는가"를 본다.

    · 실계좌에는 기본적으로 주문이 안 나간다
    · 손절이 진입과 같은 요청에 붙는다 (프로그램이 죽어도 지켜진다)
    · 최소 랏 미만이면 올려서 넣지 않고 건너뛴다 (조용한 리스크 증가 방지)
    · 같은 봉에 두 번 들어가지 않는다
    · 사람이 손으로 연 포지션은 건드리지 않는다

가짜 MT5 를 끼워 실제 터미널 없이 돌린다. 진짜 브로커에 주문을 내는 테스트는
쓰지 않는다 — 테스트가 돈을 움직이면 안 된다.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.fx import broker
from src.fx.broker import LiveAccountBlocked, OrderRejected
from src.fx.mt5 import Account, MT5Error, SymbolInfo


DONE = 10009
UNSUPPORTED_FILLING = 10030


class FakeTick:
    def __init__(self, bid: float, ask: float) -> None:
        self.bid, self.ask, self.time = bid, ask, 1_700_000_000


class FakeResult:
    def __init__(self, retcode=DONE, order=555, volume=0.1, price=1.1,
                 comment="Done") -> None:
        self.retcode, self.order = retcode, order
        self.volume, self.price, self.comment = volume, price, comment


class FakeMT5:
    """
    주문 요청을 받아 적기만 하는 가짜 브로커.

    **무엇을 보냈는지**가 검사 대상이다. 손절이 붙어 있는지, 랏이 얼마인지,
    magic 이 우리 것인지는 요청을 들여다봐야 알 수 있다.
    """

    ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1
    TRADE_ACTION_DEAL, TRADE_ACTION_SLTP = 1, 6
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2
    TRADE_RETCODE_DONE = DONE

    def __init__(self, *, filling_mode=1, loss_per_lot=-1000.0) -> None:
        self.requests: list[dict] = []
        self.result = FakeResult()
        self.filling_mode = filling_mode
        self.loss_per_lot = loss_per_lot
        self.error = (0, "")

    def order_send(self, req):
        self.requests.append(dict(req))
        return self.result

    def order_calc_profit(self, kind, symbol, lots, entry, stop):
        return self.loss_per_lot * lots

    def symbol_info(self, name):
        return type("S", (), {"filling_mode": self.filling_mode})()

    def symbol_info_tick(self, name):
        return FakeTick(1.1000, 1.1002)

    def last_error(self):
        return self.error


DEMO = Account(login=1, server="Demo", currency="USD", balance=10_000.0,
               equity=10_000.0, margin_free=9_000.0, is_demo=True,
               trade_allowed=True, leverage=100)
LIVE = Account(login=2, server="Real", currency="USD", balance=10_000.0,
               equity=10_000.0, margin_free=9_000.0, is_demo=False,
               trade_allowed=True, leverage=100)

INFO = SymbolInfo(name="EURUSD", digits=5, point=0.00001, spread_points=10,
                  spread_price=0.0001, volume_min=0.01, volume_max=50.0,
                  volume_step=0.01, contract_size=100_000.0,
                  bid=1.1000, ask=1.1002)


@pytest.fixture
def fake(monkeypatch):
    """가짜 MT5 + 데모 계좌 + 빈 포지션."""
    m = FakeMT5()
    live_positions: dict[str, dict] = {}
    monkeypatch.setattr(broker, "_mt5", lambda: m)
    monkeypatch.setattr(broker, "initialise", lambda: True)
    monkeypatch.setattr(broker, "account", lambda: DEMO)
    monkeypatch.setattr(broker, "symbol_info", lambda name: INFO)
    monkeypatch.setattr(broker, "positions", lambda: live_positions)
    m.positions = live_positions          # 테스트가 채워 넣을 수 있게
    return m


def a_position(ticket=1, symbol="EURUSD", side="LONG", magic=broker.MAGIC,
               comment="rqEURUSD2509051200", volume=0.1):
    return {"ticket": ticket, "symbol": symbol, "side": side, "volume": volume,
            "entry_price": 1.1, "current_price": 1.1, "stop_loss": 1.09,
            "take_profit": None, "profit": 0.0, "swap": 0.0, "magic": magic,
            "opened_at": "2025-09-05T12:00:00+00:00", "comment": comment}


# ─── 실계좌 관문 ─────────────────────────────────────────────────

def test_a_live_account_is_refused_by_default(fake, monkeypatch):
    """
    **기본값이 사람을 지켜야 한다.** 사용자의 터미널에는 실계좌가 붙어 있을 수
    있고, FX 에서 검증되지 않은 전략이 거기로 흘러 들어가면 그게 이 프로젝트에서
    가장 큰 사고다.
    """
    monkeypatch.setattr(broker, "account", lambda: LIVE)
    with pytest.raises(LiveAccountBlocked) as exc:
        broker.open_position("EURUSD", "LONG", 0.1, 1.09)
    assert "실계좌" in str(exc.value)
    assert fake.requests == [], "실계좌인데 주문이 나갔다"


def test_a_live_account_needs_an_explicit_opt_in(fake, monkeypatch):
    monkeypatch.setattr(broker, "account", lambda: LIVE)
    broker.open_position("EURUSD", "LONG", 0.1, 1.09, allow_live=True)
    assert len(fake.requests) == 1


def test_nothing_is_sent_when_algo_trading_is_switched_off(fake, monkeypatch):
    """MT5 에서 자동매매를 꺼 두면 주문이 조용히 실패한다. 먼저 알려 준다."""
    off = Account(**{**DEMO.as_dict(), "trade_allowed": False})
    monkeypatch.setattr(broker, "account", lambda: off)
    with pytest.raises(MT5Error) as exc:
        broker.open_position("EURUSD", "LONG", 0.1, 1.09)
    assert "자동매매" in str(exc.value)
    assert fake.requests == []


# ─── 손절이 진입에 붙는가 ────────────────────────────────────────

def test_the_stop_rides_along_with_the_entry(fake):
    """
    **가장 중요한 성질이다.** 진입 후에 손절을 따로 걸면, 그 사이에 프로그램이
    죽거나 네트워크가 끊겼을 때 포지션이 맨몸으로 남는다. 브로커 쪽에 손절이
    있으면 우리가 꺼져 있어도 지켜진다.
    """
    broker.open_position("EURUSD", "LONG", 0.1, 1.0950, 1.1100)
    req = fake.requests[0]
    assert req["sl"] == pytest.approx(1.0950)
    assert req["tp"] == pytest.approx(1.1100)
    assert len(fake.requests) == 1, "손절을 별도 요청으로 보냈다"


def test_a_long_with_the_stop_above_entry_is_refused(fake):
    """부호를 뒤집으면 손절이 즉시 체결된다. 보내기 전에 잡는다."""
    with pytest.raises(OrderRejected) as exc:
        broker.open_position("EURUSD", "LONG", 0.1, 1.2000)
    assert "롱" in str(exc.value)
    assert fake.requests == []


def test_a_short_with_the_stop_below_entry_is_refused(fake):
    with pytest.raises(OrderRejected):
        broker.open_position("EURUSD", "SHORT", 0.1, 1.0000)
    assert fake.requests == []


def test_entry_marks_the_order_as_ours(fake):
    broker.open_position("EURUSD", "LONG", 0.1, 1.09, tag_value="rqEURUSD2509051200")
    req = fake.requests[0]
    assert req["magic"] == broker.MAGIC
    assert req["comment"] == "rqEURUSD2509051200"


def test_a_broker_rejection_is_raised_not_swallowed(fake):
    """조용히 실패하면 '왜 진입 안 했지'에 답할 수 없다."""
    fake.result = FakeResult(retcode=UNSUPPORTED_FILLING, comment="Unsupported filling")
    with pytest.raises(OrderRejected) as exc:
        broker.open_position("EURUSD", "LONG", 0.1, 1.09)
    assert exc.value.retcode == UNSUPPORTED_FILLING


def test_filling_mode_follows_what_the_broker_allows(fake):
    """
    체결 방식이 안 맞으면 10030 으로 거부된다. 종목마다 다르므로 물어보고 고른다.
    """
    fake.filling_mode = 2            # IOC 만 허용
    broker.open_position("EURUSD", "LONG", 0.1, 1.09)
    assert fake.requests[0]["type_filling"] == FakeMT5.ORDER_FILLING_IOC


# ─── 랏 계산 ─────────────────────────────────────────────────────

def test_lots_come_out_so_the_stop_costs_exactly_one_r(fake):
    """
    1R = 계좌 x 위험비율. 고정 랏으로 하면 변동성이 큰 통화쌍에서 같은 1R 이
    몇 배가 되고, 백테스트의 R 배수와 실전이 어긋난다.
    """
    fake.loss_per_lot = -1000.0      # 1랏이 손절까지 가면 1,000 손실
    lots, why = broker.lots_for_risk("EURUSD", "LONG", 1.1000, 1.0900, 100.0)
    assert why == ""
    assert lots == pytest.approx(0.10)


def test_lots_are_rounded_down_never_up(fake):
    """
    **올림하면 의도한 것보다 큰 위험을 진다.** 조용히 리스크가 커지는 쪽이
    훨씬 나쁘다.
    """
    fake.loss_per_lot = -1000.0
    lots, _ = broker.lots_for_risk("EURUSD", "LONG", 1.1, 1.09, 179.0)  # 0.179랏
    assert lots == pytest.approx(0.17)


def test_a_risk_too_small_for_the_minimum_lot_is_skipped(fake):
    """
    최소 랏이 0.01 인데 계산이 0.003 이면, 0.01 로 올려 넣는 것은 의도한 위험의
    3배를 지는 것이다. 그럴 바에는 건너뛴다.
    """
    fake.loss_per_lot = -1000.0
    lots, why = broker.lots_for_risk("EURUSD", "LONG", 1.1, 1.09, 3.0)
    assert lots == 0.0
    assert "최소" in why and "건너뜁니다" in why


def test_lots_are_capped_at_the_brokers_maximum(fake):
    fake.loss_per_lot = -1.0
    lots, why = broker.lots_for_risk("EURUSD", "LONG", 1.1, 1.09, 100_000.0)
    assert lots <= INFO.volume_max


def test_the_broker_does_the_currency_conversion(fake):
    """
    EUR/USD 와 USD/JPY 는 계좌통화 환산이 다르다. 직접 계산하면 크로스에서
    틀리므로 order_calc_profit 에 맡긴다 — 브로커의 실제 환산을 그대로 쓴다.
    """
    calls = []
    fake.order_calc_profit = lambda *a: (calls.append(a), -1000.0)[1]
    broker.lots_for_risk("USDJPY", "SHORT", 156.0, 156.5, 100.0)
    assert calls and calls[0][1] == "USDJPY"


def test_a_zero_stop_distance_is_refused(fake):
    lots, why = broker.lots_for_risk("EURUSD", "LONG", 1.1, 1.1, 100.0)
    assert lots == 0.0 and "손절 거리" in why


# ─── 중복 진입 ───────────────────────────────────────────────────

def test_the_same_bar_cannot_be_entered_twice(fake):
    """
    루프가 두 번 돌거나 재시작해도 두 번 들어가면 안 된다. MT5 에는 클라이언트
    주문 ID 가 없어서 comment 태그로 대신한다.
    """
    t = broker.tag("EURUSD", datetime(2025, 9, 5, 12, 0, tzinfo=timezone.utc))
    fake.positions["1"] = a_position(comment=t)
    with pytest.raises(OrderRejected) as exc:
        broker.open_position("EURUSD", "LONG", 0.1, 1.09, tag_value=t)
    assert "중복" in str(exc.value)
    assert fake.requests == []


def test_the_tag_is_stable_for_one_bar_and_changes_on_the_next():
    bar = datetime(2025, 9, 5, 12, 0, tzinfo=timezone.utc)
    assert broker.tag("EURUSD", bar) == broker.tag("EURUSD", bar.isoformat())
    later = datetime(2025, 9, 5, 13, 0, tzinfo=timezone.utc)
    assert broker.tag("EURUSD", bar) != broker.tag("EURUSD", later)
    assert broker.tag("EURUSD", bar) != broker.tag("GBPUSD", bar)


def test_the_tag_fits_in_the_mt5_comment_field():
    """MT5 comment 는 대개 31자에서 잘린다. 잘리면 중복 판정이 무너진다."""
    t = broker.tag("EURUSD", datetime.now(timezone.utc))
    assert len(t) <= 31


# ─── 남의 포지션 ─────────────────────────────────────────────────

def test_a_hand_opened_position_is_not_ours(fake):
    """
    사용자 계좌에는 사람이 연 금·지수 포지션이 섞여 있다. 우리 것으로 착각해
    청산하면 그건 사고다.
    """
    assert not broker.ours(a_position(symbol="GOLD", magic=0, comment=""))
    assert broker.ours(a_position())


def test_flatten_leaves_hand_opened_positions_alone(fake):
    fake.positions.update({
        "1": a_position(ticket=1, symbol="EURUSD"),
        "2": a_position(ticket=2, symbol="GOLD", magic=0, comment=""),
    })
    closed = broker.close_all()
    assert [c["ticket"] for c in closed] == [1]
    assert all(r["symbol"] == "EURUSD" for r in fake.requests)


def test_reconcile_reports_positions_we_did_not_open(fake):
    fake.positions.update({
        "1": a_position(ticket=1),
        "2": a_position(ticket=2, symbol="US100Cash", magic=0, comment=""),
    })
    r = broker.reconcile(known={"1": {}})
    assert r["not_ours"] == ["US100Cash"]
    assert list(r["open"]) == ["1"]


# ─── 청산 ────────────────────────────────────────────────────────

def test_closing_sends_the_opposite_side_for_that_ticket(fake):
    fake.positions["7"] = a_position(ticket=7, side="LONG", volume=0.3)
    out = broker.close_position(7, reason="signal")
    req = fake.requests[0]
    assert req["type"] == FakeMT5.ORDER_TYPE_SELL
    assert req["position"] == 7
    assert req["volume"] == pytest.approx(0.3)
    assert out["closed"] is True


def test_closing_a_short_buys_it_back(fake):
    fake.positions["8"] = a_position(ticket=8, side="SHORT")
    broker.close_position(8)
    assert fake.requests[0]["type"] == FakeMT5.ORDER_TYPE_BUY


def test_closing_something_already_gone_is_not_an_error(fake):
    """
    손절이 브로커 쪽에서 이미 체결됐을 수 있다. 그건 실패가 아니라 정상이다 —
    예외를 던지면 루프가 거기서 멈춘다.
    """
    out = broker.close_position(999)
    assert out["closed"] is False
    assert fake.requests == []


def test_a_rejected_close_is_raised(fake):
    fake.positions["9"] = a_position(ticket=9)
    fake.result = FakeResult(retcode=10018, comment="Market closed")
    with pytest.raises(OrderRejected) as exc:
        broker.close_position(9)
    assert exc.value.retcode == 10018


# ─── 손절 이동 ───────────────────────────────────────────────────

def test_moving_the_stop_does_not_touch_the_position(fake):
    fake.positions["3"] = a_position(ticket=3)
    broker.move_stop(3, 1.1050)
    req = fake.requests[0]
    assert req["action"] == FakeMT5.TRADE_ACTION_SLTP
    assert req["sl"] == pytest.approx(1.1050)
    assert "volume" not in req, "손절만 옮기려는데 수량이 실렸다"


# ─── 대조 ────────────────────────────────────────────────────────

def test_reconcile_notices_a_position_that_closed_elsewhere(fake):
    """
    손절 체결·수동 청산·재시작으로 늘 어긋난다. 조용히 넘기면 이미 닫힌 것을
    들고 있다고 착각한 채 동시 보유 한도를 채운다.
    """
    r = broker.reconcile(known={"1": {}, "2": {}})
    assert sorted(r["closed_elsewhere"]) == ["1", "2"]


def test_reconcile_notices_our_position_missing_from_our_records(fake):
    fake.positions["5"] = a_position(ticket=5)
    r = broker.reconcile(known={})
    assert r["unknown_to_us"] == ["5"]
