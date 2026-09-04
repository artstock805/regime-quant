"""
실행 계층 — 돈이 걸리는 부분. 안전 속성을 코드로 못박는다.

자동매매에서 계좌를 날리는 건 대개 전략이 아니라 **배관**이다.
같은 주문이 두 번 나가거나, 손절이 안 걸렸거나, 멈추라고 했는데 안 멈추거나,
내 기록과 거래소가 어긋난 채로 계속 돌거나.

여기서 고정하는 것
    1) 기본이 테스트넷인가
    2) 같은 봉·같은 목적이면 같은 주문 id 인가 (중복 주문 방지)
    3) 진입 직후 **반드시** 손절이 거래소에 걸리는가
    4) 가드가 막으면 주문이 나가지 않는가
    5) 킬 스위치가 다른 모든 조건을 이기는가
    6) 거래소 상태를 진실로 삼는가

거래소는 전부 대역으로 세운다. 실제 네트워크를 두드리는 테스트는 여기 없다.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.config import Config
from src.execution import guards as guards_mod
from src.execution.exchange import (
    LIVE, TESTNET, BinanceFutures, ExchangeError, MissingCredentials, SymbolRules,
)
from src.execution.guards import RiskGuard, RiskLimits
from src.execution.journal import Journal
from src.execution.trader import Trader, TraderConfig, client_id


# ─── 대역 ─────────────────────────────────────────────────────────

RULES = SymbolRules(symbol="BTCUSDT", tick_size=Decimal("0.1"),
                    step_size=Decimal("0.001"), min_qty=Decimal("0.001"),
                    min_notional=Decimal("5"))


class FakeExchange:
    """거래소 대역. 무엇을 받았는지 전부 기록한다."""

    def __init__(self, positions=None, balance=10_000.0, fail_on=None):
        self._positions = positions or {}
        self._balance = balance
        self.orders: list[dict] = []
        self.cancelled: list[str] = []
        self.fail_on = fail_on or set()
        self.testnet = True

    def sync_time(self):
        return 0

    def balance(self):
        return {"wallet": self._balance, "available": self._balance}

    def positions(self):
        return dict(self._positions)

    def rules(self, symbol):
        return RULES

    def cancel_all(self, symbol):
        self.cancelled.append(symbol)
        return {}

    def _record(self, kind, **kw):
        if kind in self.fail_on:
            raise ExchangeError(f"{kind} 거부 (테스트)")
        self.orders.append({"kind": kind, **kw})
        return {"clientOrderId": kw.get("client_id"), "status": "NEW"}

    def place_market(self, symbol, side, qty, *, client_id, reduce_only=False):
        return self._record("market", symbol=symbol, side=side, qty=qty,
                            client_id=client_id, reduce_only=reduce_only)

    def place_stop(self, symbol, side, stop_price, *, client_id):
        return self._record("stop", symbol=symbol, side=side,
                            stop_price=stop_price, client_id=client_id)

    def place_target(self, symbol, side, stop_price, *, client_id):
        return self._record("target", symbol=symbol, side=side,
                            stop_price=stop_price, client_id=client_id)


class FakeSignal:
    def __init__(self, symbol="BTCUSDT", action="롱진입", close=100.0,
                 stop=96.0, target=106.0, bar_time="2026-09-04T10:00:00+00:00"):
        self.symbol, self.action, self.close = symbol, action, close
        self.stop_price, self.target_price = stop, target
        self.risk_per_unit = abs(close - stop) if stop else None
        self.bar_time = bar_time
        self.score, self.regime_korean = 0.9, "상승추세"
        self.direction = 1 if action == "롱진입" else -1 if action == "숏진입" else 0


def _cfg():
    return Config({
        "market": "futures_um",
        "universe": ["BTCUSDT"],
        "strategies": ["taker_flow_momentum"],
        "risk": {"risk_per_trade": 0.01, "atr_stop_mult": 4.0,
                 "reward_ratio": 1.5, "daily_loss_limit": 0.03},
        "backtest": {"fee_rate": 0.0002, "slippage_bps": 2.0,
                     "walk_forward": {"train_months": 6, "test_months": 2}},
        "execution": {"testnet": True, "max_positions": 3},
        "live": {"interval": "1h", "symbols": ["BTCUSDT"]},
        "paths": {"derivatives_dir": "data/derivatives"},
    })


def _trader(tmp_path, exchange=None, **tcfg):
    ex = exchange or FakeExchange()
    t = Trader(cfg=_cfg(),
               tcfg=TraderConfig(testnet=True, interval="1h",
                                 symbols=["BTCUSDT"], **tcfg),
               exchange=ex, journal=Journal(tmp_path / "j.jsonl"))
    t.balance = ex.balance()
    return t, ex


# ─── 1) 기본이 테스트넷인가 ───────────────────────────────────────

def test_exchange_defaults_to_testnet():
    """기본값이 실전이면 실수 한 번이 돈이 된다."""
    assert BinanceFutures().base == TESTNET
    assert BinanceFutures(testnet=False).base == LIVE


def test_trader_defaults_to_testnet():
    assert TraderConfig().testnet is True


def test_missing_credentials_says_where_to_put_them(monkeypatch, tmp_path):
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    monkeypatch.delenv("BINANCE_TESTNET_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_TESTNET_API_SECRET", raising=False)
    from src.execution import exchange as ex_mod
    monkeypatch.setattr(ex_mod, "SECRETS", tmp_path / "nope.yaml")
    with pytest.raises(MissingCredentials) as e:
        ex_mod.load_credentials(testnet=True)
    assert "testnet.binancefuture.com" in str(e.value)


def test_secrets_file_is_gitignored():
    """키가 저장소에 들어가면 되돌릴 수 없다."""
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    if not (root / ".git").exists():
        pytest.skip("git 저장소가 아닙니다")
    r = subprocess.run(["git", "check-ignore", "-q", "config/secrets.yaml"],
                       cwd=root)
    assert r.returncode == 0, "config/secrets.yaml 이 .gitignore 에 없습니다"


# ─── 2) 중복 주문 방지 ────────────────────────────────────────────

def test_same_bar_and_purpose_yields_the_same_order_id():
    """
    **자동매매에서 가장 비싼 사고가 중복 주문이다.** 루프가 두 번 돌거나
    재시작해도 같은 id 라야 거래소가 두 번째를 거부한다.
    """
    a = client_id("BTCUSDT", "2026-09-04T10:00:00+00:00", "E")
    b = client_id("BTCUSDT", "2026-09-04T10:00:00+00:00", "E")
    assert a == b


@pytest.mark.parametrize("changed", [
    {"symbol": "ETHUSDT"}, {"bar_time": "2026-09-04T11:00:00+00:00"}, {"tag": "S"},
])
def test_order_id_changes_when_anything_changes(changed):
    base = {"symbol": "BTCUSDT", "bar_time": "2026-09-04T10:00:00+00:00", "tag": "E"}
    assert client_id(**base) != client_id(**{**base, **changed})


def test_order_id_fits_binance_limit():
    """clientOrderId 는 36자 제한. 넘으면 주문이 거부된다."""
    cid = client_id("1000SHIBUSDT", "2026-09-04T10:00:00+00:00", "E")
    assert len(cid) <= 36


def test_duplicate_rejection_is_treated_as_success(monkeypatch):
    """
    같은 id 를 다시 보내면 거래소가 -4015 로 거부한다. 재시작 뒤 정상 경로라
    실패로 취급하면 엔진이 멈춘다.
    """
    ex = BinanceFutures(testnet=True, key="k", secret="s")
    monkeypatch.setattr(ex, "rules", lambda s: RULES)

    def boom(*a, **k):
        raise ExchangeError("order 거부 [-4015] Duplicate order sent.")

    monkeypatch.setattr(ex, "_request", boom)
    out = ex.place_market("BTCUSDT", "BUY", 0.01, client_id="rqEBTC1")
    assert out["duplicate"] is True and out["status"] == "DUPLICATE_IGNORED"


def test_the_same_bar_is_not_acted_on_twice(tmp_path, monkeypatch):
    t, ex = _trader(tmp_path)
    sig = FakeSignal()
    monkeypatch.setattr("src.execution.trader.scan", lambda *a, **k: [sig])
    t.run_once()
    first = len(ex.orders)
    t.run_once()                       # 같은 봉
    assert len(ex.orders) == first, "같은 봉을 두 번 처리했다"


# ─── 3) 손절이 반드시 걸리는가 ────────────────────────────────────

def test_entry_always_places_a_stop_on_the_exchange(tmp_path):
    """
    **가장 중요한 검사.** 스톱을 로컬에서만 관리하면 프로그램이 죽는 순간
    포지션이 맨몸으로 남는다. 진입 직후 거래소에 걸어야 한다.
    """
    t, ex = _trader(tmp_path)
    t._open(FakeSignal(), "LONG")
    kinds = [o["kind"] for o in ex.orders]
    assert kinds[0] == "market", "진입이 먼저"
    assert "stop" in kinds, "손절이 거래소에 걸리지 않았다"
    stop = next(o for o in ex.orders if o["kind"] == "stop")
    assert stop["side"] == "SELL", "롱의 손절은 매도 쪽이어야 한다"
    assert stop["stop_price"] == 96.0


def test_short_entry_places_a_buy_side_stop(tmp_path):
    t, ex = _trader(tmp_path)
    t._open(FakeSignal(action="숏진입", close=100.0, stop=104.0, target=94.0),
            "SHORT")
    stop = next(o for o in ex.orders if o["kind"] == "stop")
    assert stop["side"] == "BUY" and stop["stop_price"] == 104.0


def test_no_entry_without_a_stop_price(tmp_path):
    """손절 없는 진입은 하지 않는다. 1R 을 정의할 수 없으면 수량도 정할 수 없다."""
    t, ex = _trader(tmp_path)
    assert t._open(FakeSignal(stop=None), "LONG") is None
    assert ex.orders == []


def test_closing_cancels_the_resting_stop_first(tmp_path):
    """
    청산 전에 스톱을 취소하지 않으면, 청산 뒤 남은 스톱이 반대 포지션을 연다.
    reduce-only 라도 잔여 주문이 남는 것은 위험하다.
    """
    t, ex = _trader(tmp_path)
    pos = {"symbol": "BTCUSDT", "qty": 0.5, "side": "LONG", "unrealised": 12.0}
    t._close("BTCUSDT", pos, "신호")
    assert ex.cancelled == ["BTCUSDT"], "스톱을 먼저 취소하지 않았다"
    assert ex.orders[-1]["reduce_only"] is True


# ─── 4) 가드가 막으면 주문이 안 나가는가 ──────────────────────────

def test_guard_block_prevents_any_order(tmp_path):
    t, ex = _trader(tmp_path)
    t.guard.limits.max_positions = 0          # 무조건 막힌다
    assert t._open(FakeSignal(), "LONG") is None
    assert ex.orders == [], "가드가 막았는데 주문이 나갔다"


def test_blocked_entries_are_journalled(tmp_path):
    """
    막힌 기록이 없으면 "왜 진입 안 하지"에 답할 수 없고, 그러면 가드를 꺼 버린다.
    """
    t, _ = _trader(tmp_path)
    t.guard.limits.max_positions = 0
    t._open(FakeSignal(), "LONG")
    blocked = t.journal.tail(10, kinds=["blocked"])
    assert blocked and "동시 보유" in blocked[0]["reason"]


def test_order_below_minimum_notional_is_refused(tmp_path):
    t, ex = _trader(tmp_path)
    t.balance = {"wallet": 1.0, "available": 1.0}      # 수량이 먼지처럼 작아진다
    t._open(FakeSignal(), "LONG")
    assert ex.orders == []


def test_position_size_targets_one_r():
    """
    고정 수량이면 변동성 큰 종목에서 1R 이 몇 배가 된다 — 백테스트의 R 배수와
    실전이 어긋난다. 손절폭으로 나눠야 한다.
    """
    g = RiskGuard(RiskLimits(risk_per_trade=0.01))
    qty, _ = g.position_size(balance=10_000, entry=100.0, stop=96.0)
    assert qty * 4.0 == pytest.approx(100.0)      # 계좌 1% = 100 USDT = 1R


def test_position_size_refuses_a_zero_stop_distance():
    g = RiskGuard()
    qty, why = g.position_size(balance=10_000, entry=100.0, stop=100.0)
    assert qty == 0 and "손절" in why


# ─── 5) 킬 스위치 ─────────────────────────────────────────────────

def test_kill_switch_beats_every_other_condition():
    """사람이 '멈춰'라고 했을 때 조건부로 멈추면 안 된다."""
    g = RiskGuard(RiskLimits(max_positions=99, min_balance=0))
    g.engage_kill("사용자 요청")
    d = g.check_entry(balance=1e9, open_positions=0, notional=1.0, symbol="BTCUSDT")
    assert not d and "킬 스위치" in d.reason


def test_kill_switch_can_be_released():
    g = RiskGuard()
    g.engage_kill("테스트")
    g.release_kill()
    assert g.check_entry(balance=10_000, open_positions=0,
                         notional=100, symbol="BTCUSDT").allowed


def test_daily_loss_limit_stops_new_entries():
    g = RiskGuard(RiskLimits(daily_loss_limit=0.03))
    g.record_realised(-400.0)                       # 계좌 10,000 의 4%
    d = g.check_entry(balance=10_000, open_positions=0, notional=100,
                      symbol="BTCUSDT")
    assert not d and "일일 손실" in d.reason


def test_daily_loss_resets_on_a_new_day():
    g = RiskGuard(RiskLimits(daily_loss_limit=0.03))
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    g.record_realised(-900.0, now=yesterday)
    d = g.check_entry(balance=10_000, open_positions=0, notional=100,
                      symbol="BTCUSDT")
    assert d.allowed, "어제 손실이 오늘을 막고 있다"


def test_order_rate_limit_catches_a_runaway_loop():
    g = RiskGuard(RiskLimits(max_orders_per_minute=3))
    for _ in range(3):
        g.record_order()
    d = g.check_entry(balance=10_000, open_positions=0, notional=100,
                      symbol="BTCUSDT")
    assert not d and "분당 주문" in d.reason


def test_notional_cap_limits_leverage_accidents():
    g = RiskGuard(RiskLimits(max_notional_ratio=1.0))
    d = g.check_entry(balance=1_000, open_positions=0, notional=5_000,
                      symbol="BTCUSDT")
    assert not d and "명목가" in d.reason


# ─── 6) 거래소가 진실인가 ─────────────────────────────────────────

def test_reconcile_records_a_mismatch(tmp_path):
    t, _ = _trader(tmp_path)
    t.positions = {"ETHUSDT": {"symbol": "ETHUSDT"}}       # 내 기록
    t._reconcile({"BTCUSDT": {"symbol": "BTCUSDT"}})        # 거래소
    rows = t.journal.tail(5, kinds=["reconcile"])
    assert rows and rows[0]["only_mine"] == ["ETHUSDT"]
    assert rows[0]["only_exchange"] == ["BTCUSDT"]


def test_no_reconcile_entry_when_they_agree(tmp_path):
    t, _ = _trader(tmp_path)
    t.positions = {"BTCUSDT": {}}
    t._reconcile({"BTCUSDT": {}})
    assert t.journal.tail(5, kinds=["reconcile"]) == []


def test_existing_same_side_position_is_left_alone(tmp_path):
    t, ex = _trader(tmp_path)
    actual = {"BTCUSDT": {"symbol": "BTCUSDT", "qty": 1.0, "side": "LONG",
                          "unrealised": 0.0}}
    assert t._act(FakeSignal(action="롱진입"), actual) is None
    assert ex.orders == [], "이미 같은 방향인데 또 진입했다"


def test_opposite_signal_closes_before_reversing(tmp_path):
    t, ex = _trader(tmp_path)
    actual = {"BTCUSDT": {"symbol": "BTCUSDT", "qty": 1.0, "side": "LONG",
                          "unrealised": 5.0}}
    t._act(FakeSignal(action="숏진입", close=100.0, stop=104.0, target=94.0), actual)
    kinds = [o["kind"] for o in ex.orders]
    assert kinds[0] == "market" and ex.orders[0]["reduce_only"] is True, "먼저 닫아야 한다"
    assert "stop" in kinds, "새 포지션에 손절이 걸려야 한다"


# ─── 웹훅 신호도 같은 관문을 지나는가 ─────────────────────────────

def test_webhook_signal_goes_through_the_same_guards(tmp_path, monkeypatch):
    """외부 신호라고 가드를 건너뛰면 그게 가장 위험한 구멍이 된다."""
    t, ex = _trader(tmp_path)
    # running 은 프로퍼티라 인스턴스에 못 꽂는다. 클래스 쪽을 잠깐 바꾼다.
    monkeypatch.setattr(Trader, "running", property(lambda self: True))
    monkeypatch.setattr("src.live.signal_for", lambda *a, **k: FakeSignal())
    t.guard.engage_kill("테스트")
    out = t.handle_external("BTCUSDT", "롱진입")
    assert ex.orders == [], "킬 스위치가 켜졌는데 웹훅 신호로 주문이 나갔다"
    assert out["ok"] is False


def test_webhook_is_refused_when_engine_is_off(tmp_path):
    t, _ = _trader(tmp_path)
    out = t.handle_external("BTCUSDT", "롱진입")
    assert not out["ok"] and "꺼져" in out["message"]


# ─── 장부 ─────────────────────────────────────────────────────────

def test_journal_rejects_unknown_event_kinds(tmp_path):
    """오타로 만든 종류가 섞이면 집계가 조용히 틀어진다."""
    j = Journal(tmp_path / "j.jsonl")
    with pytest.raises(ValueError):
        j.write("주문함", symbol="BTCUSDT")


def test_journal_survives_a_truncated_line(tmp_path):
    j = Journal(tmp_path / "j.jsonl")
    j.write("signal", symbol="BTCUSDT")
    with j.path.open("a", encoding="utf-8") as fh:
        fh.write('{"at": "half-written"\n')          # 쓰다 만 줄
    j.write("signal", symbol="ETHUSDT")
    rows = j.tail(10)
    assert len(rows) == 2, "깨진 줄 때문에 앞뒤 기록을 잃었다"


def test_journal_counts_only_todays_realised(tmp_path):
    j = Journal(tmp_path / "j.jsonl")
    j.write("close", symbol="BTCUSDT", realised=-50.0)
    j.write("close", symbol="ETHUSDT", realised=20.0)
    assert j.realised_today() == pytest.approx(-30.0)


# ─── 반올림 ───────────────────────────────────────────────────────

def test_quantity_rounds_down_to_step_size():
    """올림하면 잔고를 넘겨 주문이 거부된다. 내림이 맞다."""
    assert RULES.round_qty(0.0019) == pytest.approx(0.001)


def test_price_rounds_to_tick_size():
    assert RULES.round_price(100.17) == pytest.approx(100.1)


def test_min_notional_is_enforced():
    ok, why = RULES.is_tradable(0.001, 100.0)      # 명목가 0.1 < 5
    assert not ok and "명목" in why
