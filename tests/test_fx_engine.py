"""
FX 자동매매 루프.

여기서 검사하는 것은 **전략이 맞았는가**가 아니라 **틀렸을 때 얼마나 크게
틀리는가**다. 이 루프가 돌리는 신호는 FX 에서 우위가 확인되지 않았고
(총우위조차 1시간봉 t=0.46), 그 사실 자체도 테스트로 못박는다 — 화면과 장부가
그 판정을 조용히 지워 버리면 사용자는 검증된 것을 돌리고 있다고 착각한다.

    · 꺼진 채로 시작한다 (사람이 켜야 돈다)
    · 킬 스위치는 조건 없이 이긴다
    · 손절 없는 신호로는 진입하지 않는다
    · 이미 들고 있으면 또 사지 않는다
    · 반대 신호가 오면 먼저 닫고 나서 본다
    · 파생 데이터를 요구하는 전략은 FX 에서 제외된다
    · 막힌 이유가 장부에 남는다

MT5 와 broker 는 전부 가짜로 갈아 끼운다. 테스트가 돈을 움직이면 안 된다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.execution.guards import RiskGuard, RiskLimits
from src.execution.journal import Journal
from src.fx import engine as eng
from src.fx.broker import Fill
from src.fx.mt5 import Account, MT5Error, SymbolInfo
from src.fx.symbols import Pair


DEMO = Account(login=1, server="Demo", currency="USD", balance=10_000.0,
               equity=10_000.0, margin_free=9_000.0, is_demo=True,
               trade_allowed=True, leverage=100)

INFO = SymbolInfo(name="EURUSD", digits=5, point=0.00001, spread_points=10,
                  spread_price=0.0001, volume_min=0.01, volume_max=50.0,
                  volume_step=0.01, contract_size=100_000.0,
                  bid=1.1000, ask=1.1002)

PAIR = Pair(mt5="EURUSD", base="EUR", quote="USD")


def fake_bars(n: int = 600) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(7)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0004, n))
    return pd.DataFrame({
        "open": close, "high": close + 0.0006, "low": close - 0.0006,
        "close": close, "volume": rng.uniform(100, 500, n),
        "spread": np.full(n, 10.0),
    }, index=idx)


class Signal:
    """signal_for 가 돌려주는 것의 최소 형태."""

    def __init__(self, action="롱진입", direction=1, stop=1.0950,
                 target=1.1100, close=1.1000, bar_time="2025-06-01T12:00:00+00:00"):
        self.action, self.direction = action, direction
        self.stop_price, self.target_price = stop, target
        self.close, self.bar_time = close, bar_time
        self.risk_per_unit = None if stop is None else abs(close - stop)
        self.score, self.regime = 0.9, "상승추세"

    def as_dict(self):
        return {"action": self.action, "direction": self.direction,
                "close": self.close, "bar_time": self.bar_time,
                "score": self.score, "regime": self.regime,
                "stop_price": self.stop_price}


@pytest.fixture
def trader(tmp_path, monkeypatch):
    """가짜 MT5·가짜 브로커를 끼운 엔진."""
    monkeypatch.setattr(eng, "account", lambda: DEMO)
    monkeypatch.setattr(eng, "symbol_info", lambda name: INFO)
    monkeypatch.setattr(eng, "bars", lambda *a, **k: fake_bars())
    monkeypatch.setattr(eng, "positions", lambda: {})

    sent: list[dict] = []
    closed: list[dict] = []

    def fake_open(symbol, side, lots, sl, tp=None, *, tag_value="",
                  allow_live=False, **kw):
        sent.append({"symbol": symbol, "side": side, "lots": lots, "sl": sl,
                     "tp": tp, "tag": tag_value, "allow_live": allow_live})
        return Fill(ticket=100 + len(sent), symbol=symbol, side=side,
                    volume=lots, price=1.1002, stop_loss=sl, take_profit=tp,
                    tag=tag_value, retcode=10009, comment="Done")

    def fake_close(ticket, *, allow_live=False, reason="", **kw):
        closed.append({"ticket": ticket, "reason": reason})
        return {"ticket": ticket, "closed": True, "symbol": "EURUSD",
                "side": "LONG", "volume": 0.1, "price": 1.1, "profit": -12.0}

    monkeypatch.setattr(eng.broker, "open_position", fake_open)
    monkeypatch.setattr(eng.broker, "close_position", fake_close)
    monkeypatch.setattr(eng.broker, "lots_for_risk",
                        lambda *a, **k: (0.10, ""))
    monkeypatch.setattr(eng.broker, "reconcile",
                        lambda known: {"open": dict(known), "closed_elsewhere": [],
                                       "unknown_to_us": [], "not_ours": [],
                                       "at": "now"})

    t = eng.FxTrader(journal=Journal(tmp_path / "j.jsonl"))
    monkeypatch.setattr(t, "watchlist", lambda: [PAIR])
    # 여기는 기본 진입·청산을 본다. 역행 대응은 test_fx_defense.py 가 따로 본다.
    monkeypatch.setattr(t, "defense", lambda: eng.fx_defense.Rules(enabled=False))
    t.sent, t.closed_calls = sent, closed
    return t


def drive(t, sig, monkeypatch):
    """이 신호가 나왔다고 치고 한 바퀴 돌린다."""
    monkeypatch.setattr(eng, "signal_for", lambda *a, **k: sig)
    return t.run_once()


# ─── 켜지지 않은 채로 시작한다 ───────────────────────────────────

def test_the_engine_starts_switched_off(trader, monkeypatch):
    """
    **켜자마자 주문이 나가면 안 된다.** 검증되지 않은 신호라 사람이 켜는
    행위가 곧 동의다.
    """
    assert trader.state.running is False
    drive(trader, Signal(), monkeypatch)
    assert trader.sent == [], "엔진이 꺼져 있는데 주문이 나갔다"


def test_running_after_start(trader, monkeypatch):
    trader.start()
    drive(trader, Signal(), monkeypatch)
    assert len(trader.sent) == 1


# ─── 검증 판정을 숨기지 않는다 ───────────────────────────────────

def test_the_status_carries_the_verdict(trader):
    """
    이 판정이 화면에서 사라지면 사용자는 검증된 것을 돌리고 있다고 착각한다.
    """
    v = trader.status()["verdict"]
    assert v["passed"] is False
    assert "우위가 확인되지 않았" in v["headline"]
    assert "t=0.46" in v["detail"], "무엇을 근거로 그렇게 말하는지가 빠졌다"


def test_the_verdict_is_written_to_the_journal_on_start(trader):
    """나중에 '무엇을 알고 켰나'에 답할 수 있어야 한다."""
    trader.start()
    rows = trader.journal.tail(10, kinds=["engine"])
    assert rows and "우위가 확인되지 않았" in rows[0]["verdict"]


# ─── 킬 스위치 ───────────────────────────────────────────────────

def test_the_kill_switch_stops_the_loop(trader, monkeypatch):
    """사람이 멈추라고 한 것은 조건부가 아니다."""
    trader.start()
    trader.kill("사용자")
    drive(trader, Signal(), monkeypatch)
    assert trader.sent == []
    assert trader.status()["guard"]["kill_switch"] is True


def test_release_lets_it_run_again(trader, monkeypatch):
    trader.start()
    trader.kill("사용자")
    trader.release()
    trader.start()
    drive(trader, Signal(), monkeypatch)
    assert len(trader.sent) == 1


# ─── 손절 없는 신호 ──────────────────────────────────────────────

def test_no_entry_without_a_stop(trader, monkeypatch):
    """
    손절 없이 들어가면 1R 이 정의되지 않는다. 그러면 수량도 정할 수 없고,
    최악의 손실에 상한이 없다.
    """
    trader.start()
    drive(trader, Signal(stop=None), monkeypatch)
    assert trader.sent == []
    reasons = [r["reason"] for r in trader.journal.tail(10, kinds=["blocked"])]
    assert any("손절" in r for r in reasons)


def test_the_stop_is_passed_to_the_order(trader, monkeypatch):
    trader.start()
    drive(trader, Signal(stop=1.0950, target=1.1100), monkeypatch)
    assert trader.sent[0]["sl"] == pytest.approx(1.0950)
    assert trader.sent[0]["tp"] == pytest.approx(1.1100)


# ─── 중복 진입 ───────────────────────────────────────────────────

def test_a_pair_already_held_is_not_bought_again(trader, monkeypatch):
    trader.start()
    drive(trader, Signal(), monkeypatch)
    drive(trader, Signal(), monkeypatch)
    assert len(trader.sent) == 1, "이미 들고 있는데 또 샀다"


def test_the_order_carries_a_bar_tag(trader, monkeypatch):
    """같은 봉에 두 번 나가지 않게 막는 표식이 실려야 한다."""
    trader.start()
    drive(trader, Signal(), monkeypatch)
    assert trader.sent[0]["tag"].startswith("rq")


# ─── 반대 신호 ───────────────────────────────────────────────────

def test_an_opposite_signal_closes_before_it_opens(trader, monkeypatch):
    """
    양방향을 동시에 들면 수수료만 내고 서로 상쇄된다. 먼저 닫고 나서 본다.
    """
    trader.start()
    drive(trader, Signal(action="롱진입", direction=1), monkeypatch)
    drive(trader, Signal(action="숏진입", direction=-1), monkeypatch)
    assert len(trader.closed_calls) == 1
    assert [s["side"] for s in trader.sent] == ["LONG", "SHORT"]


def test_an_exit_signal_closes_the_position(trader, monkeypatch):
    trader.start()
    drive(trader, Signal(action="롱진입", direction=1), monkeypatch)
    drive(trader, Signal(action="청산", direction=0, stop=None), monkeypatch)
    assert len(trader.closed_calls) == 1


def test_exits_still_happen_when_the_engine_is_stopped(trader, monkeypatch):
    """
    **정지는 새 진입을 막는 것이지 포지션을 방치하는 것이 아니다.**
    들고 있는 것을 못 닫으면 정지 버튼이 위험을 키운다.
    """
    trader.start()
    drive(trader, Signal(action="롱진입", direction=1), monkeypatch)
    trader.stop()
    drive(trader, Signal(action="청산", direction=0, stop=None), monkeypatch)
    assert len(trader.closed_calls) == 1


# ─── 가드 ────────────────────────────────────────────────────────

def test_the_position_limit_is_enforced(trader, monkeypatch):
    trader.start()
    trader.guard.limits.max_positions = 0
    drive(trader, Signal(), monkeypatch)
    assert trader.sent == []
    reasons = [r["reason"] for r in trader.journal.tail(10, kinds=["blocked"])]
    assert any("동시 보유" in r for r in reasons)


def test_a_zero_lot_is_not_ordered(trader, monkeypatch):
    """
    최소 랏 미만이면 broker 가 0 을 준다. 그걸 그대로 주문하면 거부되거나,
    올려서 넣으면 의도한 위험보다 커진다.
    """
    trader.start()
    monkeypatch.setattr(eng.broker, "lots_for_risk",
                        lambda *a, **k: (0.0, "최소 랏보다 작습니다"))
    drive(trader, Signal(), monkeypatch)
    assert trader.sent == []
    reasons = [r["reason"] for r in trader.journal.tail(10, kinds=["blocked"])]
    assert any("최소 랏" in r for r in reasons)


def test_a_rejected_order_is_journalled_not_raised(trader, monkeypatch):
    """조용히 삼키면 '왜 진입 안 했지'에 답할 수 없고, 터지면 루프가 멈춘다."""
    trader.start()

    def boom(*a, **k):
        raise MT5Error("시장이 닫혀 있습니다")

    monkeypatch.setattr(eng.broker, "open_position", boom)
    drive(trader, Signal(), monkeypatch)
    rows = trader.journal.tail(10, kinds=["reject"])
    assert rows and "시장이 닫혀" in rows[0]["message"]


# ─── 데모 전용 ───────────────────────────────────────────────────

def test_orders_are_sent_as_demo_only_by_default(trader, monkeypatch):
    """allow_live 가 그대로 브로커까지 전달돼야 관문이 의미가 있다."""
    trader.start()
    drive(trader, Signal(), monkeypatch)
    assert trader.sent[0]["allow_live"] is False


def test_allow_live_comes_from_config(trader, monkeypatch):
    assert trader.allow_live is False, "설정 기본값이 실계좌 허용이면 안 된다"


# ─── 전략 선택 ───────────────────────────────────────────────────

def test_strategies_needing_derivatives_are_excluded(trader):
    """
    FX 에는 자금조달률도 미결제약정도 테이커 비율도 없다. 없는 열을 0 으로
    채우면 신호가 아니라 잡음이 된다.
    """
    for s in trader.strategies():
        assert not any(str(n).startswith(("perp_", "funding", "oi_", "basis"))
                       for n in getattr(s, "needs", ()))


def test_some_strategies_survive_the_filter(trader):
    assert len(trader.strategies()) > 0, "FX 에서 돌릴 전략이 하나도 없다"


def test_a_banned_strategy_is_left_out(trader):
    """fx.exclude_strategies 에 적힌 이름은 투표에 들어오지 않는다."""
    banned = set(trader.cfg.get("fx.exclude_strategies") or [])
    assert banned, "금지목록이 비어 있다 — 이 테스트가 아무것도 재지 않는다"
    assert banned.isdisjoint({s.name for s in trader.strategies()})


def test_the_ban_is_a_denylist_so_new_strategies_still_arrive():
    """
    **허용목록이 아니라 금지목록이어야 하는 이유.** 하나를 빼려고 나머지를
    열거하면, 나중에 전략을 추가할 때 목록에 넣는 것을 잊고 조용히 빠진다.
    고장 난 것처럼 보이지 않아서 더 나쁘다.
    """
    from src.config import Config
    from src.fx.engine import FxTrader

    everything = {s.name for s in FxTrader(cfg=Config({})).strategies()}
    one_banned = {s.name for s in FxTrader(
        cfg=Config({"fx": {"exclude_strategies": ["order_flow_fade"]}})
    ).strategies()}

    assert "order_flow_fade" in everything, "금지하지 않으면 들어와야 한다"
    assert everything - one_banned == {"order_flow_fade"}, (
        "금지한 하나 말고 다른 전략까지 사라졌다")


# ─── 대조 ────────────────────────────────────────────────────────

def test_a_position_closed_at_the_broker_is_noticed(trader, monkeypatch):
    """
    손절이 브로커 쪽에서 체결됐는데 그걸 모르면, 이미 없는 포지션을 들고
    있다고 착각한 채 동시 보유 한도를 채운다.
    """
    trader.start()
    drive(trader, Signal(), monkeypatch)
    assert trader._known

    monkeypatch.setattr(eng.broker, "reconcile",
                        lambda known: {"open": {}, "closed_elsewhere": list(known),
                                       "unknown_to_us": [], "not_ours": [],
                                       "at": "now"})
    drive(trader, Signal(action="유지", direction=1), monkeypatch)
    assert trader._known == {}
    assert trader.journal.tail(10, kinds=["reconcile"])


# ─── 오류를 삼키지 않는다 ────────────────────────────────────────

def test_a_broken_symbol_does_not_stop_the_whole_scan(trader, monkeypatch):
    """한 종목이 실패해도 나머지는 봐야 한다."""
    trader.start()

    def boom(*a, **k):
        raise MT5Error("EURUSD 데이터를 받지 못했습니다")

    monkeypatch.setattr(eng, "signal_for", boom)
    out = trader.run_once()
    assert out["last_error"] is None, "종목 하나 실패가 루프 전체를 세웠다"
    assert trader.journal.tail(10, kinds=["error"])


# ─── 조용히 멈춰 있는 앱을 막는다 ────────────────────────────────
#
# 아래 둘은 실제로 겪은 결함이다. 화면은 멀쩡하고 오류도 없는데 엔진이
# 영원히 아무것도 하지 않았다. 고장 난 것처럼 보이지 않아서 더 나빴다.


def test_enough_bars_are_pulled_to_cover_the_training_window(trader):
    """
    signal_for 는 최근 6개월에서 전략을 돌려 가중치를 뽑는다. 그보다 적게
    받아 오면 레짐별 거래 수가 min_trades(20)에 못 미쳐 **모든 가중치가 0**이
    되고, 점수가 항상 0.000 이 나온다.

    실측: 1500봉(86일)이면 적격 칸 0/21 · 가중치 합 0.000.
          3488봉(203일)이면 세 전략이 발언권을 얻는다.
    """
    train_days = int(trader.cfg.get("backtest.walk_forward.train_months", 6)) * 31
    per_day = eng._BARS_PER_DAY[trader.interval]
    # 주말을 빼도 학습 구간을 다 덮어야 한다
    assert trader.bars_needed() >= train_days * (5 / 7) * per_day


def test_the_bar_count_follows_the_timeframe(trader, monkeypatch):
    """4시간봉으로 바꾸면 봉 개수도 따라 줄어야 한다 — 같은 기간을 덮어야 하므로."""
    monkeypatch.setattr(type(trader), "interval",
                        property(lambda self: "4h"))
    four = trader.bars_needed()
    monkeypatch.setattr(type(trader), "interval",
                        property(lambda self: "1h"))
    one = trader.bars_needed()
    assert four < one


def test_fx_does_not_inherit_the_crypto_entry_threshold(trader):
    """
    암호화폐 문턱 0.85 를 그대로 쓰면 **신호가 한 번도 안 나온다.**
    거기서는 한 전략이 가중치를 독식했지만 FX 는 서너 전략에 갈리고 동시에
    같은 방향으로 강하게 말하는 일이 드물다.

    EUR/USD 3,488봉 실측: |점수| 95분위 0.17, 0.85 돌파 0봉, 0.35 돌파 1.89%.
    """
    wp = trader.weight_params()
    assert wp.entry_threshold < 0.85, "암호화폐 문턱을 그대로 물려받았다"
    assert wp.entry_threshold > 0, "문턱이 0 이면 잡음까지 다 진입한다"


def test_the_threshold_comes_from_the_fx_config(trader, monkeypatch):
    monkeypatch.setattr(trader.cfg, "get",
                        lambda k, d=None: 0.5 if k == "fx.entry_threshold"
                        else (True if k == "fx.calibrate_strength" else d))
    assert trader.weight_params().entry_threshold == 0.5


# ─── 명목가는 계좌 통화로 잰다 ───────────────────────────────────

# USD/JPY 159엔. 1랏이 1틱(0.001엔) 움직이면 100엔 = 약 0.63달러다.
YEN = SymbolInfo(name="USDJPY", digits=3, point=0.001, spread_points=10,
                 spread_price=0.01, volume_min=0.01, volume_max=50.0,
                 volume_step=0.01, contract_size=100_000.0, bid=159.00, ask=159.02,
                 tick_size=0.001, tick_value=100_000 * 0.001 / 159.0)


def test_notional_is_measured_in_the_account_currency():
    """0.1랏 USD/JPY 는 1만 달러어치다. 159만이 아니다."""
    assert YEN.notional(0.1, 159.0) == pytest.approx(10_000.0)
    # 틱 정보가 없으면 호가 통화가 계좌 통화라고 본다 (EUR/USD 는 그게 맞다)
    assert INFO.notional(0.1, 1.1) == pytest.approx(11_000.0)


def test_a_yen_pair_is_not_blocked_by_the_notional_cap(trader, monkeypatch):
    """
    **실제로 있었던 일이다.** 명목가를 엔 그대로 달러 한도와 비교해서 AUD/JPY
    진입이 "명목가 상한 초과 (2,245,593 > 300,060)"로 거부됐다. 엔 통화쌍은 한 번도
    진입하지 못했다.
    """
    monkeypatch.setattr(eng, "symbol_info", lambda name: YEN)
    trader.start()
    drive(trader, Signal(close=159.0, stop=158.0, target=160.5), monkeypatch)
    assert len(trader.sent) == 1, trader.journal.tail(3, kinds=["blocked"])
