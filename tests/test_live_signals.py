"""
실시간 신호 — 백테스트와 같은 규칙을 쓰는가, 그리고 주문을 내지 않는가.

두 가지를 고정한다.

**1) 인과 규칙이 백테스트와 같아야 한다.**
    신호는 마감된 봉의 종가로 계산하고, 체결은 다음 봉 시가를 가정한다.
    진행 중인 봉으로 판단하면 화면의 신호와 검증 숫자가 다른 것을 가리키게 되고,
    그러면 5.9년 검증이 신호를 보증하지 못한다.

**2) 주문 실행 코드가 없어야 한다.**
    검증 1/5 상태에서 주문 경로가 생기면, 그게 실수로 열릴 수 있다.
    코드에 없으면 실수로 열릴 수 없다. 이 검사가 그 사실을 고정한다.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from src import live
from src.live import signals as live_signals


# ─── 주문 경로가 없다는 것을 고정한다 ─────────────────────────────

def test_live_module_has_no_order_execution():
    """
    주문 관련 이름이 하나도 없어야 한다. 있으면 누군가 열어 둔 것이다.
    이 테스트가 실패하면 '왜 추가했는가'를 먼저 답해야 한다.
    """
    src = Path(live_signals.__file__).read_text(encoding="utf-8")
    lowered = src.lower()
    banned = ["create_order", "place_order", "new_order", "submit_order",
              "post_order", "cancel_order", "api_secret", "api_key",
              "/fapi/v1/order", "futures_create_order"]
    hits = [b for b in banned if b in lowered]
    assert not hits, f"주문/인증 관련 코드가 들어왔습니다: {hits}"


def test_live_package_exports_only_read_paths():
    exported = set(live.__all__)
    assert exported == {"SymbolSignal", "StrategyVote", "actionable",
                        "scan", "signal_for"}


def test_signal_carries_the_no_order_warning():
    """화면에 경고가 남아야 한다 — 신호를 보는 것과 거는 것은 다른 결정이다."""
    sig = _fake_signal(action="롱진입")
    assert "주문" in sig.note


# ─── 인과 규칙 ────────────────────────────────────────────────────

def test_loader_requests_closed_bars_only(monkeypatch):
    """
    **가장 중요한 검사.** 진행 중인 봉을 받으면 신호가 매분 흔들리고
    백테스트 규칙과 어긋난다. closed_only=True 로 요청해야 한다.
    """
    captured = {}

    def fake_fetch(symbol, interval, start, end, *, market, closed_only):
        captured["closed_only"] = closed_only
        return pd.DataFrame()

    monkeypatch.setattr(live_signals, "fetch_klines", fake_fetch)
    cfg = _fake_cfg()
    live_signals._load(cfg, "BTCUSDT", "1h", use_derivatives=False, train_days=180)
    assert captured["closed_only"] is True, "진행 중인 봉을 받고 있습니다"


def test_warmup_is_long_enough_for_wilder_convergence():
    """
    Wilder 평활(RSI·ATR·ADX)은 초기 시딩 차이가 400봉에서 0 이 된다(실측).
    파생 z-score 는 168봉을 쓴다. 워밍업이 그보다 짧으면 화면 지표가
    백테스트 지표와 미묘하게 다른 값이 된다.
    """
    assert live_signals.WARMUP_BARS >= 400 + 168


def test_next_bar_time_is_one_interval_after_the_signal_bar():
    sig = _fake_signal()
    bar = pd.Timestamp(sig.bar_time)
    nxt = pd.Timestamp(sig.next_bar_time)
    assert nxt - bar == pd.Timedelta("1h")


# ─── 동작 판정 ────────────────────────────────────────────────────

@pytest.mark.parametrize("prev,cur,expected", [
    (0, 1, "롱진입"),
    (0, -1, "숏진입"),
    (1, 0, "청산"),
    (-1, 0, "청산"),
    (1, -1, "숏진입"),
    (-1, 1, "롱진입"),
    (1, 1, "유지"),
    (-1, -1, "유지"),
    (0, 0, "관망"),
])
def test_action_mapping_covers_every_transition(prev, cur, expected):
    assert live_signals._action(prev, cur) == expected


def test_actionable_filters_out_hold_and_wait():
    sigs = [_fake_signal(action=a) for a in
            ("롱진입", "유지", "청산", "관망", "숏진입")]
    got = [s.action for s in live_signals.actionable(sigs)]
    assert got == ["롱진입", "청산", "숏진입"]


# ─── 실패에 강한가 ────────────────────────────────────────────────

def test_scan_continues_when_one_symbol_fails(monkeypatch):
    """
    한 종목의 네트워크 실패가 전체 조회를 죽이면 안 된다.
    아카이브 수집에서 이미 겪은 실패 양상이다(503 한 장이 129종목을 날렸다).
    """
    calls = []

    def fake_signal_for(cfg, symbol, **kw):
        calls.append(symbol)
        if symbol == "BAD":
            raise RuntimeError("네트워크 오류")
        return _fake_signal(symbol=symbol)

    monkeypatch.setattr(live_signals, "signal_for", fake_signal_for)
    out = live_signals.scan(_fake_cfg(), ["A", "BAD", "B"])
    assert calls == ["A", "BAD", "B"], "실패 후에도 계속 조회해야 한다"
    assert [s.symbol for s in out] == ["A", "B"]


def test_scan_skips_symbols_with_insufficient_data(monkeypatch):
    monkeypatch.setattr(live_signals, "signal_for",
                        lambda cfg, symbol, **kw: None)
    assert live_signals.scan(_fake_cfg(), ["A", "B"]) == []


def test_signal_for_returns_none_when_bars_are_empty(monkeypatch):
    monkeypatch.setattr(live_signals, "_load",
                        lambda *a, **k: (pd.DataFrame(), pd.DataFrame()))
    assert live_signals.signal_for(_fake_cfg(), "BTCUSDT") is None


# ─── 직렬화 ───────────────────────────────────────────────────────

def test_signal_serialises_to_json_safe_dict():
    import json
    d = _fake_signal().as_dict()
    json.dumps(d, ensure_ascii=False)          # 터지지 않아야 한다
    assert d["votes"] and isinstance(d["votes"][0], dict)


# ─── 헬퍼 ─────────────────────────────────────────────────────────

def _fake_cfg():
    from src.config import Config
    return Config({
        "market": "futures_um",
        "universe": ["BTCUSDT"],
        "strategies": ["taker_flow_momentum"],
        "risk": {"risk_per_trade": 0.01, "atr_stop_mult": 4.0,
                 "reward_ratio": 1.5, "daily_loss_limit": 0.03},
        "backtest": {"fee_rate": 0.0002, "slippage_bps": 2.0,
                     "walk_forward": {"train_months": 6, "test_months": 2}},
        "paths": {"derivatives_dir": "data/derivatives"},
    })


def _fake_signal(symbol="BTCUSDT", action="유지"):
    return live_signals.SymbolSignal(
        symbol=symbol, interval="1h",
        bar_time="2026-09-03T10:00:00+00:00",
        next_bar_time="2026-09-03T11:00:00+00:00",
        close=100.0, regime="uptrend", regime_korean="상승추세",
        regime_confidence=0.5, score=0.9, threshold=0.85,
        action=action, direction=1, prev_direction=0,
        atr=1.0, stop_price=96.0, target_price=106.0, risk_per_unit=4.0,
        votes=[live_signals.StrategyVote(
            strategy="taker_flow_momentum", korean="테이커 흐름 추종",
            direction=1, strength=0.9, weight=1.0, contribution=0.9,
            has_voice=True)],
        note="신호는 마감된 봉 기준이다. 주문 기능은 없다.",
    )
