"""
상황판 서버 검증.

세션은 검증된 백테스트 결과를 되돌려 재구성한다. 그 되돌리기가 틀리면 화면의
숫자가 백테스트와 달라지므로, 되돌리기 자체를 검사한다.

특히 창 번호 정렬을 본다 — 건너뛴 창 때문에 창 목록과 가중치 목록의 인덱스가
어긋나서 상황판의 가중치가 전부 0으로 보인 사고가 있었다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_candles
from src.backtest.metrics import Metrics
from src.backtest.walkforward import WalkForwardResult, Window, make_windows
from src.regime import Regime, classify
from src.indicators import compute_all
from src.server.session import (
    PANEL_FEATURES, Session, _equity_track, _rebuild_position_track, _window_index,
    first_traded_bar,
)


def _bars(n: int = 400) -> pd.DataFrame:
    return make_candles(n, "1h", start="2026-01-01", seed=9)


# ─── 창 번호 정렬 ─────────────────────────────────────────────────

def test_window_index_marks_only_test_ranges():
    bars = _bars(24 * 40)
    windows = [
        Window(pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-01-11", tz="UTC"),
               pd.Timestamp("2026-01-11", tz="UTC"), pd.Timestamp("2026-01-21", tz="UTC")),
        Window(pd.Timestamp("2026-01-11", tz="UTC"), pd.Timestamp("2026-01-21", tz="UTC"),
               pd.Timestamp("2026-01-21", tz="UTC"), pd.Timestamp("2026-01-31", tz="UTC")),
    ]
    idx = _window_index(bars, windows)

    # 학습 구간(첫 10일)과 마지막 창 이후는 어느 창에도 속하지 않는다
    assert (idx[bars.index < windows[0].test_start] == -1).all()
    assert (idx[bars.index >= windows[-1].test_end] == -1).all()
    inside = (bars.index >= windows[0].test_start) & (bars.index < windows[-1].test_end)
    assert set(np.unique(idx[inside])) == {0, 1}
    # 각 창의 검증 구간만 그 창 번호를 갖는다
    m0 = (bars.index >= windows[0].test_start) & (bars.index < windows[0].test_end)
    assert (idx[m0] == 0).all()
    m1 = (bars.index >= windows[1].test_start) & (bars.index < windows[1].test_end)
    assert (idx[m1] == 1).all()


def test_skipped_windows_keep_list_alignment():
    """
    walk_forward 는 데이터가 부족한 창을 건너뛴다. 그때 가중치 리스트를 촘촘하게
    만들면 창 번호로 조회했을 때 남의 창 가중치를 읽는다. None 으로 자리를 채워
    창 목록과 길이가 같아야 한다.
    """
    from src.config import load_config
    from src.backtest import walk_forward
    from src.strategies import build_all

    cfg = load_config()
    bars = make_candles(24 * 400, "1h", start="2024-01-01", seed=4)
    feats = compute_all(bars)
    reg = classify(feats)["regime"]
    strats = build_all()[:2]

    # 데이터는 2024-01 부터인데 창은 2023-01 부터 만들게 해서 앞쪽 창을 굶긴다
    res = walk_forward(bars, feats, reg, strats, cfg, interval="1h",
                       start="2023-01-01", end="2025-06-01")

    assert len(res.weights_per_window) == len(res.windows), (
        "건너뛴 창 때문에 가중치 리스트가 창 목록과 어긋났다"
    )
    assert len(res.train_scores) == len(res.windows)
    assert any(w is None for w in res.weights_per_window), "굶은 창이 있어야 하는 시나리오"


def test_average_weights_ignores_skipped_windows():
    from src.backtest import average_weights, weight_stability

    cols = ["a", "b"]
    idx = [str(r) for r in Regime]
    w1 = pd.DataFrame(0.5, index=idx, columns=cols)
    w2 = pd.DataFrame(0.1, index=idx, columns=cols)
    avg = average_weights([w1, None, w2, None])
    assert avg.loc["uptrend", "a"] == pytest.approx(0.3)
    assert not weight_stability([w1, None, w2]).empty


# ─── 포지션 되돌리기 ─────────────────────────────────────────────

def _trade(bars, lo: int, hi: int, direction: int = 1, qty: float = 0.5) -> dict:
    entry = float(bars["close"].iloc[lo])
    exit_ = float(bars["close"].iloc[hi])
    net = (exit_ - entry) * qty * direction
    return {
        "symbol": "T", "strategy": "ensemble", "direction": direction,
        "entry_time": bars.index[lo], "entry_price": entry,
        "exit_time": bars.index[hi], "exit_price": exit_,
        "qty": qty, "notional": entry * qty, "risk_amount": 100.0,
        "gross_pnl": net, "fees": 1.0, "slip_cost": 0.5, "cost_r": 0.015,
        "net_pnl": net, "r_multiple": net / 100.0, "gross_r": net / 100.0 + 0.015,
        "price_return_pct": (exit_ / entry - 1) * 100 * direction,
        "hold_bars": hi - lo, "exit_reason": "target", "regime": "uptrend",
        "strength": 0.8, "mae_pct": 0.4, "mfe_pct": 1.2,
    }


def test_position_track_covers_holding_bars_only():
    bars = _bars(200)
    trades = pd.DataFrame([_trade(bars, 40, 55), _trade(bars, 120, 130, -1)])
    pos, fills = _rebuild_position_track(bars, trades, "1h")

    assert pos[39] is None and pos[56] is None
    assert all(pos[i] is not None for i in range(40, 56))
    assert pos[40]["held_bars"] == 0 and pos[55]["held_bars"] == 15
    assert pos[40]["side"] == "롱" and pos[120]["side"] == "숏"
    # 체결 이벤트는 진입 봉과 청산 봉에 하나씩
    assert [f["kind"] for f in fills[40]] == ["entry"]
    assert [f["kind"] for f in fills[55]] == ["exit"]


def test_position_upnl_matches_close_price():
    bars = _bars(200)
    t = _trade(bars, 40, 55, direction=1, qty=0.5)
    pos, _ = _rebuild_position_track(bars, pd.DataFrame([t]), "1h")
    j = 48
    expected = (bars["close"].iloc[j] - t["entry_price"]) * t["qty"]
    assert pos[j]["upnl"] == pytest.approx(expected)
    assert pos[j]["upnl_r"] == pytest.approx(expected / t["risk_amount"])


def test_empty_trades_yield_empty_track():
    bars = _bars(50)
    pos, fills = _rebuild_position_track(bars, pd.DataFrame(), "1h")
    assert all(p is None for p in pos) and fills == {}


def test_equity_track_fills_gaps_forward():
    bars = _bars(100)
    eq = pd.Series([10_000.0, 10_100.0], index=[bars.index[30], bars.index[60]])
    res = WalkForwardResult(symbol="T", oos_trades=pd.DataFrame(), oos_equity=eq,
                            metrics=Metrics.zero(), initial_equity=10_000.0)
    track = _equity_track(bars, res)

    assert track[0] == pytest.approx(10_000.0), "거래 전 구간은 초기자본"
    assert track[45] == pytest.approx(10_000.0), "구멍은 직전 값 유지"
    assert track[99] == pytest.approx(10_100.0)


# ─── 프레임 ──────────────────────────────────────────────────────

def _session(n: int = 460) -> Session:
    # bbw_percentile 이 200봉 lookback + 볼린저 20봉을 쓰므로, 레짐이 판정되려면
    # 최소 220봉이 지나야 한다. 짧게 잡으면 레짐이 전부 NaN 이 되어
    # 레짐 띠·투표가 빈 채로 통과하는 껍데기 테스트가 된다.
    bars = _bars(n)
    feats = compute_all(bars)
    regimes = classify(feats)
    strats = __import__("src.strategies", fromlist=["build_all"]).build_all()[:2]
    signals = {s.name: s.generate(bars, feats) for s in strats}

    trades = pd.DataFrame([_trade(bars, 300, 318)])
    idx = [str(r) for r in Regime]
    weights = pd.DataFrame(0.5, index=idx, columns=sorted(s.name for s in strats))
    res = WalkForwardResult(
        symbol="T", oos_trades=trades,
        oos_equity=pd.Series(10_000.0, index=bars.index),
        metrics=Metrics.zero(), initial_equity=10_000.0,
        weights_per_window=[weights],
        train_scores=[pd.DataFrame(columns=["strategy", "regime", "trades",
                                            "expectancy_r", "win_rate", "eligible"])],
        windows=[Window(bars.index[0], bars.index[50], bars.index[50], bars.index[-1])],
    )
    pos, fills = _rebuild_position_track(bars, trades, "1h")
    return Session(
        symbol="T", interval="1h", bars=bars, feats=feats, regimes=regimes,
        signals=signals, result=res, strategies=strats,
        weight_params=__import__("src.backtest", fromlist=["WeightParams"]).WeightParams(),
        window_of=_window_index(bars, res.windows),
        equity=_equity_track(bars, res), position_at=pos, fills_at=fills,
    )


def test_frame_shape_and_bounds():
    s = _session()
    f = s.frame(320)
    assert f["i"] == 320
    assert set(f) >= {"t", "bar", "regime", "votes", "score", "position", "equity",
                      "fills", "features", "semantics"}
    assert len(f["votes"]) == len(s.strategies)
    assert all(k in f["features"] for k in ("atr", "adx", "rsi") if k in PANEL_FEATURES)
    # 범위를 벗어난 인덱스는 잘려야 한다 (예외를 던지지 않는다)
    assert s.frame(-5)["i"] == 0
    assert s.frame(10**6)["i"] == s.n - 1


def test_frame_score_equals_sum_of_contributions():
    s = _session()
    for i in (300, 318, 400):
        f = s.frame(i)
        assert f["score"] == pytest.approx(sum(v["contrib"] for v in f["votes"]))


def test_undeclared_regime_gets_no_vote():
    """전략이 선언하지 않은 국면에서는 방향·확신도가 0이어야 한다."""
    s = _session()
    for i in range(260, 450):
        f = s.frame(i)
        for v in f["votes"]:
            if not v["declared"]:
                assert v["direction"] == 0.0 and v["strength"] == 0.0


def test_entry_fill_carries_the_decision_score():
    """
    체결은 직전 봉 판단으로 일어났다. 이번 봉 점수와 나란히 보여주면
    '점수 미달인데 진입했다'로 오해하므로, 판단 시점 점수를 실어야 한다.
    """
    s = _session()
    entry_bar = min(i for i, v in s.fills_at.items()
                    if any(x["kind"] == "entry" for x in v))
    f = s.frame(entry_bar)
    e = [x for x in f["fills"] if x["kind"] == "entry"][0]
    assert "decision_score" in e and "decision_t" in e
    prev = s.frame(entry_bar - 1)
    assert e["decision_score"] == pytest.approx(prev["score"])
    assert e["decision_t"] == prev["t"]


def test_meta_keeps_live_locked():
    """합격 기준을 통과하지 못했다. 실전 모드는 잠겨 있어야 한다."""
    m = _session().meta()
    assert m["live_unlocked"] is False
    assert m["live_locked_reason"]


def test_first_traded_bar_is_before_first_fill():
    s = _session()
    assert first_traded_bar(s) <= min(s.fills_at)


def test_candles_slice_shape():
    s = _session()
    c = s.candles(10, 20)
    assert len(c) == 10
    assert set(c[0]) == {"t", "o", "h", "l", "c", "v"}


# ─── API ─────────────────────────────────────────────────────────

@pytest.fixture
def client():
    pytest.importorskip("httpx", reason="TestClient 는 httpx 가 필요하다")
    from fastapi.testclient import TestClient

    from src.server import app as app_mod
    app_mod.app.state.session = _session()
    with TestClient(app_mod.app) as c:
        yield c
    app_mod.app.state.session = None


def test_api_meta(client):
    d = client.get("/api/meta").json()
    assert d["symbol"] == "T" and d["interval"] == "1h"
    assert d["live_unlocked"] is False
    assert "start_bar" in d


def test_api_frame(client):
    d = client.get("/api/frame/320").json()
    assert d["i"] == 320 and "votes" in d


def test_api_candles_clamps_range(client):
    d = client.get("/api/candles?lo=-10&hi=5").json()
    assert d["lo"] == 0 and len(d["candles"]) == d["hi"] - d["lo"]


def test_api_fills_filters_range(client):
    all_f = client.get("/api/fills").json()["fills"]
    assert all_f, "체결이 있어야 하는 시나리오"
    lo = all_f[0]["i"]
    narrowed = client.get(f"/api/fills?lo={lo+1}").json()["fills"]
    assert all(f["i"] > lo for f in narrowed)


def test_api_regime_bands_are_contiguous(client):
    bands = client.get("/api/regime-bands").json()["bands"]
    assert bands
    for a, b in zip(bands, bands[1:]):
        assert a["hi"] == b["lo"], "레짐 띠에 틈이 있으면 배경이 어긋난다"
        assert a["regime"] != b["regime"], "같은 레짐이 연속 띠로 쪼개졌다"


def test_api_leaderboard_bad_window_is_404(client):
    assert client.get("/api/leaderboard/99").status_code == 404


def test_api_leaderboard_covers_every_regime(client):
    d = client.get("/api/leaderboard/0").json()
    regimes = {r["regime"] for r in d["rows"]}
    assert regimes == {str(r) for r in Regime}


def test_live_trading_is_the_main_screen(client):
    """
    자동매매 앱을 켰는데 첫 화면이 백테스트면 지금 무슨 일이 일어나는지 알 수 없다.
    본 화면은 엔진 상태·포지션이다.
    """
    r = client.get("/")
    assert r.status_code == 200
    assert "자동매매" in r.text
    assert "포지션" in r.text and "리스크 가드" in r.text


def test_live_screen_warns_which_venue_it_is_pointed_at(client):
    """테스트넷인지 실전인지 화면에서 바로 보이지 않으면 사고가 난다."""
    r = client.get("/")
    assert "테스트넷" in r.text and "실전" in r.text


def test_review_screen_holds_the_backtest_tools(client):
    """백테스트는 '왜 그랬는지 돌아보는' 도구라 복기 쪽에 둔다."""
    r = client.get("/review")
    assert r.status_code == 200
    assert "복기" in r.text
    # 작업 목록은 API 로 채워지므로 정적 HTML 이 아니라 목록 자체를 본다
    names = {t["name"] for t in client.get("/api/home/tasks").json()["tasks"]}
    assert {"validate", "scoreboard", "signals"} <= names


def test_screens_link_to_each_other(client):
    for path in ("/", "/review"):
        text = client.get(path).text
        assert 'href="/"' in text and 'href="/review"' in text, f"{path} 에 내비게이션 없음"


def test_cockpit_moved_to_its_own_path(client):
    r = client.get("/cockpit")
    assert r.status_code == 200 and "레짐 퀀트 상황판" in r.text


def test_home_status_works_without_a_session(client):
    """
    홈은 walk-forward 를 돌리지 않는다. 세션 구성이 필요하면 데이터가 없을 때
    홈조차 열리지 않는다.
    """
    r = client.get("/api/home/status")
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"]["passed"] == 1 and body["verdict"]["total"] == 5
    assert body["verdict"]["order_execution"] is False
    assert "klines_dir" in body["data"]


def test_home_only_runs_whitelisted_tasks(client):
    """
    웹에서 임의 인자를 받아 실행하면 원격 코드 실행이 된다. 목록에 있는 것만 돈다.
    """
    r = client.post("/api/home/run/rm-rf")
    assert r.status_code == 404


def test_task_catalog_has_no_order_execution(client):
    r = client.get("/api/home/tasks")
    assert r.status_code == 200
    blob = " ".join(t["argv"][0] if isinstance(t.get("argv"), list) else ""
                    for t in r.json()["tasks"]).lower()
    assert "order" not in blob and "trade.py" not in blob


def test_task_log_strips_terminal_colour_codes():
    """
    pytest 는 파이프로 보내도 색상 코드를 넣는 경우가 있다. 그대로 브라우저에
    넘기면 깨진 글자로 보인다 — 실제로 홈 화면에서 그랬다.
    """
    from src.server.tasks import ANSI
    raw = "\x1b[32m.\x1b[0m\x1b[32m.\x1b[0m [ 12%]"
    assert ANSI.sub("", raw) == ".. [ 12%]"


def test_task_runner_allows_only_one_job_at_a_time():
    """두 개가 겹치면 둘 다 느려지고 사용자는 무엇이 멈춘 건지 알 수 없다."""
    from src.server.tasks import TaskRunner
    r = TaskRunner()
    ok1, _ = r.start("a", "A", ["-c", "import time; time.sleep(5)"])
    ok2, msg = r.start("b", "B", ["-c", "print(1)"])
    assert ok1 and not ok2 and "실행 중" in msg
    r.stop()



# ─── 자동매매 API ─────────────────────────────────────────────────

def test_live_status_works_without_credentials(client):
    """
    키가 없어도 화면은 열려야 한다. 열리지 않으면 "왜 안 되는지"를 볼 곳이 없다.
    대신 키가 없다는 사실과 어디에 넣는지를 상태에 담는다.
    """
    r = client.get("/api/live/status")
    assert r.status_code == 200
    body = r.json()
    assert body["running"] is False
    assert body["testnet"] is True, "기본은 테스트넷이어야 한다"
    assert "credentials" in body


def test_live_status_never_leaks_the_key(client, monkeypatch):
    """
    상태에 키 **값**이 섞이면 화면·로그·스크린샷으로 새어 나간다.
    필드 이름이 안내 문구에 나오는 것은 괜찮다 — 값이 나오면 안 된다.
    """
    import json
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "KEY-abc123-DO-NOT-LEAK")
    monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "SEC-xyz789-DO-NOT-LEAK")
    blob = json.dumps(client.get("/api/live/status").json(), ensure_ascii=False)
    assert "abc123" not in blob and "xyz789" not in blob
    # 키가 있다는 사실만 알린다
    assert client.get("/api/live/status").json()["credentials"]["present"] is True


def test_webhook_is_refused_without_a_configured_secret(client, monkeypatch):
    """
    비밀이 없으면 누구나 이 서버에 주문을 시킬 수 있다. 로컬 전용이어도 열지 않는다.
    """
    monkeypatch.delenv("REGIME_QUANT_WEBHOOK_SECRET", raising=False)
    r = client.post("/api/webhook/tradingview",
                    json={"symbol": "BTCUSDT", "action": "롱진입"})
    assert r.status_code == 503


def test_webhook_rejects_a_wrong_secret(client, monkeypatch):
    monkeypatch.setenv("REGIME_QUANT_WEBHOOK_SECRET", "right")
    r = client.post("/api/webhook/tradingview",
                    json={"secret": "wrong", "symbol": "BTCUSDT",
                          "action": "롱진입"})
    assert r.status_code == 401


def test_webhook_rejects_an_unknown_action(client, monkeypatch):
    monkeypatch.setenv("REGIME_QUANT_WEBHOOK_SECRET", "s3cret")
    r = client.post("/api/webhook/tradingview",
                    json={"secret": "s3cret", "symbol": "BTCUSDT",
                          "action": "전량매수"})
    assert r.status_code == 400


def test_webhook_ignores_any_price_the_alert_sends(client, monkeypatch):
    """
    알림이 보낸 가격을 그대로 믿으면 지연이나 오타가 그대로 주문이 된다.
    종목과 방향만 받는다.
    """
    from src.server import live_api
    monkeypatch.setenv("REGIME_QUANT_WEBHOOK_SECRET", "s3cret")
    seen = {}

    class FakeTrader:
        def handle_external(self, symbol, action, source="tradingview"):
            seen.update(symbol=symbol, action=action, source=source)
            return {"ok": True}

    monkeypatch.setattr(live_api, "trader", lambda: FakeTrader())
    r = client.post("/api/webhook/tradingview",
                    json={"secret": "s3cret", "symbol": "btcusdt",
                          "action": "롱진입", "price": 1.0, "qty": 999})
    assert r.status_code == 200
    assert seen == {"symbol": "BTCUSDT", "action": "롱진입", "source": "tradingview"}
