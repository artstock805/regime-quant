"""
매매 내역과 잔고 현황.

**왜 MT5 내역이 진실인가.** 우리 장부(Journal)는 앱이 켜져 있던 동안만 안다.
손절이 새벽에 브로커에서 터졌거나 사용자가 MT5 창에서 직접 닫으면 장부에는
아무 줄도 없다. 그때 장부만 보여 주면 화면은 "거래 없음"이라고 말하는데
잔고는 줄어 있다 — 화면이 거짓말을 하는 것이다. 그래서 내역은 MT5 에서 읽고,
장부는 MT5 가 모르는 것(손절가·점수·국면)을 채우는 데만 쓴다.

여기서 지키는 것:
    · 진입 체결과 청산 체결이 **왕복 한 건**으로 묶인다
    · 아직 열려 있는 포지션은 내역에 안 나온다
    · 손익은 수수료·스왑까지 뺀 **순액** (계좌 잔고와 어긋나면 안 된다)
    · 우리 거래와 사람이 손으로 낸 거래가 **섞이지 않는다**
    · 입출금은 매매로 세지 않는다
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.fx import mt5 as m
from src.server import fx_api


NOW = datetime.now(timezone.utc)


def deal(pid, *, entry=0, kind=0, vol=0.10, price=1.1000, profit=0.0,
         comm=0.0, swap=0.0, magic=m.MAGIC, comment="rq-eurusd", minutes_ago=60,
         symbol="EURUSD", ticket=None):
    """MT5 가 돌려주는 거래 하나의 최소 형태."""
    at = NOW - timedelta(minutes=minutes_ago)
    return SimpleNamespace(
        ticket=ticket if ticket is not None else pid * 10 + entry,
        order=pid, position_id=pid, time=int(at.timestamp()), symbol=symbol,
        type=kind, entry=entry, volume=vol, price=price, profit=profit,
        commission=comm, swap=swap, fee=0.0, magic=magic, comment=comment)


@pytest.fixture
def fake_mt5(monkeypatch):
    """history_deals_get 을 우리가 정한 거래 목록으로 갈아 끼운다."""
    box: dict[str, list] = {"deals": []}
    monkeypatch.setattr(m, "initialise", lambda: True)
    monkeypatch.setattr(
        m, "_mt5",
        lambda: SimpleNamespace(history_deals_get=lambda a, b: tuple(box["deals"])))
    return box


# ─── 왕복 묶기 ───────────────────────────────────────────────────

def test_entry_and_exit_become_one_trade(fake_mt5):
    """체결은 두 줄로 오는데 사람이 보고 싶은 것은 들어가서 나온 한 건이다."""
    fake_mt5["deals"] = [
        deal(1, entry=0, kind=0, price=1.1000, minutes_ago=120),
        deal(1, entry=1, kind=1, price=1.1050, profit=50.0, minutes_ago=60),
    ]
    trades = m.closed_trades()
    assert len(trades) == 1
    t = trades[0]
    assert t["side"] == "LONG"
    assert t["entry_price"] == pytest.approx(1.1000)
    assert t["exit_price"] == pytest.approx(1.1050)
    assert t["held_minutes"] == pytest.approx(60.0)


def test_an_open_position_is_not_history(fake_mt5):
    """청산 체결이 없으면 아직 들고 있는 것이다 — 내역이 아니다."""
    fake_mt5["deals"] = [deal(1, entry=0)]
    assert m.closed_trades() == []


def test_a_short_is_read_from_the_entry_deal(fake_mt5):
    """청산 체결은 방향이 반대다. 그걸로 방향을 읽으면 롱숏이 전부 뒤집힌다."""
    fake_mt5["deals"] = [
        deal(1, entry=0, kind=1, price=1.1050, minutes_ago=120),   # 매도로 진입
        deal(1, entry=1, kind=0, price=1.1000, profit=50.0, minutes_ago=60),
    ]
    assert m.closed_trades()[0]["side"] == "SHORT"


def test_partial_exits_are_volume_weighted(fake_mt5):
    """나눠서 청산했으면 가중평균이어야 한다. 단순평균은 수량을 무시한다."""
    fake_mt5["deals"] = [
        deal(1, entry=0, vol=1.00, price=1.1000, minutes_ago=120),
        deal(1, entry=1, vol=0.75, price=1.1100, profit=75.0, minutes_ago=60),
        deal(1, entry=1, vol=0.25, price=1.1500, profit=25.0, minutes_ago=30),
    ]
    t = m.closed_trades()[0]
    # 0.75*1.11 + 0.25*1.15 = 1.12  (단순평균이면 1.13)
    assert t["exit_price"] == pytest.approx(1.12)


# ─── 손익은 순액이어야 한다 ──────────────────────────────────────

def test_profit_includes_commission_and_swap(fake_mt5):
    """수수료를 빼지 않으면 화면 숫자가 계좌 잔고와 어긋난다."""
    fake_mt5["deals"] = [
        deal(1, entry=0, comm=-3.0, minutes_ago=120),
        deal(1, entry=1, kind=1, profit=50.0, comm=-3.0, swap=-1.5, minutes_ago=60),
    ]
    t = m.closed_trades()[0]
    assert t["gross"] == pytest.approx(50.0)
    assert t["profit"] == pytest.approx(42.5)      # 50 - 3 - 3 - 1.5
    assert t["commission"] == pytest.approx(-6.0)
    assert t["swap"] == pytest.approx(-1.5)


# ─── 우리 거래와 사람 거래 ───────────────────────────────────────

def test_our_trades_are_marked_by_magic(fake_mt5):
    fake_mt5["deals"] = [
        deal(1, entry=0, magic=m.MAGIC, comment=""),
        deal(1, entry=1, kind=1, magic=m.MAGIC, comment="", profit=1.0),
    ]
    assert m.closed_trades()[0]["ours"] is True


def test_a_hand_placed_trade_is_not_ours(fake_mt5):
    """
    계좌에는 사람이 손으로 연 금·지수 거래가 섞여 있다. 그것을 엔진 성적으로
    세면 엔진이 벌었는지 사람이 벌었는지 알 수 없게 된다.
    """
    fake_mt5["deals"] = [
        deal(1, entry=0, magic=0, comment="", symbol="GOLD"),
        deal(1, entry=1, kind=1, magic=0, comment="", profit=9.0, symbol="GOLD"),
    ]
    assert m.closed_trades()[0]["ours"] is False


def test_the_tag_prefix_also_marks_our_trade(fake_mt5):
    """일부 브로커는 magic 을 안 실어 준다. 태그 접두사가 두 번째 단서다."""
    fake_mt5["deals"] = [
        deal(1, entry=0, magic=0, comment=m.PREFIX + "eurusd0601"),
        deal(1, entry=1, kind=1, magic=0, comment="", profit=1.0),
    ]
    assert m.closed_trades()[0]["ours"] is True


def test_magic_is_defined_once():
    """
    주문을 내는 쪽과 내역을 읽는 쪽이 같은 번호를 봐야 한다. 양쪽에 따로
    적어 두면 한쪽만 바뀌었을 때 과거 거래가 조용히 남의 것이 된다.
    """
    from src.fx import broker
    assert broker.MAGIC is m.MAGIC
    assert broker.PREFIX is m.PREFIX


# ─── 입출금 ──────────────────────────────────────────────────────

def test_deposits_are_not_trades(fake_mt5):
    """입금은 매매가 아니다. 이걸 섞으면 성적이 입금액만큼 좋아 보인다."""
    fake_mt5["deals"] = [
        deal(0, entry=0, kind=2, profit=50_000.0, symbol="", comment="deposit"),
        deal(1, entry=0, minutes_ago=120),
        deal(1, entry=1, kind=1, profit=10.0, minutes_ago=60),
    ]
    trades = m.closed_trades()
    assert len(trades) == 1
    assert trades[0]["profit"] == pytest.approx(10.0)
    assert m.cash_flows()["deposit"] == pytest.approx(50_000.0)


def test_withdrawals_are_netted(fake_mt5):
    fake_mt5["deals"] = [
        deal(0, entry=0, kind=2, profit=50_000.0, symbol=""),
        deal(0, entry=0, kind=2, profit=-2_000.0, symbol="", ticket=99),
    ]
    f = m.cash_flows()
    assert f["deposit"] == pytest.approx(50_000.0)
    assert f["withdrawal"] == pytest.approx(-2_000.0)
    assert f["net_funding"] == pytest.approx(48_000.0)


# ─── 조회 구간 ───────────────────────────────────────────────────

def test_deals_outside_the_window_are_dropped(fake_mt5):
    """
    MT5 는 서버 시간(XM 은 GMT+2/+3)으로 구간을 해석한다. 그래서 앞뒤로
    하루씩 넉넉히 요청하는데, 받은 뒤에 잘라내지 않으면 기간 밖 거래가
    섞여 들어온다.
    """
    fake_mt5["deals"] = [
        deal(1, entry=0, minutes_ago=60 * 24 * 40),      # 40일 전
        deal(1, entry=1, kind=1, minutes_ago=60 * 24 * 39, profit=5.0),
        deal(2, entry=0, minutes_ago=120),
        deal(2, entry=1, kind=1, minutes_ago=60, profit=7.0),
    ]
    recent = m.deals(days=30)
    assert {d["position_id"] for d in recent} == {2}


# ─── R배수 ───────────────────────────────────────────────────────

def test_r_multiple_uses_the_stop_we_entered_with():
    """
    R배수는 **가격으로** 잰다. 랏·계약크기·환율이 끼면 같은 판단이 종목마다
    다른 숫자로 보여서 비교가 안 된다.
    """
    t = {"entry_price": 1.1000, "exit_price": 1.1150, "side": "LONG"}
    order = {"price": 1.1000, "stop_loss": 1.0950}      # 1R = 50핍
    assert fx_api._r_multiple(t, order) == pytest.approx(3.0)


def test_r_multiple_is_negative_when_the_stop_was_hit():
    t = {"entry_price": 1.1000, "exit_price": 1.0950, "side": "LONG"}
    order = {"price": 1.1000, "stop_loss": 1.0950}
    assert fx_api._r_multiple(t, order) == pytest.approx(-1.0)


def test_r_multiple_flips_for_a_short():
    """숏은 내려가야 이긴다. 부호를 안 뒤집으면 성적이 정반대로 보인다."""
    t = {"entry_price": 1.1000, "exit_price": 1.0900, "side": "SHORT"}
    order = {"price": 1.1000, "stop_loss": 1.1050}      # 1R = 50핍
    assert fx_api._r_multiple(t, order) == pytest.approx(2.0)


def test_no_r_multiple_without_a_stop():
    """
    손절가를 모르면 1R 이 정의되지 않는다. **0 이나 추정값을 쓰면 안 된다** —
    사람이 그 숫자를 성적으로 읽는다. 모른다고 말하는 게 맞다.
    """
    t = {"entry_price": 1.1000, "exit_price": 1.1150, "side": "LONG"}
    assert fx_api._r_multiple(t, None) is None
    assert fx_api._r_multiple(t, {"price": 1.1, "stop_loss": 0}) is None
    assert fx_api._r_multiple(t, {"price": 1.1, "stop_loss": 1.1}) is None


# ─── 요약 ────────────────────────────────────────────────────────

def test_summary_keeps_engine_and_manual_apart():
    trades = [
        {"ours": True,  "profit": 10.0, "commission": -1.0, "swap": 0.0,
         "r_multiple": 1.0, "closed_at": "2026-09-05T10:00:00+00:00"},
        {"ours": False, "profit": -50.0, "commission": -2.0, "swap": -1.0,
         "r_multiple": None, "closed_at": "2026-09-05T11:00:00+00:00"},
    ]
    s = fx_api._history_summary(trades)
    assert s["ours"]["trades"] == 1 and s["ours"]["net"] == pytest.approx(10.0)
    assert s["manual"]["trades"] == 1 and s["manual"]["net"] == pytest.approx(-50.0)
    assert s["all"]["trades"] == 2
    # 엔진 것만 R 이 있다 — 사람 거래엔 손절 기록이 없다
    assert s["ours"]["avg_r"] == pytest.approx(1.0)
    assert s["manual"]["avg_r"] is None
    assert s["manual"]["r_known"] == 0


def test_a_high_win_rate_does_not_hide_a_loss():
    """
    작게 여러 번 이기고 크게 한 번 지는 전략은 승률이 높은데 돈을 잃는다.
    이 프로젝트에서 이미 겪었다(평균회귀 승률 54.8%, 기대R 음수).
    그래서 승률과 순손익을 **같이** 내보낸다.
    """
    trades = [{"ours": True, "profit": 1.0, "commission": 0.0, "swap": 0.0,
               "r_multiple": None, "closed_at": "2026-09-05T10:00:00+00:00"}
              for _ in range(9)]
    trades.append({"ours": True, "profit": -100.0, "commission": 0.0,
                   "swap": 0.0, "r_multiple": None,
                   "closed_at": "2026-09-05T10:00:00+00:00"})
    b = fx_api._history_summary(trades)["ours"]
    assert b["win_rate"] == pytest.approx(0.9)
    assert b["net"] == pytest.approx(-91.0)      # 승률 90% 인데 손실
    assert b["worst"] == pytest.approx(-100.0)


def test_empty_history_does_not_crash():
    s = fx_api._history_summary([])
    assert s["all"]["trades"] == 0
    assert s["all"]["win_rate"] is None
    assert s["realised_today"] == 0


# ─── 누적 R 곡선 ─────────────────────────────────────────────────

def test_r_curve_accumulates_in_time_order():
    """
    **금액 곡선보다 R 곡선이 전략 평가에 맞다.** 금액은 랏 크기에 따라 달라져서
    같은 판단을 크게 걸었는지 작게 걸었는지가 곡선 모양을 바꾼다. R 은 랏과
    무관하게 판단의 질만 남는다.
    """
    trades = [
        {"r_multiple": 2.0, "closed_at": "2026-09-03T00:00:00+00:00",
         "display": "EUR/USD"},
        {"r_multiple": -1.0, "closed_at": "2026-09-01T00:00:00+00:00",
         "display": "GBP/USD"},
        {"r_multiple": 0.5, "closed_at": "2026-09-02T00:00:00+00:00",
         "display": "USD/JPY"},
    ]
    curve = fx_api._r_curve(trades)
    assert [p["cum_r"] for p in curve] == [-1.0, -0.5, 1.5]
    assert [p["symbol"] for p in curve] == ["GBP/USD", "USD/JPY", "EUR/USD"]


def test_r_curve_skips_trades_without_a_known_stop():
    """
    손절가를 모르면 1R 이 정의되지 않는다. 0 으로 넣으면 곡선이 평평해져서
    "본전이었다"로 읽힌다 — 실제로는 모르는 것이다.
    """
    trades = [
        {"r_multiple": 1.5, "closed_at": "2026-09-01T00:00:00+00:00",
         "display": "EUR/USD"},
        {"r_multiple": None, "closed_at": "2026-09-02T00:00:00+00:00",
         "display": "GOLD"},
    ]
    curve = fx_api._r_curve(trades)
    assert len(curve) == 1
    assert curve[0]["cum_r"] == pytest.approx(1.5)


def test_r_curve_is_engine_only_and_reports_what_it_dropped():
    """
    사람이 손으로 낸 거래는 우리 장부가 없어 손절가를 모른다. 그것을 섞으면
    곡선이 거짓말을 한다. 대신 몇 건이 빠졌는지 같이 내보낸다.
    """
    trades = [
        {"ours": True, "profit": 10.0, "commission": 0.0, "swap": 0.0,
         "r_multiple": 2.0, "closed_at": "2026-09-01T00:00:00+00:00",
         "display": "EUR/USD"},
        {"ours": True, "profit": 5.0, "commission": 0.0, "swap": 0.0,
         "r_multiple": None, "closed_at": "2026-09-02T00:00:00+00:00",
         "display": "EUR/USD"},
        {"ours": False, "profit": -99.0, "commission": 0.0, "swap": 0.0,
         "r_multiple": 9.9, "closed_at": "2026-09-03T00:00:00+00:00",
         "display": "GOLD"},
    ]
    s = fx_api._history_summary(trades)
    assert [p["cum_r"] for p in s["r_curve"]] == [2.0]     # 수동 9.9 는 안 들어간다
    assert s["r_missing"] == 1


def test_empty_r_curve_is_empty_not_zero():
    """
    거래가 없을 때 0 짜리 점을 하나 넣으면 화면이 "본전"을 그린다.
    없으면 없다고 해야 한다.
    """
    assert fx_api._r_curve([]) == []
    assert fx_api._history_summary([])["r_curve"] == []


# ─── CSV 내보내기 ────────────────────────────────────────────────

def _csv_of(monkeypatch, trades, **kw):
    monkeypatch.setattr(fx_api, "_enriched", lambda days: (trades, {}))
    return fx_api.fx_history_csv(**kw)


def test_csv_starts_with_a_bom(monkeypatch):
    """
    **엑셀은 BOM 없는 UTF-8 을 ANSI(949)로 읽어 한글을 전부 깨뜨린다.**
    내보낸 파일을 열었더니 글자가 깨져 있으면 그 기능은 없는 것과 같다.
    """
    res = _csv_of(monkeypatch, [])
    assert res.body.startswith(b"\xef\xbb\xbf")
    assert "charset=utf-8" in res.media_type


def test_csv_header_is_korean_and_readable(monkeypatch):
    res = _csv_of(monkeypatch, [])
    head = res.body.decode("utf-8-sig").splitlines()[0]
    assert head.startswith("청산시각,진입시각,종목")
    assert "R배수" in head and "출처" in head


def test_csv_marks_engine_and_manual(monkeypatch):
    trades = [
        {"ours": True, "display": "EUR/USD", "profit": 1.0, "position_id": 1},
        {"ours": False, "display": "GOLD", "profit": -2.0, "position_id": 2},
    ]
    body = _csv_of(monkeypatch, trades).body.decode("utf-8-sig")
    assert "엔진" in body and "수동" in body


def test_csv_quotes_values_that_would_break_columns(monkeypatch):
    """쉼표가 든 값을 그냥 쓰면 칸이 밀려서 파일 전체가 어긋난다."""
    trades = [{"ours": True, "display": "EUR/USD", "regime": "횡보, 고변동",
               "position_id": 1}]
    line = _csv_of(monkeypatch, trades).body.decode("utf-8-sig").splitlines()[1]
    assert '"횡보, 고변동"' in line

    import csv, io
    rows = list(csv.DictReader(io.StringIO(
        _csv_of(monkeypatch, trades).body.decode("utf-8-sig"))))
    assert rows[0]["국면"] == "횡보, 고변동"       # 되읽어도 한 칸이다


def test_csv_filename_says_whether_it_is_engine_only(monkeypatch):
    """파일이 여러 개 쌓였을 때 이름만 보고 무엇인지 알아야 한다."""
    plain = _csv_of(monkeypatch, [], only_ours=False)
    mine = _csv_of(monkeypatch, [], only_ours=True)
    assert "engine" not in plain.headers["content-disposition"]
    assert "engine" in mine.headers["content-disposition"]
    assert "attachment" in plain.headers["content-disposition"]


def test_csv_only_ours_filters_rows(monkeypatch):
    trades = [
        {"ours": True, "display": "EUR/USD", "position_id": 1},
        {"ours": False, "display": "GOLD", "position_id": 2},
    ]
    body = _csv_of(monkeypatch, trades, only_ours=True).body.decode("utf-8-sig")
    assert "EUR/USD" in body and "GOLD" not in body
