"""
호가창 중계 — 서버가 상태를 쌓는 부분.

**여기서 지키는 것은 "누적이 거짓말하지 않는가"다.**

    · m=true 는 "매수자가 메이커" → 공격자는 매도자. 뒤집으면 누적 델타가
      정반대로 쌓인다. 이게 틀리면 화면과 전략이 동시에 거짓이 된다.
    · 분기는 e 로 한다. 필드 존재로 판단하면 체결의 a(집계 체결 ID)를 호가
      배열로 오인한다 (브라우저 판에서 실제로 겪었다).
    · 눈금은 문자열로 계산한다. float 로 빼면 0.1 이 0.09999999999126885 가
      되고, 그 값으로 가격을 맞추면 같은 가격이 여러 칸으로 갈라진다.
    · 초기화는 누적만 지우고 호가창은 남긴다.
    · 메모리 상한이 실제로 걸린다. 무한히 도는 연결이라 상한이 없으면 며칠
      뒤 서버가 죽는다.

거래소에 붙는 부분(_reader)은 여기서 돌리지 않는다. 네트워크가 필요하고,
붙는 것 자체는 이미 실측으로 확인했다(@trade 정상, @aggTrade 는 0건).
"""

from __future__ import annotations

import pytest

from src.live import orderbook as ob


def depth(bids, asks):
    """MT5 아니라 바이낸스 부분 호가창 메시지 모양."""
    return {"e": "depthUpdate",
            "b": [[str(p), str(q)] for p, q in bids],
            "a": [[str(p), str(q)] for p, q in asks]}


def trade(px, qty, *, buyer_is_maker):
    return {"e": "trade", "p": str(px), "q": str(qty),
            "m": buyer_is_maker, "T": 1788700000000}


@pytest.fixture
def stream():
    return ob.Stream("btcusdt")


# ─── 공격자 방향 ─────────────────────────────────────────────────

def test_buyer_maker_false_means_the_buyer_attacked(stream):
    """
    **m=false 면 매수자가 테이커다** → 공격 매수 → 델타가 올라간다.
    """
    stream.apply_trade(trade(100.0, 2.0, buyer_is_maker=False))
    assert stream.cvd == pytest.approx(2.0)
    assert stream.buy_vol == pytest.approx(2.0)
    assert stream.sell_vol == 0.0


def test_buyer_maker_true_means_the_seller_attacked(stream):
    """
    m=true 는 "매수자가 메이커"다 — 즉 **공격자는 매도자**. 이 부호를 뒤집는
    것이 이 파일에서 가장 조용하고 치명적인 버그다.
    """
    stream.apply_trade(trade(100.0, 2.0, buyer_is_maker=True))
    assert stream.cvd == pytest.approx(-2.0)
    assert stream.sell_vol == pytest.approx(2.0)
    assert stream.buy_vol == 0.0


def test_delta_nets_out(stream):
    stream.apply_trade(trade(100.0, 5.0, buyer_is_maker=False))
    stream.apply_trade(trade(100.0, 3.0, buyer_is_maker=True))
    assert stream.cvd == pytest.approx(2.0)
    assert stream.n_trades == 2


def test_a_malformed_trade_is_ignored_not_counted(stream):
    """
    값이 깨진 메시지를 세면 체결 수와 델타가 조용히 어긋난다.
    """
    stream.apply_trade({"e": "trade", "p": "abc", "q": "1", "m": False})
    stream.apply_trade({"e": "trade", "m": False})
    assert stream.n_trades == 0 and stream.cvd == 0.0


# ─── 호가 눈금 (문자열로 계산해야 한다) ──────────────────────────

def test_tick_is_exact_not_a_float_artifact(stream):
    """
    거래소는 "80219.30" 처럼 정확한 십진수를 준다. float 로 빼면 0.1 이
    0.09999999999126885 로 나오고, 그 값으로 가격을 눈금에 맞추면 같은
    가격이 여러 칸으로 갈라진다.
    """
    stream.apply_depth(depth(
        [(80219.1, 1), (80219.0, 1), (80218.9, 1)],
        [(80219.2, 1), (80219.3, 1), (80219.4, 1)]))
    assert stream.tick == 0.1          # 0.09999999999... 이면 실패한다


def test_tick_shrinks_to_the_smallest_gap_seen(stream):
    """
    빈 레벨이 있으면 간격이 눈금의 배수로 보인다. 계속 최솟값으로 줄여야
    참값에 수렴한다.
    """
    stream.apply_depth(depth([(100.0, 1), (99.8, 1)], [(100.2, 1), (100.4, 1)]))
    assert stream.tick == pytest.approx(0.2)
    stream.apply_depth(depth([(100.0, 1), (99.9, 1)], [(100.1, 1), (100.2, 1)]))
    assert stream.tick == pytest.approx(0.1)


def test_footprint_buckets_land_on_clean_prices(stream):
    """눈금에 맞춘 키가 미세하게 갈라지면 같은 레벨이 두 칸으로 나뉜다."""
    stream.apply_depth(depth([(100.0, 1), (99.9, 1)], [(100.1, 1), (100.2, 1)]))
    for _ in range(3):
        stream.apply_trade(trade(100.1, 1.0, buyer_is_maker=False))
    assert list(stream.footprint) == [100.1]
    assert stream.footprint[100.1][0] == pytest.approx(3.0)


# ─── 풋프린트 (가격대별 델타) ────────────────────────────────────

def test_footprint_separates_buy_and_sell_at_the_same_price(stream):
    """
    **이것이 봉으로 뭉개면 사라지는 구조다.** 같은 가격에서 공격 매수와 공격
    매도를 따로 세야 어느 레벨에서 흡수됐는지가 보인다.
    """
    stream.apply_depth(depth([(100.0, 1), (99.9, 1)], [(100.1, 1), (100.2, 1)]))
    stream.apply_trade(trade(100.1, 4.0, buyer_is_maker=False))   # 공격 매수
    stream.apply_trade(trade(100.1, 1.5, buyer_is_maker=True))    # 공격 매도
    buy, sell = stream.footprint[100.1]
    assert buy == pytest.approx(4.0) and sell == pytest.approx(1.5)

    rows = stream.snapshot()["footprint"]
    row = next(r for r in rows if r["px"] == pytest.approx(100.1))
    assert row["delta"] == pytest.approx(2.5)


def test_footprint_is_capped_and_drops_the_farthest_levels(stream, monkeypatch):
    """
    무한히 도는 연결이라 상한이 없으면 며칠 뒤 서버가 죽는다. 버릴 때는
    **현재가에서 먼 눈금부터** 버려야 지금 보는 구간이 남는다.
    """
    monkeypatch.setattr(ob, "FOOT_MAX", 10)
    stream.apply_depth(depth([(100.0, 1), (99.9, 1)], [(100.1, 1), (100.2, 1)]))
    # 현재가(100.05) 에서 멀리까지 넓게 체결시킨다
    for i in range(-40, 41):
        stream.apply_trade(trade(round(100.0 + i * 0.1, 1), 1.0,
                                 buyer_is_maker=False))
    assert len(stream.footprint) <= 10
    mid = 100.05
    # 남은 것은 현재가 근처여야 한다
    assert max(abs(k - mid) for k in stream.footprint) < 1.0


def test_tape_is_capped(stream, monkeypatch):
    monkeypatch.setattr(ob, "TAPE_MAX", 5)
    s = ob.Stream("btcusdt")           # 상한은 생성 시점에 잡힌다
    for i in range(20):
        s.apply_trade(trade(100.0 + i, 1.0, buyer_is_maker=False))
    assert len(s.tape) == 5
    # 최신이 앞이다 — 화면이 최근 체결을 위에 보여 준다
    assert s.tape[0]["px"] == pytest.approx(119.0)


# ─── 호가창 갱신 ─────────────────────────────────────────────────

def test_depth_replaces_it_does_not_merge(stream):
    """
    부분 호가창(@depth20)은 스냅샷이다. 차분처럼 병합하면 사라진 레벨이
    영원히 남는다.
    """
    stream.apply_depth(depth([(100.0, 5)], [(100.1, 5)]))
    stream.apply_depth(depth([(99.0, 1)], [(101.0, 1)]))
    assert stream.bids == [["99.0", "1"]]
    assert stream.asks == [["101.0", "1"]]


def test_a_non_list_side_is_ignored(stream):
    """
    필드 존재로 분기하면 체결의 a(집계 체결 ID, 숫자)가 호가 배열로 들어와
    렌더링이 터진다. 브라우저 판에서 실제로 겪은 버그다.
    """
    stream.apply_depth(depth([(100.0, 5)], [(100.1, 5)]))
    stream.apply_depth({"e": "depthUpdate", "a": 123456, "b": None})
    assert stream.bids == [["100.0", "5"]]
    assert stream.asks == [["100.1", "5"]]


def test_levels_are_capped_to_twenty(stream):
    many = [(100.0 - i * 0.1, 1) for i in range(50)]
    ups = [(100.1 + i * 0.1, 1) for i in range(50)]
    stream.apply_depth(depth(many, ups))
    assert len(stream.bids) == ob.LEVELS and len(stream.asks) == ob.LEVELS


# ─── 초기화 ──────────────────────────────────────────────────────

def test_reset_clears_accumulation_but_keeps_the_book(stream):
    """
    호가창은 지금 시장 상태이고 누적은 우리가 센 것이다. 초기화가 호가창까지
    지우면 화면이 잠깐 빈다.
    """
    stream.apply_depth(depth([(100.0, 5)], [(100.1, 5)]))
    stream.apply_trade(trade(100.1, 3.0, buyer_is_maker=False))
    stream.reset()
    assert stream.cvd == 0.0 and stream.n_trades == 0
    assert stream.footprint == {} and len(stream.tape) == 0
    assert stream.bids and stream.asks        # 호가창은 남는다


# ─── 스냅샷 ──────────────────────────────────────────────────────

def test_snapshot_copies_so_the_reader_cannot_mutate_state(stream):
    """
    참조를 넘기면 거래소 스레드가 같은 리스트를 고치는 중에 직렬화되어 값이
    섞인다. 읽는 쪽이 실수로 고치는 것도 막아야 한다.
    """
    stream.apply_depth(depth([(100.0, 5)], [(100.1, 5)]))
    snap = stream.snapshot()
    snap["bids"].append(["1", "1"])
    snap["bids"][0][1] = "999"
    assert len(stream.bids) == 1 and stream.bids[0][1] == "5"


def test_snapshot_reports_staleness_so_the_screen_can_say_it(stream):
    """
    **끊긴 것을 조용히 넘기면 안 된다.** 화면은 마지막 호가를 그대로 들고
    있어서 살아 있는 것처럼 보인다.
    """
    assert stream.snapshot()["stale_s"] is None      # 아직 아무것도 안 받았다
    stream.apply_depth(depth([(100.0, 5)], [(100.1, 5)]))
    assert stream.snapshot()["stale_s"] >= 0.0


def test_footprint_rows_are_ordered_high_to_low(stream):
    """사다리와 같은 방향이어야 눈이 두 패널을 같이 읽는다."""
    stream.apply_depth(depth([(100.0, 1), (99.9, 1)], [(100.1, 1), (100.2, 1)]))
    for px in (99.9, 100.1, 100.0, 100.2):
        stream.apply_trade(trade(px, 1.0, buyer_is_maker=False))
    prices = [r["px"] for r in stream.snapshot()["footprint"]]
    assert prices == sorted(prices, reverse=True)


# ─── 종목 허용 목록 ──────────────────────────────────────────────

def test_an_unknown_symbol_is_refused():
    """
    문자열을 그대로 URL 에 넣으면 남이 임의의 스트림을 열게 만들 수 있고,
    오타 하나로 조용히 빈 화면이 된다.
    """
    with pytest.raises(ob.UnknownSymbol):
        ob.subscribe("btcusd")           # 코인 선물에는 없는 이름
    with pytest.raises(ob.UnknownSymbol):
        ob.subscribe("../../etc/passwd")


def test_get_does_not_start_a_stream():
    """읽기만 하는 경로가 연결을 열면 잊힌 연결이 늘어난다."""
    assert ob.get("linkusdt") is None
