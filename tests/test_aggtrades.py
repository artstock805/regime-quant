"""
과거 풋프린트 복원.

**여기서 지키는 것은 "복원한 것이 실시간과 같은 것인가"다.**

    · is_buyer_maker=true 는 "매수자가 메이커" → 공격자는 매도자.
      실시간 스트림의 m 필드와 같은 의미여야 한다. 한쪽만 뒤집히면 과거와
      실시간이 서로 다른 것을 재게 되고, 검증 결과가 실거래를 보증하지 못한다.
    · 문자열 "false" 를 bool() 로 감싸면 True 다. 이 한 줄로 델타 전체가
      뒤집힌다.
    · 봉 라벨은 봉의 시작 시각 — 클라인 open_time 과 맞아야 한다.
      한 칸 밀리면 미래 참조가 된다.
    · 특징 정의를 바꿨는데 캐시 버전을 안 올리면 옛 값이 조용히 재사용된다.

**특징이 포화되지 않는지도 테스트한다.** 만들다가 두 번 겪었다 —
최대 |불균형| 은 항상 ±1 이었고(레벨이 700~1500개고 21% 가 한쪽만 체결되는데
그 레벨 거래량이 중위의 1/40), 상·하 델타를 |총델타| 로 나누니 ±7 까지 튀었다.
상수나 극단에 붙는 값으로 IC 를 재면 잡음만 나온다.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data import aggtrades as at


HEADER = ("agg_trade_id,price,quantity,first_trade_id,last_trade_id,"
          "transact_time,is_buyer_maker")

# 2026-08-15 00:00:00 UTC
T0 = 1786752000000


def zip_of(rows: list[str], *, header: bool = True) -> bytes:
    """아카이브 zip 한 장을 만든다."""
    lines = ([HEADER] if header else []) + rows
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("BTCUSDT-aggTrades-2026-08-15.csv", "\n".join(lines) + "\n")
    return buf.getvalue()


def row(i: int, px, qty, ms_offset: int, buyer_maker: str) -> str:
    return f"{i},{px},{qty},{i},{i},{T0 + ms_offset},{buyer_maker}"


@pytest.fixture
def cached(monkeypatch, tmp_path):
    """네트워크를 타지 않게 캐시 디렉터리를 갈아 끼운다."""
    monkeypatch.setattr(at, "cache_root", lambda: tmp_path)
    def put(content: bytes, symbol: str = "BTCUSDT",
            day: date = date(2026, 8, 15)) -> None:
        root = tmp_path / symbol
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{symbol}-aggTrades-{day:%Y-%m-%d}.zip").write_bytes(content)
    return put


# ─── 공격자 방향 ─────────────────────────────────────────────────

def test_buyer_maker_true_means_the_seller_attacked(cached):
    """
    **이 부호가 이 파일에서 가장 조용하고 치명적인 버그다.**
    is_buyer_maker=true 면 매수자가 메이커이므로 공격자는 매도자다.
    """
    cached(zip_of([row(1, "100.0", "2", 0, "true")]))
    raw = at.raw_day("BTCUSDT", date(2026, 8, 15))
    assert len(raw) == 1
    assert bool(raw["buy"].iloc[0]) is False


def test_buyer_maker_false_means_the_buyer_attacked(cached):
    cached(zip_of([row(1, "100.0", "2", 0, "false")]))
    raw = at.raw_day("BTCUSDT", date(2026, 8, 15))
    assert bool(raw["buy"].iloc[0]) is True


def test_the_string_false_is_not_read_as_true(cached):
    """
    `bool("false")` 는 True 다. 문자열을 그대로 감싸면 모든 체결이 공격 매수가
    되고 델타가 통째로 거짓이 된다.
    """
    cached(zip_of([row(1, "100.0", "1", 0, "false"),
                   row(2, "100.0", "1", 1, "true"),
                   row(3, "100.0", "1", 2, "False"),
                   row(4, "100.0", "1", 3, "True")]))
    raw = at.raw_day("BTCUSDT", date(2026, 8, 15))
    assert raw["buy"].tolist() == [True, False, True, False]


def test_delta_matches_a_hand_count(cached):
    cached(zip_of([row(1, "100.0", "5", 0, "false"),      # 공격 매수 +5
                   row(2, "100.0", "2", 1, "true")]))     # 공격 매도 -2
    f = at.day_features("BTCUSDT", date(2026, 8, 15), "1h")
    assert f["foot_delta"].sum() == pytest.approx(3.0)
    assert f["foot_vol"].sum() == pytest.approx(7.0)


# ─── 아카이브 형식 ───────────────────────────────────────────────

def test_a_file_without_a_header_row_is_read_too(cached):
    """아카이브에 헤더가 있는 날과 없는 날이 섞여 있다."""
    cached(zip_of([row(1, "100.0", "1", 0, "false")], header=False))
    raw = at.raw_day("BTCUSDT", date(2026, 8, 15))
    assert len(raw) == 1 and raw["px"].iloc[0] == pytest.approx(100.0)


def test_a_header_row_is_not_counted_as_a_trade(cached):
    cached(zip_of([row(1, "100.0", "1", 0, "false")], header=True))
    raw = at.raw_day("BTCUSDT", date(2026, 8, 15))
    assert len(raw) == 1


def test_a_missing_day_is_empty_not_an_error(monkeypatch, tmp_path):
    """
    아직 안 올라온 날은 오류가 아니라 정상적인 '없음'이다. 예외를 던지면
    180일 루프가 한 날 때문에 멈춘다.
    """
    monkeypatch.setattr(at, "cache_root", lambda: tmp_path)

    class Resp:
        status_code = 404
        def raise_for_status(self): raise AssertionError("불려선 안 된다")

    class Sess:
        def get(self, *a, **k): return Resp()

    out = at.raw_day("BTCUSDT", date(2026, 8, 15), session=Sess())
    assert out.empty


# ─── 호가 눈금 (문자열로 계산해야 한다) ──────────────────────────

def test_tick_is_exact_from_the_price_strings():
    """
    float 로 빼면 0.1 이 0.09999999999854481 로 나온다. 그 값으로 가격을 눈금에
    맞추면 같은 가격이 여러 칸으로 갈라진다.
    """
    px = pd.Series(["63014.9", "63015.0", "63015.1", "63015.2"])
    assert at.infer_tick(px) == 0.1


def test_tick_falls_back_to_floats_when_prices_are_numeric():
    px = pd.Series([100.0, 100.5, 101.0])
    assert at.infer_tick(px) == pytest.approx(0.5)


def test_footprint_levels_land_on_clean_prices(cached):
    """눈금에 맞춘 키가 갈라지면 같은 레벨이 두 칸으로 나뉜다."""
    rows = [row(i, f"{63014.9 + (i % 3) * 0.1:.1f}", "1", i, "false")
            for i in range(1, 31)]
    cached(zip_of(rows))
    f = at.day_features("BTCUSDT", date(2026, 8, 15), "1h")
    assert f["foot_levels"].iloc[0] == 3      # 갈라지면 3 보다 커진다
    assert f["foot_tick"].iloc[0] == 0.1


# ─── 봉 라벨 (미래 참조 방지) ────────────────────────────────────

def test_bar_label_is_the_bar_open(cached):
    """
    클라인 인덱스는 봉의 **시작 시각**이다. 마감 시각으로 라벨을 달면 특징이
    다음 봉에 붙어 미래 참조가 된다.
    """
    cached(zip_of([row(1, "100.0", "1", 0, "false"),               # 00:00
                   row(2, "100.0", "1", 59 * 60_000, "false")]))   # 00:59
    f = at.day_features("BTCUSDT", date(2026, 8, 15), "1h")
    assert len(f) == 1
    assert f.index[0] == pd.Timestamp("2026-08-15 00:00", tz="UTC")


def test_trades_go_into_the_hour_they_happened_in(cached):
    cached(zip_of([row(1, "100.0", "1", 0, "false"),
                   row(2, "100.0", "1", 3600_000, "false"),
                   row(3, "100.0", "1", 3660_000, "false")]))
    f = at.day_features("BTCUSDT", date(2026, 8, 15), "1h")
    assert list(f.index) == [pd.Timestamp("2026-08-15 00:00", tz="UTC"),
                             pd.Timestamp("2026-08-15 01:00", tz="UTC")]
    assert f["foot_vol"].tolist() == [1.0, 2.0]


# ─── 봉 안 위치 ──────────────────────────────────────────────────

def test_poc_position_is_zero_at_the_low_and_one_at_the_high(cached):
    """
    최다거래 레벨이 봉 어디에 있었나 — 0=저가, 1=고가.
    이게 뒤집히면 "고점에서 물량이 몰렸다"가 "저점"으로 읽힌다.
    """
    # 100.0 에 5회, 100.1~100.2 에 1회씩 → POC 는 최저가
    rows = [row(i, "100.0", "1", i, "false") for i in range(1, 6)]
    rows += [row(6, "100.1", "1", 6, "false"), row(7, "100.2", "1", 7, "false")]
    cached(zip_of(rows))
    f = at.day_features("BTCUSDT", date(2026, 8, 15), "1h")
    assert f["foot_poc_pos"].iloc[0] == pytest.approx(0.0)


def test_position_is_a_half_when_the_bar_had_one_price(cached):
    """
    범위가 0 이면 위치를 정의할 수 없다. 0 이나 1 로 두면 "저가에 붙었다"는
    거짓 신호가 된다.
    """
    cached(zip_of([row(1, "100.0", "1", 0, "false")]))
    f = at.day_features("BTCUSDT", date(2026, 8, 15), "1h")
    assert f["foot_poc_pos"].iloc[0] == pytest.approx(0.5)
    assert f["foot_imb_pos"].iloc[0] == pytest.approx(0.5)


# ─── 미완성 경매 (거래량으로 가중해야 한다) ──────────────────────

def test_unfinished_is_weighted_by_how_much_traded_there(cached):
    """
    처음엔 ±1 을 더했더니 거의 항상 0 이었다 — 고가와 저가가 둘 다 한쪽인
    경우가 흔해 상쇄됐다. 그런데 실측 봉에서 고가 물량은 1.856, 저가는
    0.006 이었다. 같은 "한쪽"이 아니다.
    """
    # 고가에 큰 공격매수만, 저가에 아주 작은 공격매도만
    rows = [row(1, "100.2", "10", 0, "false"),     # 고가: 매수만
            row(2, "100.1", "1", 1, "false"),
            row(3, "100.1", "1", 2, "true"),
            row(4, "100.0", "0.01", 3, "true")]    # 저가: 매도만 (아주 작음)
    cached(zip_of(rows))
    f = at.day_features("BTCUSDT", date(2026, 8, 15), "1h")
    v = f["foot_unfinished"].iloc[0]
    # 고가 쪽이 압도적이므로 확실히 양수여야 한다 (상쇄되면 0 근처)
    assert v > 0.5
    assert -1.0 <= v <= 1.0


# ─── 포화되지 않아야 한다 ────────────────────────────────────────

def test_imbalance_share_is_not_saturated(cached):
    """
    최댓값을 쓰면 항상 ±1 이 된다. 강하게 기운 레벨의 **거래량 비중**을
    세면 세기가 남는다.
    """
    # 절반은 한쪽으로, 절반은 균형
    rows = []
    i = 1
    for _ in range(5):
        rows.append(row(i, "100.0", "1", i, "false")); i += 1
    for _ in range(5):
        rows.append(row(i, "100.1", "1", i, "false")); i += 1
        rows.append(row(i, "100.1", "1", i, "true")); i += 1
    cached(zip_of(rows))
    f = at.day_features("BTCUSDT", date(2026, 8, 15), "1h")
    v = f["foot_imb_share"].iloc[0]
    assert 0.0 < v < 1.0          # ±1 에 붙지 않는다


def test_delta_skew_is_bounded_and_not_a_ratio_of_a_small_number(cached):
    """
    (상-하)/|총델타| 는 상·하가 크고 서로 반대일 때 ±7 까지 튀었다.
    각 절반을 그 절반의 거래량으로 나누면 -2~+2 안에 있다.
    """
    rows = [row(1, "100.2", "10", 0, "false"),     # 위: 공격매수
            row(2, "100.0", "10", 1, "true")]      # 아래: 공격매도
    cached(zip_of(rows))
    f = at.day_features("BTCUSDT", date(2026, 8, 15), "1h")
    v = f["foot_delta_skew"].iloc[0]
    assert v == pytest.approx(2.0)                 # 상한이 정확히 2 다
    assert -2.0 <= v <= 2.0


# ─── 캐시 ────────────────────────────────────────────────────────

def test_feature_cache_is_keyed_by_version(cached, monkeypatch, tmp_path):
    """
    특징 정의를 바꾸고 버전을 안 올리면 옛 값이 조용히 재사용되어, 고친 것이
    반영 안 된 채 결과를 해석하게 된다.
    """
    cached(zip_of([row(1, "100.0", "1", 0, "false")]))
    at.day_features("BTCUSDT", date(2026, 8, 15), "1h")
    made = list((tmp_path / "BTCUSDT").glob("feat-v*.parquet"))
    assert len(made) == 1
    assert f"-v{at.FEATURE_VERSION}-" in made[0].name

    monkeypatch.setattr(at, "FEATURE_VERSION", at.FEATURE_VERSION + 1)
    at.day_features("BTCUSDT", date(2026, 8, 15), "1h")
    assert len(list((tmp_path / "BTCUSDT").glob("feat-v*.parquet"))) == 2


def test_cache_goes_to_the_data_drive():
    """C 는 만성 부족이고 180일 한 종목이 2~4GB 다."""
    assert at.cache_root().name == "aggtrades"


def test_an_unknown_interval_fails_loudly(cached):
    cached(zip_of([row(1, "100.0", "1", 0, "false")]))
    with pytest.raises(at.ArchiveError):
        at.day_features("BTCUSDT", date(2026, 8, 15), "3h")
