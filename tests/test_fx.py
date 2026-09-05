"""
FX 계층 — 차트는 트레이딩뷰, 데이터·주문은 MT5.

여기서 고정하는 것은 **두 시스템을 잇는 이름 맞추기**다. 브로커는 EURUSDm,
EURUSD., EURUSD_raw 처럼 제각각 부르고 트레이딩뷰는 EURUSD 하나뿐이다.
이걸 못 맞추면 차트가 안 뜨거나 — 더 나쁘게 — **엉뚱한 종목에 주문이 나간다.**

그리고 이 앱은 FX 전용이다. 계좌에 금·지수가 섞여 있어도 그것을 FX 로 잘못
인식하면 안 된다.

MT5 는 터미널이 필요하므로 여기서는 부르지 않는다. 순수 로직만 검사한다.
"""

from __future__ import annotations

import pytest

from src.fx.symbols import (
    CROSSES, CURRENCIES, DEFAULT_PROVIDER, MAJORS, Pair, catalogue, parse,
    pip_size, resolve, to_pips,
)


# ─── 브로커 접미사 ────────────────────────────────────────────────

@pytest.mark.parametrize("broker_name", [
    "EURUSD", "EURUSDm", "EURUSD.", "EURUSD_raw", "EURUSDpro", "EURUSD-ECN",
])
def test_broker_suffixes_all_map_to_one_tradingview_symbol(broker_name):
    """
    브로커마다 접미사가 다른데 트레이딩뷰는 EURUSD 하나뿐이다.
    못 맞추면 차트가 안 뜬다.
    """
    p = parse(broker_name)
    assert p is not None, f"{broker_name} 을 FX 로 못 읽었다"
    assert p.canonical == "EURUSD"
    assert p.tradingview == "FX_IDC:EURUSD"
    # 주문은 브로커 이름으로 나가야 한다
    assert p.mt5 == broker_name


def test_order_uses_the_broker_name_not_the_canonical_one():
    """
    **엉뚱한 종목에 주문이 나가는 것을 막는다.** 표준 이름으로 주문하면
    브로커가 그런 심볼이 없다고 거부하거나, 더 나쁘면 다른 것을 잡는다.
    """
    p = resolve("EURUSD", ["EURUSDm", "GBPUSDm"])
    assert p.mt5 == "EURUSDm"
    assert p.canonical == "EURUSD"


def test_resolve_prefers_an_exact_match():
    p = resolve("EURUSD", ["EURUSDm", "EURUSD", "EURUSD."])
    assert p.mt5 == "EURUSD"


def test_resolve_returns_none_when_the_broker_lacks_the_pair():
    assert resolve("EURNOK", ["EURUSD", "GBPUSD"]) is None


# ─── FX 가 아닌 것을 걸러 내는가 ──────────────────────────────────

@pytest.mark.parametrize("not_fx", [
    "XAUUSD",      # 금 — USD 는 통화지만 XAU 는 아니다
    "US30", "US100Cash", "NAS100",
    "BTCUSD",      # 이 앱은 암호화폐를 다루지 않는다
    "USDUSD",      # 같은 통화
    "EUR", "EURUSDJPY",
])
def test_non_fx_symbols_are_rejected(not_fx):
    """계좌에 금·지수가 섞여 있어도 FX 로 잘못 인식하면 안 된다."""
    assert parse(not_fx) is None, f"{not_fx} 을 FX 로 잘못 읽었다"


def test_gold_is_not_fx_even_though_it_ends_in_usd():
    p = parse("XAUUSD")
    assert p is None
    assert "XAU" not in CURRENCIES


def test_catalogue_drops_duplicate_suffix_variants():
    """EURUSD, EURUSDm, EURUSD. 이 다 있으면 하나만 남아야 화면이 지저분해지지 않는다."""
    pairs = catalogue(["EURUSD", "EURUSDm", "EURUSD.", "GBPUSD"])
    assert [p.canonical for p in pairs] == ["EURUSD", "GBPUSD"]


def test_catalogue_puts_the_tightest_spread_pairs_first():
    """
    주요 통화쌍을 앞에 둔다. 스프레드가 좁아 비용/1R 이 낮기 때문이다 —
    이 프로젝트에서 전략을 죽인 것이 그 비율이었다.
    """
    pairs = catalogue(["EURGBP", "USDCHF", "EURUSD", "GBPJPY"])
    names = [p.canonical for p in pairs]
    assert names.index("EURUSD") < names.index("GBPJPY")
    assert set(MAJORS) & set(names)


def test_catalogue_can_limit_itself_to_liquid_pairs():
    pairs = catalogue(["EURUSD", "EURNOK", "EURTRY"], majors_only=True)
    assert [p.canonical for p in pairs] == ["EURUSD"]


# ─── 핍 계산 ──────────────────────────────────────────────────────

def test_jpy_pairs_use_a_different_pip_size():
    """엔화는 0.01 이 1핍, 나머지는 0.0001. 이걸 틀리면 손절폭이 100배 어긋난다."""
    eur = parse("EURUSD")
    jpy = parse("USDJPY")
    assert pip_size(eur, 5) == 0.0001
    assert pip_size(jpy, 3) == 0.01
    assert jpy.is_jpy and not eur.is_jpy


def test_price_delta_converts_to_pips():
    eur = parse("EURUSD")
    assert to_pips(eur, 0.0023, 5) == pytest.approx(23.0)
    jpy = parse("USDJPY")
    assert to_pips(jpy, 1.23, 3) == pytest.approx(123.0)


# ─── 제공자 ───────────────────────────────────────────────────────

def test_provider_can_be_switched():
    """브로커 호가와 가장 가까운 제공자를 고를 수 있어야 한다."""
    p = parse("EURUSD", provider="OANDA")
    assert p.tradingview == "OANDA:EURUSD"


def test_default_provider_is_the_widely_free_one():
    assert DEFAULT_PROVIDER == "FX_IDC"
    assert parse("EURUSD").tradingview.startswith("FX_IDC:")


# ─── 이 앱이 FX 전용인가 ──────────────────────────────────────────

def test_majors_and_crosses_are_all_real_currency_pairs():
    for name in MAJORS + CROSSES:
        p = parse(name)
        assert p is not None, f"{name} 이 통화쌍으로 안 읽힌다"
        assert p.base in CURRENCIES and p.quote in CURRENCIES


def test_pair_serialises_for_the_screen():
    d = parse("EURUSDm").as_dict()
    assert d["mt5"] == "EURUSDm"          # 주문용
    assert d["canonical"] == "EURUSD"
    assert d["tradingview"] == "FX_IDC:EURUSD"   # 차트용
    assert d["display"] == "EUR/USD"      # 사람용


# ─── MT5 모듈이 안전한가 (터미널 없이 확인 가능한 것만) ───────────

def test_mt5_calls_are_serialised():
    """
    MetaTrader5 패키지는 스레드 안전하지 않다. 웹 서버는 여러 스레드에서
    요청을 처리하므로 락 없이 부르면 값이 섞이거나 터미널이 멈춘다.
    """
    from src.fx import mt5 as mt5_mod
    import threading
    assert isinstance(mt5_mod._lock, type(threading.RLock()))


def test_bars_drops_the_in_progress_candle():
    """
    진행 중인 봉을 섞으면 신호가 매초 흔들리고 백테스트 규칙(마감봉 기준)과
    어긋난다. 소스에서 그 사실을 고정한다.
    """
    from pathlib import Path
    from src.fx import mt5 as mt5_mod
    src = Path(mt5_mod.__file__).read_text(encoding="utf-8")
    assert "iloc[:-1]" in src, "마지막(진행 중) 봉을 버리지 않는다"


def test_supported_timeframes_cover_what_the_app_uses():
    from src.fx.mt5 import TIMEFRAMES
    assert {"1h", "4h", "1d"} <= set(TIMEFRAMES)
