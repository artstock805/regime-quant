"""
백테스트 엔진 검증.

손으로 계산할 수 있는 작은 시나리오를 만들어 체결 순서·봉 내부 선착 판정·비용
회계를 확인한다. 이 엔진이 틀리면 뒤의 모든 숫자가 무의미하므로 여기가 제일 촘촘하다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Costs
from src.backtest.engine import ExecParams, run

FREE = Costs(fee_rate=0.0, slippage_bps=0.0)


def bars(rows: list[tuple[float, float, float, float]], freq: str = "5min") -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(rows), freq=freq, tz="UTC", name="open_time")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(
        volume=1.0, quote_volume=1.0, trades=1, taker_buy_base=0.5
    )


def minutes(rows: list[tuple[float, float, float, float]], start: str = "2026-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(rows), freq="1min", tz="UTC", name="open_time")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def sigs(directions: list[float], index: pd.DatetimeIndex, strength: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {"direction": directions, "strength": [strength if d else 0.0 for d in directions]},
        index=index,
    )


def const_atr(index: pd.DatetimeIndex, value: float) -> pd.Series:
    return pd.Series(value, index=index, name="atr")


P = ExecParams(risk_per_trade=0.01, atr_stop_mult=1.0, reward_ratio=2.0, trailing=False)


# ─── 체결 타이밍 ──────────────────────────────────────────────────

def test_entry_fills_at_next_bar_open_not_signal_bar_close():
    """
    신호는 봉 1의 종가에서 나온다. 체결은 봉 2의 시가여야 한다.
    봉 1 종가로 체결하면 종가를 알기 전에 주문을 낸 셈이 되어 미래 참조다.
    """
    df = bars([(100, 101, 99, 100), (100, 101, 99, 100.5), (103, 104, 102.5, 103), (103, 103.5, 102, 103)])
    r = run(df, sigs([0, 1, 0, 0], df.index), const_atr(df.index, 2.0),
            interval="5m", costs=FREE, params=P, initial_equity=10_000)

    assert len(r.trades) == 1
    t = r.trades.iloc[0]
    assert t["entry_time"] == df.index[2], "봉 2 시가에 체결돼야 한다"
    assert t["entry_price"] == pytest.approx(103.0), "봉 1 종가(100.5)가 아니라 봉 2 시가(103)"


def test_signal_exit_fills_at_next_bar_open():
    df = bars([(100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99.5, 100), (105, 106, 104, 105)])
    r = run(df, sigs([0, 1, 0, 0], df.index), const_atr(df.index, 5.0),
            interval="5m", costs=FREE, params=P, initial_equity=10_000)
    t = r.trades.iloc[0]
    assert t["exit_reason"] == "signal"
    assert t["exit_time"] == df.index[3]
    assert t["exit_price"] == pytest.approx(105.0)


# ─── 봉 내부 선착 판정 ────────────────────────────────────────────

def _scenario(minute_rows: list[tuple[float, float, float, float]]):
    """봉 2에서 롱 진입(시가 100, ATR 2 → 손절 98 / 목표 104), 봉 2의 1분봉을 지정."""
    df = bars([(100, 100, 100, 100), (100, 100, 100, 100),
               (100, 106, 96, 100), (100, 100, 100, 100)])
    # 봉 2는 00:10 시작 → 그 5분간의 1분봉
    mins = pd.concat([
        minutes([(100, 100, 100, 100)] * 10, start="2026-01-01 00:00"),
        minutes(minute_rows, start="2026-01-01 00:10"),
        minutes([(100, 100, 100, 100)] * 5, start="2026-01-01 00:15"),
    ])
    return run(df, sigs([0, 1, 1, 1], df.index), const_atr(df.index, 2.0),
               interval="5m", costs=FREE, params=P, initial_equity=10_000,
               minute_df=mins)


def test_target_hit_first_is_recorded_as_target():
    r = _scenario([
        (100, 101, 99.5, 100),
        (100, 105, 99.5, 104),   # 목표 104 도달
        (100, 100, 96, 97),      # 그 뒤에 손절권
        (100, 100, 100, 100),
        (100, 100, 100, 100),
    ])
    t = r.trades.iloc[0]
    assert t["exit_reason"] == "target"
    assert t["exit_price"] == pytest.approx(104.0)


def test_stop_hit_first_is_recorded_as_stop():
    r = _scenario([
        (100, 101, 99.5, 100),
        (100, 100, 97, 98),      # 손절 98 도달
        (100, 106, 100, 105),    # 그 뒤에 목표권
        (100, 100, 100, 100),
        (100, 100, 100, 100),
    ])
    t = r.trades.iloc[0]
    assert t["exit_reason"] == "stop"
    assert t["exit_price"] == pytest.approx(98.0)


def test_both_hit_in_same_minute_resolves_to_stop():
    """
    한 분봉 안에서 손절과 목표가 모두 닿으면 순서를 알 수 없다.
    유리한 쪽을 고르면 결과가 부풀려지므로 손절을 택해야 한다.
    """
    r = _scenario([
        (100, 106, 96, 100),     # 96~106 → 손절 98과 목표 104 모두 포함
        (100, 100, 100, 100),
        (100, 100, 100, 100),
        (100, 100, 100, 100),
        (100, 100, 100, 100),
    ])
    t = r.trades.iloc[0]
    assert t["exit_reason"] == "stop", "같은 분봉에서 둘 다 닿으면 보수적으로 손절"


def test_without_minute_data_only_stop_is_evaluated():
    """
    1분봉이 없으면 목표 도달을 인정하지 않는다. 어느 쪽이 먼저였는지 모르는 상태에서
    목표를 인정하면 손익비 2:1 전략의 결과가 두 배로 부풀려진다.
    """
    df = bars([(100, 100, 100, 100), (100, 100, 100, 100),
               (100, 106, 99, 100), (100, 100, 100, 100)])
    r = run(df, sigs([0, 1, 1, 1], df.index), const_atr(df.index, 2.0),
            interval="5m", costs=FREE, params=P, initial_equity=10_000)
    # 고가 106 은 목표 104 를 넘겼지만 1분봉이 없어 인정하지 않는다
    assert (r.trades["exit_reason"] != "target").all()


def test_minute_resolution_changes_the_answer():
    """1분봉을 주는 것과 안 주는 것이 실제로 다른 결과를 낸다 — 판정이 형식이 아니다."""
    df = bars([(100, 100, 100, 100), (100, 100, 100, 100),
               (100, 106, 99.5, 100), (100, 100, 100, 100)])
    mins = pd.concat([
        minutes([(100, 100, 100, 100)] * 10, start="2026-01-01 00:00"),
        minutes([(100, 105, 100, 104), (100, 100, 100, 100), (100, 100, 100, 100),
                 (100, 100, 100, 100), (100, 100, 100, 100)], start="2026-01-01 00:10"),
        minutes([(100, 100, 100, 100)] * 5, start="2026-01-01 00:15"),
    ])
    with_min = run(df, sigs([0, 1, 1, 1], df.index), const_atr(df.index, 2.0),
                   interval="5m", costs=FREE, params=P, initial_equity=10_000, minute_df=mins)
    without = run(df, sigs([0, 1, 1, 1], df.index), const_atr(df.index, 2.0),
                  interval="5m", costs=FREE, params=P, initial_equity=10_000)
    assert with_min.trades.iloc[0]["exit_reason"] == "target"
    assert (without.trades["exit_reason"] != "target").all()


# ─── 비용 회계 ────────────────────────────────────────────────────

def test_fees_charged_on_both_entry_and_exit():
    costs = Costs(fee_rate=0.001, slippage_bps=0.0)
    df = bars([(100, 100, 100, 100), (100, 100, 100, 100),
               (100, 100, 100, 100), (100, 100, 100, 100)])
    r = run(df, sigs([0, 1, 0, 0], df.index), const_atr(df.index, 2.0),
            interval="5m", costs=costs, params=P, initial_equity=10_000)
    t = r.trades.iloc[0]
    # 진입가 = 청산가 = 100 이므로 총손익 0, 순손익은 왕복 수수료만큼 음수
    assert t["gross_pnl"] == pytest.approx(0.0)
    expected = 2 * t["notional"] * 0.001
    assert t["fees"] == pytest.approx(expected)
    assert t["net_pnl"] == pytest.approx(-expected)


def test_slippage_is_always_adverse():
    costs = Costs(fee_rate=0.0, slippage_bps=10.0)   # 0.1%
    df = bars([(100, 100, 100, 100)] * 4)
    long_r = run(df, sigs([0, 1, 0, 0], df.index), const_atr(df.index, 2.0),
                 interval="5m", costs=costs, params=P, initial_equity=10_000)
    short_r = run(df, sigs([0, -1, 0, 0], df.index), const_atr(df.index, 2.0),
                  interval="5m", costs=costs, params=P, initial_equity=10_000)

    lt, st = long_r.trades.iloc[0], short_r.trades.iloc[0]
    assert lt["entry_price"] > 100.0, "롱은 비싸게 산다"
    assert lt["exit_price"] < 100.0, "롱은 싸게 판다"
    assert st["entry_price"] < 100.0, "숏은 싸게 판다"
    assert st["exit_price"] > 100.0, "숏은 비싸게 산다"
    assert lt["net_pnl"] < 0 and st["net_pnl"] < 0, "왕복 슬리피지만으로 손실이어야 한다"


def test_position_size_makes_stop_loss_equal_risk_budget():
    """손절에 닿았을 때의 손실이 계좌의 risk_per_trade 에 맞아야 한다."""
    df = bars([(100, 100, 100, 100), (100, 100, 100, 100),
               (100, 100, 97, 98), (100, 100, 100, 100)])
    r = run(df, sigs([0, 1, 1, 1], df.index), const_atr(df.index, 2.0),
            interval="5m", costs=FREE, params=P, initial_equity=10_000)
    t = r.trades.iloc[0]
    assert t["exit_reason"] == "stop"
    assert t["net_pnl"] == pytest.approx(-100.0, rel=1e-6), "1% 리스크 = $100"
    assert t["r_multiple"] == pytest.approx(-1.0, rel=1e-6)


def test_leverage_cap_limits_size_in_low_volatility():
    """ATR이 아주 작으면 사이즈가 무한히 커지므로 상한이 걸려야 한다."""
    df = bars([(100, 100, 100, 100)] * 4)
    p = ExecParams(risk_per_trade=0.01, atr_stop_mult=1.0, max_leverage=2.0, trailing=False)
    r = run(df, sigs([0, 1, 0, 0], df.index), const_atr(df.index, 0.001),
            interval="5m", costs=FREE, params=p, initial_equity=10_000)
    t = r.trades.iloc[0]
    assert t["notional"] <= 10_000 * 2.0 + 1e-6


def test_equity_curve_matches_sum_of_trades():
    rng = np.random.default_rng(0)
    n = 300
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    rows = [(c, c + 0.6, c - 0.6, c) for c in close]
    df = bars(rows)
    d = np.where(rng.random(n) > 0.7, np.sign(rng.normal(size=n)), 0.0)
    r = run(df, sigs(list(d), df.index), const_atr(df.index, 1.0),
            interval="5m", costs=Costs(0.0004, 1.5), params=P, initial_equity=10_000)

    assert len(r.trades) > 5
    expected_final = 10_000 + r.trades["net_pnl"].sum()
    assert r.equity.iloc[-1] == pytest.approx(expected_final, rel=1e-9), (
        "자산곡선 종점이 거래 순손익 합과 맞지 않는다 — 회계 누락"
    )


def test_open_position_is_closed_at_end():
    """마지막에 열려 있는 포지션을 성과에서 빼면 손실 중인 포지션을 숨기는 셈이다."""
    df = bars([(100, 100, 100, 100)] * 5)
    r = run(df, sigs([0, 1, 1, 1, 1], df.index), const_atr(df.index, 2.0),
            interval="5m", costs=FREE, params=P, initial_equity=10_000)
    assert len(r.trades) == 1
    assert r.trades.iloc[0]["exit_reason"] == "end"


# ─── 미래 참조 ────────────────────────────────────────────────────

def test_engine_is_causal():
    """
    데이터를 뒤에서 잘라도 그 전에 끝난 거래는 동일해야 한다.
    엔진이 뒤쪽 봉을 참조하면 잘랐을 때 결과가 달라진다.
    """
    rng = np.random.default_rng(7)
    n = 400
    close = 100 + np.cumsum(rng.normal(0, 0.4, n))
    df = bars([(c, c + 0.5, c - 0.5, c) for c in close])
    d = np.where(rng.random(n) > 0.8, np.sign(rng.normal(size=n)), 0.0)
    signals = sigs(list(d), df.index)
    atr = const_atr(df.index, 1.0)
    kw = dict(interval="5m", costs=Costs(0.0004, 1.5), params=P, initial_equity=10_000)

    full = run(df, signals, atr, **kw)
    cut = 250
    trunc = run(df.iloc[:cut], signals.iloc[:cut], atr.iloc[:cut], **kw)

    # 잘린 지점 전에 청산이 끝난 거래만 비교한다
    boundary = df.index[cut - 1]
    a = full.trades[full.trades["exit_time"] < boundary].reset_index(drop=True)
    b = trunc.trades[trunc.trades["exit_time"] < boundary].reset_index(drop=True)
    assert len(a) == len(b) and len(a) > 3
    pd.testing.assert_series_equal(a["net_pnl"], b["net_pnl"], rtol=1e-9)
    pd.testing.assert_series_equal(a["entry_price"], b["entry_price"], rtol=1e-9)


def test_daily_loss_limit_blocks_new_entries():
    """하루 손실 한도를 넘기면 그날은 더 진입하지 않아야 한다."""
    n = 60
    # 진입할 때마다 손절에 걸리는 구간
    rows = []
    for i in range(n):
        rows.append((100, 100.2, 96.0, 97.0))
    df = bars(rows)
    p = ExecParams(risk_per_trade=0.01, atr_stop_mult=1.0, reward_ratio=2.0,
                   daily_loss_limit=0.03, trailing=False)
    r = run(df, sigs([1] * n, df.index), const_atr(df.index, 2.0),
            interval="5m", costs=FREE, params=p, initial_equity=10_000)

    assert r.skipped.get("일손실한도", 0) > 0, "한도가 작동하지 않았다"
    # 1% 리스크로 3% 한도면 4거래 안쪽에서 멈춘다
    assert len(r.trades) <= 4


def test_index_mismatch_raises():
    df = bars([(100, 100, 100, 100)] * 4)
    bad = sigs([0, 1, 0, 0], df.index).iloc[:3]
    with pytest.raises(ValueError, match="signals 인덱스"):
        run(df, bad, const_atr(df.index, 1.0), interval="5m", costs=FREE,
            params=P, initial_equity=10_000)


def test_zero_trade_window_does_not_crash():
    """
    신호가 없는 구간에서도 성적 계산이 터지지 않아야 한다.
    Metrics 필드를 추가했을 때 위치 인자 개수가 어긋나 실제로 터진 적이 있다.
    """
    df = bars([(100, 100, 100, 100)] * 10)
    r = run(df, sigs([0] * 10, df.index), const_atr(df.index, 2.0),
            interval="5m", costs=FREE, params=P, initial_equity=10_000)
    assert r.trades.empty
    assert r.metrics.trades == 0
    assert r.metrics.total_fees == 0.0
    assert r.equity.iloc[-1] == pytest.approx(10_000)


def test_metrics_zero_covers_every_field():
    from src.backtest.metrics import Metrics
    z = Metrics.zero()
    for name in Metrics.__dataclass_fields__:
        assert getattr(z, name) == 0, f"{name} 이 0으로 초기화되지 않았습니다"
