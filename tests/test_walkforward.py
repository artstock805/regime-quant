"""
앙상블·walk-forward 검증.

가장 중요한 것: **학습 구간 정보가 검증 구간으로 새지 않는가.**
누수가 있으면 결과가 좋게 나오므로, 음(-)의 결과일 때도 누수 검사는 필요하다 —
다음에 양의 결과가 나왔을 때 그게 진짜인지 판별할 수 있어야 하기 때문이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.ensemble import WeightParams, combine, derive_weights, score_by_regime
from src.backtest.walkforward import make_windows
from src.regime import Regime


# ─── 창 생성 ──────────────────────────────────────────────────────

def test_windows_do_not_overlap_train_and_test():
    ws = make_windows("2024-09-01", "2026-08-31", train_months=6, test_months=2)
    assert len(ws) >= 8
    for w in ws:
        assert w.train_end == w.test_start, "학습 끝과 검증 시작이 맞닿아야 한다"
        assert w.train_start < w.train_end < w.test_end
        # 검증 구간이 학습 구간과 겹치면 안 된다
        assert w.test_start >= w.train_end


def test_windows_roll_forward_by_test_length():
    ws = make_windows("2024-09-01", "2026-08-31", train_months=6, test_months=2)
    for a, b in zip(ws, ws[1:]):
        assert b.test_start > a.test_start
        assert (b.train_start.year - a.train_start.year) * 12 + \
               (b.train_start.month - a.train_start.month) == 2


def test_windows_never_exceed_end():
    end = pd.Timestamp("2026-08-31", tz="UTC")
    for w in make_windows("2024-09-01", "2026-08-31", 6, 2):
        assert w.test_end <= end


def test_short_period_yields_no_windows():
    assert make_windows("2026-01-01", "2026-03-01", 6, 2) == []


# ─── 가중치 산정 ──────────────────────────────────────────────────

def _trades(rows: list[tuple[str, str, int, float]]) -> pd.DataFrame:
    """(strategy, regime, n, expectancy_r) 를 거래 표로 펼친다."""
    out = []
    for strat, reg, n, exp_r in rows:
        for _ in range(n):
            out.append({"strategy": strat, "regime": reg, "r_multiple": exp_r,
                        "net_pnl": exp_r})
    return pd.DataFrame(out)


DECLARED = {
    "a": (Regime.UPTREND,),
    "b": (Regime.UPTREND,),
    "c": (Regime.RANGE,),
}


def test_weights_sum_to_one_per_regime():
    tr = _trades([("a", "uptrend", 40, 0.3), ("b", "uptrend", 40, 0.1)])
    w = derive_weights(score_by_regime(tr), DECLARED)
    assert w.loc["uptrend"].sum() == pytest.approx(1.0)
    assert w.loc["range"].sum() == pytest.approx(0.0), "표본 없는 레짐은 0"


def test_negative_expectancy_strategy_gets_no_weight():
    """기대값이 음수인 전략은 발언권을 잃어야 한다 — 상황판의 '제외'."""
    tr = _trades([("a", "uptrend", 40, 0.3), ("b", "uptrend", 40, -0.2)])
    w = derive_weights(score_by_regime(tr), DECLARED)
    assert w.at["uptrend", "b"] == 0.0
    assert w.at["uptrend", "a"] == pytest.approx(1.0)


def test_small_sample_gets_no_weight():
    """표본이 부족하면 성과가 좋아도 가중치를 주지 않는다."""
    tr = _trades([("a", "uptrend", 40, 0.2), ("b", "uptrend", 5, 0.9)])
    w = derive_weights(score_by_regime(tr, WeightParams(min_trades=20)), DECLARED)
    assert w.at["uptrend", "b"] == 0.0


def test_undeclared_regime_gets_no_weight():
    """
    전략이 선언하지 않은 국면에서 성과가 좋아도 가중치를 주지 않는다.
    우연히 맞은 것을 실력으로 오인하면 가중치가 엉뚱하게 배분된다.
    """
    tr = _trades([("c", "uptrend", 50, 0.9)])   # c 는 range 만 선언
    w = derive_weights(score_by_regime(tr), DECLARED)
    assert w.at["uptrend", "c"] == 0.0


DECLARED3 = {**DECLARED, "c": (Regime.UPTREND,)}


def test_max_weight_cap_is_respected_when_feasible():
    """전략 3개 + 상한 0.45 → 합 1.0이 가능하므로 상한이 지켜져야 한다."""
    tr = _trades([("a", "uptrend", 40, 5.0), ("b", "uptrend", 40, 0.3),
                  ("c", "uptrend", 40, 0.2)])
    wp = WeightParams(max_weight=0.45)
    w = derive_weights(score_by_regime(tr, wp), DECLARED3, wp)
    assert w.at["uptrend", "a"] <= 0.45 + 1e-9, "한 전략이 독식하면 안 된다"
    assert w.loc["uptrend"].sum() == pytest.approx(1.0)


def test_infeasible_cap_is_relaxed_to_keep_conviction():
    """
    전략 2개 + 상한 0.45 는 합의 최대가 0.90 이라 실현 불가능하다.
    상한을 그대로 밀어붙이면 그 레짐의 확신도가 45%로 깎여 진입 문턱을 못 넘는다.
    상한을 1/n 으로 완화해 합 1.0을 유지해야 한다.
    """
    tr = _trades([("a", "uptrend", 40, 5.0), ("b", "uptrend", 40, 0.1)])
    wp = WeightParams(max_weight=0.45)
    w = derive_weights(score_by_regime(tr, wp), DECLARED, wp)
    assert w.loc["uptrend"].sum() == pytest.approx(1.0), "확신도가 조용히 깎였다"
    assert w.at["uptrend", "a"] == pytest.approx(0.5)


def test_single_eligible_strategy_gets_full_weight():
    tr = _trades([("a", "uptrend", 40, 0.3)])
    w = derive_weights(score_by_regime(tr), DECLARED)
    assert w.at["uptrend", "a"] == pytest.approx(1.0)
    assert w.loc["uptrend"].sum() == pytest.approx(1.0)


# ─── 신호 합산 ────────────────────────────────────────────────────

def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC", name="open_time")


def test_combine_respects_regime_weights():
    idx = _index(4)
    regimes = pd.Series(["uptrend", "uptrend", "range", "range"], index=idx, dtype=object)
    sig = {
        "a": pd.DataFrame({"direction": [1.0] * 4, "strength": [1.0] * 4}, index=idx),
        "b": pd.DataFrame({"direction": [1.0] * 4, "strength": [1.0] * 4}, index=idx),
    }
    weights = pd.DataFrame(
        {"a": [0.6, 0.0, 0.0, 0.0], "b": [0.4, 0.0, 0.0, 0.0]},
        index=["uptrend", "downtrend", "range", "high_vol"],
    )
    out = combine(sig, regimes, weights, WeightParams(entry_threshold=0.35))
    # 상승추세 구간은 0.6+0.4 = 1.0 → 진입
    assert out["direction"].iloc[0] == 1.0
    assert out["score"].iloc[0] == pytest.approx(1.0)
    # 횡보 구간은 가중치 0 → 진입 없음
    assert out["direction"].iloc[2] == 0.0


def test_combine_below_threshold_does_not_enter():
    idx = _index(2)
    regimes = pd.Series(["uptrend"] * 2, index=idx, dtype=object)
    sig = {"a": pd.DataFrame({"direction": [1.0, 1.0], "strength": [0.2, 0.2]}, index=idx)}
    weights = pd.DataFrame({"a": [1.0, 0, 0, 0]},
                           index=["uptrend", "downtrend", "range", "high_vol"])
    out = combine(sig, regimes, weights, WeightParams(entry_threshold=0.35))
    assert (out["direction"] == 0.0).all(), "합산 점수가 문턱 미만이면 진입하지 않는다"


def test_combine_opposing_votes_cancel():
    idx = _index(2)
    regimes = pd.Series(["uptrend"] * 2, index=idx, dtype=object)
    sig = {
        "a": pd.DataFrame({"direction": [1.0, 1.0], "strength": [1.0, 1.0]}, index=idx),
        "b": pd.DataFrame({"direction": [-1.0, -1.0], "strength": [1.0, 1.0]}, index=idx),
    }
    weights = pd.DataFrame({"a": [0.5, 0, 0, 0], "b": [0.5, 0, 0, 0]},
                           index=["uptrend", "downtrend", "range", "high_vol"])
    out = combine(sig, regimes, weights, WeightParams())
    assert (out["score"].abs() < 1e-9).all()
    assert (out["direction"] == 0.0).all()


def test_combine_rejects_mismatched_index():
    idx = _index(4)
    regimes = pd.Series(["uptrend"] * 4, index=idx, dtype=object)
    sig = {
        "a": pd.DataFrame({"direction": [1.0] * 4, "strength": [1.0] * 4}, index=idx),
        "b": pd.DataFrame({"direction": [1.0] * 3, "strength": [1.0] * 3}, index=idx[:3]),
    }
    weights = pd.DataFrame({"a": [1.0, 0, 0, 0], "b": [0.0, 0, 0, 0]},
                           index=["uptrend", "downtrend", "range", "high_vol"])
    with pytest.raises(ValueError, match="신호 인덱스"):
        combine(sig, regimes, weights, WeightParams())


def test_score_by_regime_marks_eligibility():
    tr = _trades([("a", "uptrend", 40, 0.3), ("b", "uptrend", 5, 0.3),
                  ("c", "range", 40, -0.1)])
    s = score_by_regime(tr, WeightParams(min_trades=20))
    got = {(r["strategy"], r["regime"]): r["eligible"] for _, r in s.iterrows()}
    assert got[("a", "uptrend")] is True
    assert got[("b", "uptrend")] is False, "표본 부족"
    assert got[("c", "range")] is False, "기대값 음수"
