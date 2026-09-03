"""
전략 검증.

전략은 `_hold_until` / `latch` 로 상태를 앞으로 끌고 간다. 앞으로 채우는 연산은
미래 참조를 넣기 쉬우므로, 지표와 같은 '잘라도 같은 값' 검사를 전략에도 돌린다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_candles
from src.indicators import compute_all
from src.strategies import all_strategies, build_all, required_features, strategy_table
from src.strategies.base import latch
from src.strategies.library import _hold_until

CUTS = (520, 640, 780)
BARS = 850


@pytest.fixture(scope="module")
def data():
    df = make_candles(BARS, "5min", seed=17)
    return df, compute_all(df)


# ─── 상태 유지 도구 ───────────────────────────────────────────────

def test_hold_until_holds_between_enter_and_exit():
    enter = pd.Series([False, True, False, False, False, False])
    exit_c = pd.Series([False, False, False, True, False, False])
    out = _hold_until(enter, exit_c)
    assert list(out) == [False, True, True, False, False, False]


def test_hold_until_reenters_after_exit():
    enter = pd.Series([True, False, False, True, False])
    exit_c = pd.Series([False, True, False, False, False])
    out = _hold_until(enter, exit_c)
    assert list(out) == [True, False, False, True, True]


def test_hold_until_is_causal():
    rng = np.random.default_rng(3)
    enter = pd.Series(rng.random(300) > 0.9)
    exit_c = pd.Series(rng.random(300) > 0.9)
    full = _hold_until(enter, exit_c)
    for cut in (120, 200, 290):
        trunc = _hold_until(enter.iloc[:cut], exit_c.iloc[:cut])
        assert trunc.iloc[-1] == full.iloc[cut - 1]


def test_latch_holds_entry_value_through_the_run():
    value = pd.Series([0.1, 0.9, 0.2, 0.3, 0.4])
    active = pd.Series([False, True, True, True, False])
    out = latch(value, active)
    # 구간이 시작된 봉의 값(0.9)이 구간 내내 유지된다
    assert list(out) == [0.0, 0.9, 0.9, 0.9, 0.0]


def test_latch_resets_on_new_run():
    value = pd.Series([0.5, 0.1, 0.8, 0.2])
    active = pd.Series([True, False, True, True])
    assert list(latch(value, active)) == [0.5, 0.0, 0.8, 0.8]


def test_latch_is_causal():
    rng = np.random.default_rng(5)
    value = pd.Series(rng.random(300))
    active = pd.Series(rng.random(300) > 0.5)
    full = latch(value, active)
    for cut in (120, 200, 290):
        trunc = latch(value.iloc[:cut], active.iloc[:cut])
        assert trunc.iloc[-1] == pytest.approx(full.iloc[cut - 1])


# ─── 전략 전체 ────────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(all_strategies()))
def test_strategy_is_causal(name: str, data):
    """
    데이터를 t에서 잘라도 t의 신호는 같아야 한다.
    다르면 그 전략은 미래를 보고 진입·청산을 결정한 것이다.
    """
    df, feats = data
    strat = all_strategies()[name]()
    full = strat.generate(df, feats)

    for cut in CUTS:
        trunc = strat.generate(df.iloc[:cut], feats.iloc[:cut])
        for col in ("direction", "strength"):
            expected = full[col].iloc[cut - 1]
            actual = trunc[col].iloc[-1]
            assert actual == pytest.approx(expected, abs=1e-9), (
                f"{name}.{col} 이 미래를 참조합니다 (cut={cut}): "
                f"{expected} vs {actual}"
            )


@pytest.mark.parametrize("name", sorted(all_strategies()))
def test_strategy_signal_shape(name: str, data):
    df, feats = data
    out = all_strategies()[name]().generate(df, feats)
    assert list(out.columns) == ["direction", "strength"]
    assert out.index.equals(df.index)
    assert out["direction"].isin([-1.0, 0.0, 1.0]).all()
    assert out["strength"].between(0.0, 1.0).all()
    # 방향이 없으면 확신도도 0이어야 한다 — 앙상블이 0방향에 가중을 주지 않도록
    assert (out.loc[out["direction"] == 0.0, "strength"] == 0.0).all()


@pytest.mark.parametrize("name", sorted(all_strategies()))
def test_strategy_actually_trades(name: str, data):
    """850봉을 줬는데 신호가 없으면 그 전략은 죽어 있다."""
    df, feats = data
    out = all_strategies()[name]().generate(df, feats)
    active = (out["direction"] != 0).sum()
    assert active >= 10, f"{name}: 활성 봉이 {active}개뿐입니다"


@pytest.mark.parametrize("name", sorted(all_strategies()))
def test_strength_is_not_collapsed(name: str, data):
    """
    확신도가 진입 시점에 고정되지 않으면 보유 중 0으로 붕괴하고,
    그 전략은 실력과 무관하게 앙상블에서 저평가된다.
    """
    df, feats = data
    out = all_strategies()[name]().generate(df, feats)
    active = out.loc[out["direction"] != 0, "strength"]
    if active.empty:
        pytest.skip("신호 없음")
    assert active.mean() > 0.05, f"{name}: 평균 확신도가 {active.mean():.4f} 로 붕괴했습니다"


def test_missing_feature_raises_instead_of_guessing(data):
    df, feats = data
    strat = all_strategies()["range_position_fade"]()
    with pytest.raises(KeyError, match="dc_pos"):
        strat.generate(df, feats.drop(columns=["dc_pos"]))


def test_registry_has_both_families():
    """모멘텀 계열과 평균회귀 계열이 모두 등록돼 있어야 한다."""
    names = set(all_strategies())
    assert {"donchian_breakout", "ema_cross_adx", "keltner_squeeze"} <= names
    assert {"range_position_fade", "vwap_reversion",
            "flow_extension_fade", "trend_extension_fade"} <= names


def test_required_features_resolve_to_indicators():
    """전략의 needs 를 compute_all 에 그대로 넘길 수 있어야 한다."""
    from src.indicators import resolve
    cols = required_features(build_all())
    assert resolve(cols), "출력 열 이름이 지표로 해석되지 않습니다"


def test_strategy_table_is_complete():
    t = strategy_table()
    assert len(t) == len(all_strategies())
    assert t["korean"].str.len().gt(0).all()
    assert t["regimes"].str.len().gt(0).all()
