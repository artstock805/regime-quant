"""
레짐 분류기 검증.

이력 현상(hysteresis)은 상태를 이어가는 루프라서 미래 참조를 넣기 쉽다.
지표와 마찬가지로 '잘라서 계산해도 같은 값' 성질을 검사한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_candles
from src.indicators import compute_all
from src.regime import (
    ALL_REGIMES, REQUIRED, Regime, RegimeParams,
    _apply_hysteresis, classify, classify_from_ohlcv, classify_raw,
    regime_stats, trend_score, vol_score,
)

NEEDED = ["adx", "efficiency_ratio", "choppiness", "bbw_percentile", "atr_ratio", "ema_spread"]


@pytest.fixture(scope="module")
def feats() -> pd.DataFrame:
    return compute_all(make_candles(900, "5min", seed=5), NEEDED)


# ─── 이력 현상 ────────────────────────────────────────────────────

def test_hysteresis_requires_consecutive_bars():
    raw = pd.Series(["range"] * 5 + ["uptrend"] * 2 + ["range"] * 5, dtype="object")
    out = _apply_hysteresis(raw, confirm_bars=3)
    # uptrend 가 2봉만 나왔으므로 전환되지 않아야 한다
    assert set(out.unique()) == {"range"}


def test_hysteresis_switches_after_enough_bars():
    raw = pd.Series(["range"] * 5 + ["uptrend"] * 6, dtype="object")
    out = _apply_hysteresis(raw, confirm_bars=3)
    assert out.iloc[6] == "range", "3봉째까지는 아직 전환 전이어야 한다"
    assert out.iloc[7] == "uptrend", "4봉째(연속 3 달성 다음)에는 전환돼 있어야 한다"
    assert out.iloc[-1] == "uptrend"


def test_hysteresis_resets_streak_on_different_candidate():
    """후보가 바뀌면 연속 카운트가 초기화돼야 한다."""
    raw = pd.Series(["range"] * 3 + ["uptrend", "downtrend", "uptrend", "downtrend"], dtype="object")
    out = _apply_hysteresis(raw, confirm_bars=3)
    assert set(out.unique()) == {"range"}


def test_hysteresis_is_causal():
    rng = np.random.default_rng(0)
    raw = pd.Series(rng.choice(["range", "uptrend", "downtrend", "high_vol"], 400), dtype="object")
    full = _apply_hysteresis(raw, confirm_bars=4)
    for cut in (150, 260, 380):
        trunc = _apply_hysteresis(raw.iloc[:cut], confirm_bars=4)
        assert trunc.iloc[-1] == full.iloc[cut - 1], (
            f"이력 현상이 미래를 참조합니다 (cut={cut})"
        )


def test_hysteresis_reduces_switching(feats: pd.DataFrame):
    raw = classify_raw(feats)
    smoothed = _apply_hysteresis(raw, confirm_bars=8)
    raw_switches = int(raw.ne(raw.shift(1)).sum())
    smooth_switches = int(smoothed.ne(smoothed.shift(1)).sum())
    assert smooth_switches < raw_switches, "이력 현상이 전환을 줄이지 못했다"


# ─── 분류 전체 ────────────────────────────────────────────────────

def test_classify_is_causal(feats: pd.DataFrame):
    full = classify(feats)
    for cut in (500, 700, 860):
        trunc = classify(feats.iloc[:cut])
        assert trunc["regime"].iloc[-1] == full["regime"].iloc[cut - 1]
        assert trunc["bars_in"].iloc[-1] == full["bars_in"].iloc[cut - 1]


def test_classify_returns_only_known_regimes(feats: pd.DataFrame):
    got = set(classify(feats)["regime"].dropna().unique())
    assert got <= {str(r) for r in ALL_REGIMES}, f"알 수 없는 레짐: {got}"


def test_confidence_is_bounded(feats: pd.DataFrame):
    conf = classify(feats)["confidence"].dropna()
    assert not conf.empty
    assert conf.between(0.0, 1.0).all(), "신뢰도가 0~1을 벗어났다"


def test_scores_are_bounded(feats: pd.DataFrame):
    assert trend_score(feats).dropna().between(0, 1).all()
    assert vol_score(feats).dropna().between(0, 1).all()


def test_bars_in_counts_current_run(feats: pd.DataFrame):
    reg = classify(feats).dropna(subset=["regime"])
    changed = reg["regime"].ne(reg["regime"].shift(1))
    # 국면이 바뀐 첫 봉의 bars_in 은 1
    firsts = reg.loc[changed, "bars_in"].dropna()
    assert (firsts == 1).all() or firsts.iloc[1:].eq(1).all()
    # 안 바뀐 봉은 직전보다 1 크다
    same = ~changed
    diff = reg["bars_in"].diff()
    assert diff[same].dropna().eq(1).all()


def test_missing_input_raises_instead_of_guessing():
    feats = compute_all(make_candles(400, "5min"), NEEDED).drop(columns=["adx"])
    with pytest.raises(KeyError, match="adx"):
        classify_raw(feats)


def test_warmup_bars_are_unclassified(feats: pd.DataFrame):
    """입력이 NaN인 워밍업 구간에 국면을 지어내지 않아야 한다."""
    raw = classify_raw(feats)
    warm = feats[list(REQUIRED)].notna().all(axis=1)
    assert raw[~warm].isna().all()
    assert raw[warm].notna().all()


# ─── 합성 시나리오 ────────────────────────────────────────────────

def _ramp(n: int, slope: float, noise: float, seed: int = 0) -> pd.DataFrame:
    """일정한 기울기 + 소량의 잡음으로 만든 명확한 추세."""
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC", name="open_time")
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(np.full(n, slope) + rng.normal(0, noise, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    span = np.abs(close) * 0.001
    return pd.DataFrame({
        "open": open_, "high": np.maximum(open_, close) + span,
        "low": np.minimum(open_, close) - span, "close": close,
        "volume": rng.uniform(1, 10, n), "quote_volume": rng.uniform(100, 1000, n),
        "trades": rng.integers(10, 100, n), "taker_buy_base": rng.uniform(0.5, 5, n),
    }, index=idx)


def test_clear_uptrend_is_labelled_uptrend():
    reg = classify_from_ohlcv(_ramp(700, slope=0.0012, noise=0.0004))
    tail = reg["regime"].dropna().iloc[-200:]
    share = (tail == Regime.UPTREND).mean()
    assert share > 0.7, f"명확한 상승 구간에서 상승추세 비율이 {share:.0%}뿐"


def test_clear_downtrend_is_labelled_downtrend():
    reg = classify_from_ohlcv(_ramp(700, slope=-0.0012, noise=0.0004))
    tail = reg["regime"].dropna().iloc[-200:]
    share = (tail == Regime.DOWNTREND).mean()
    assert share > 0.7, f"명확한 하락 구간에서 하락추세 비율이 {share:.0%}뿐"


def test_flat_noise_is_not_labelled_trend():
    """방향 없는 잡음을 추세로 부르면 안 된다."""
    reg = classify_from_ohlcv(_ramp(700, slope=0.0, noise=0.0015, seed=3))
    tail = reg["regime"].dropna().iloc[-200:]
    trend_share = tail.isin([Regime.UPTREND, Regime.DOWNTREND]).mean()
    assert trend_share < 0.35, f"잡음 구간을 추세로 판정한 비율이 {trend_share:.0%}"


def test_regime_stats_shape(feats: pd.DataFrame):
    stats = regime_stats(classify(feats))
    assert len(stats) == len(ALL_REGIMES)
    assert abs(stats["share_%"].sum() - 100.0) < 0.5


def test_params_are_frozen():
    p = RegimeParams()
    with pytest.raises(Exception):
        p.confirm_bars = 99  # type: ignore[misc]
