"""
지표 인과성 검사 — 이 프로젝트에서 가장 중요한 테스트.

원리
    시점 t의 지표값이 t 이하의 데이터만으로 결정된다면,
    데이터를 t에서 잘라 계산해도 t의 값은 같아야 한다.
    다르다면 그 지표는 미래를 훔쳐본 것이고, 백테스트 수익률은 전부 거짓이다.

이 검사를 레지스트리 전체에 자동으로 돌린다. 새 지표를 추가하면 검사 대상에
자동으로 포함되므로, 미래 참조를 들고 들어오면 바로 걸린다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_candles
from src.indicators import all_specs, compute, compute_all, summary

# 잘라낼 지점들. 워밍업(min_periods)이 끝난 뒤여야 값이 NaN이 아니다.
CUT_POINTS = (420, 500, 610)
BARS = 700


@pytest.fixture(scope="module")
def candles() -> pd.DataFrame:
    return make_candles(BARS, "5min", seed=42)


def _values_equal(a, b) -> bool:
    """NaN == NaN 을 같다고 본다."""
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return bool(np.isclose(float(a), float(b), rtol=1e-9, atol=1e-12))


@pytest.mark.parametrize("name", sorted(all_specs()))
def test_indicator_is_causal(name: str, candles: pd.DataFrame):
    full = compute(candles, name)
    full_df = full.to_frame() if isinstance(full, pd.Series) else full

    for cut in CUT_POINTS:
        truncated = compute(candles.iloc[:cut], name)
        trunc_df = truncated.to_frame() if isinstance(truncated, pd.Series) else truncated

        for col in full_df.columns:
            expected = full_df[col].iloc[cut - 1]
            actual = trunc_df[col].iloc[-1]
            assert _values_equal(expected, actual), (
                f"{name}.{col} 이 미래를 참조합니다.\n"
                f"  전체 데이터로 계산한 {cut - 1}번째 값: {expected}\n"
                f"  {cut}봉까지만 주고 계산한 마지막 값: {actual}\n"
                f"  → center=True, shift(음수), bfill 을 쓰지 않았는지 확인하세요."
            )


@pytest.mark.parametrize("name", sorted(all_specs()))
def test_indicator_declares_its_outputs(name: str, candles: pd.DataFrame):
    """반환 열 이름이 선언과 일치해야 한다 — compute_all 이 열을 잘못 붙이지 않도록."""
    spec = all_specs()[name]
    out = compute(candles, name)
    if isinstance(out, pd.Series):
        assert out.name == spec.output_names[0]
        assert len(spec.output_names) == 1
    else:
        assert set(out.columns) == set(spec.output_names)


@pytest.mark.parametrize("name", sorted(all_specs()))
def test_indicator_produces_real_numbers(name: str, candles: pd.DataFrame):
    """
    700봉을 줬는데 전부 NaN이면 그 지표는 사실상 죽어 있다.
    파라미터 오타나 0으로 나누기를 조용히 넘기지 않기 위한 검사.
    """
    out = compute(candles, name)
    out_df = out.to_frame() if isinstance(out, pd.Series) else out
    tail = out_df.iloc[-100:]
    for col in out_df.columns:
        valid = tail[col].notna().sum()
        assert valid >= 50, f"{name}.{col}: 마지막 100봉 중 유효값이 {valid}개뿐입니다"
        assert np.isfinite(tail[col].dropna()).all(), f"{name}.{col}: inf 가 들어 있습니다"


def test_indicator_index_is_preserved(candles: pd.DataFrame):
    for name in all_specs():
        out = compute(candles, name)
        assert out.index.equals(candles.index), f"{name}: 인덱스가 바뀌었습니다"


# ─── 지표 풀 전체 ─────────────────────────────────────────────────

def test_compute_all_has_unique_columns(candles: pd.DataFrame):
    feats = compute_all(candles)
    dupes = feats.columns[feats.columns.duplicated()].tolist()
    assert not dupes, f"열 이름이 중복됩니다: {dupes}"


def test_compute_all_is_causal(candles: pd.DataFrame):
    """개별 지표가 아니라 풀 전체를 한 번에 다시 검사한다."""
    full = compute_all(candles)
    cut = 610
    trunc = compute_all(candles.iloc[:cut])
    assert list(full.columns) == list(trunc.columns)

    bad = [
        col for col in full.columns
        if not _values_equal(full[col].iloc[cut - 1], trunc[col].iloc[-1])
    ]
    assert not bad, f"미래 참조 지표: {bad}"


def test_pool_size_is_in_target_range():
    """
    지표 40~80종을 목표로 했다. 세는 단위는 '등록 지표 수'다.
    MACD(3열)·볼린저(4열)처럼 다중 출력 지표가 있어서 특징열 수는 이보다 많다.
    """
    n_specs = len(all_specs())
    assert 40 <= n_specs <= 80, (
        f"등록 지표 {n_specs}종 — 목표 범위(40~80)를 벗어났습니다\n"
        f"{summary().to_string(index=False)}"
    )


def test_feature_columns_expand_from_multi_output_indicators():
    feats = compute_all(make_candles(300, "5min"))
    n_specs = len(all_specs())
    assert len(feats.columns) >= n_specs, "특징열이 지표 수보다 적을 수는 없다"
    assert len(feats.columns) <= n_specs * 4, "다중 출력이 지나치게 부푼 것은 아닌지 확인"


def test_taker_indicators_skipped_when_column_absent():
    """거래소를 바꿔 taker_buy_base 가 없어도 파이프라인이 죽지 않아야 한다."""
    df = make_candles(300, "5min").drop(columns=["taker_buy_base"])
    feats = compute_all(df)
    assert "taker_imbalance" not in feats.columns
    assert "rsi" in feats.columns


def test_registry_covers_every_category():
    from src.indicators import by_category
    cats = by_category()
    assert set(cats) == {"trend", "momentum", "volatility", "volume",
                         "derivatives", "orderflow"}, cats
    for cat, names in cats.items():
        assert len(names) >= 5, f"{cat} 카테고리에 지표가 {len(names)}개뿐입니다"


def test_derivative_indicators_skipped_when_columns_absent():
    """
    파생 데이터를 안 받은 캔들로도 파이프라인이 그대로 돌아야 한다.
    2021-01 이전 구간과, 파생 아카이브가 없는 거래소를 위한 경로다.
    """
    from src.data.derivatives import ALL_FIELDS
    df = make_candles(300, "5min", derivatives=False)
    assert not set(ALL_FIELDS) & set(df.columns), "파생 열이 남아 있으면 검사가 무의미하다"
    feats = compute_all(df)
    assert "oi_change" not in feats.columns
    assert "funding_zscore" not in feats.columns
    assert "rsi" in feats.columns, "파생이 없다고 기본 지표까지 사라지면 안 된다"


def test_derivative_indicators_appear_when_columns_present():
    """반대 방향 — 열이 있으면 실제로 계산돼야 한다. 조용히 건너뛰면 못 알아챈다."""
    feats = compute_all(make_candles(300, "5min"))
    for col in ("oi_change", "oi_zscore", "oi_price_regime", "long_buildup",
                "funding_zscore", "funding_cum", "perp_taker_ls",
                "crowd_vs_smart", "top_trader_tilt", "account_position_gap"):
        assert col in feats.columns, f"{col} 이 계산되지 않았습니다"
