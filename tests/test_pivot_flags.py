"""
피봇 탐지의 속성을 지킨다 — 측정 스크립트가 기대는 토대다.

**왜 속성으로 재나.** 이 함수는 속도 때문에 루프에서 벡터 연산으로 바뀌었다.
구현을 바꿔도 깨지지 않는 테스트여야 의미가 있으므로, 참조 구현과 비교하는
대신 **피봇이 만족해야 할 성질**을 직접 확인한다.

    피봇 고점 = 자기 창 [i-left, i+right] 의 **유일한** 최대값
    NaN 이 섞인 창에서는 피봇이 나오면 안 된다
    창이 데이터 밖으로 나가는 양 끝에서는 피봇이 나오면 안 된다

여기가 틀리면 probe_pivot_divergence.py 와 probe_lookahead_screen.py 의
결론이 통째로 흔들린다 — 둘 다 "피봇은 right 봉 뒤에야 확정된다"에 기대고 있다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from probe_pivot_divergence import pivot_flags                      # noqa: E402

CASES = [(600, 5, 5), (400, 3, 2), (300, 8, 8), (120, 1, 1)]


def _series(n: int, seed: int, *, nans: int = 0, decimals: int = 2) -> pd.Series:
    """반올림해서 **동점을 일부러 만든다** — 유일성 조건이 실제로 걸리도록."""
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, n).round(decimals)
    if nans:
        v[rng.integers(0, n, nans)] = np.nan
    return pd.Series(v)


@pytest.mark.parametrize("n,left,right", CASES)
def test_a_pivot_high_is_the_unique_maximum_of_its_window(n, left, right):
    s = _series(n, seed=n + left)
    hi, _ = pivot_flags(s, left, right)
    v = s.to_numpy("float64")
    for i in np.flatnonzero(hi):
        w = v[i - left: i + right + 1]
        assert v[i] == w.max()
        assert (w == v[i]).sum() == 1, f"{i}번 봉이 동점인데 피봇으로 잡혔다"


@pytest.mark.parametrize("n,left,right", CASES)
def test_a_pivot_low_is_the_unique_minimum_of_its_window(n, left, right):
    s = _series(n, seed=n + right)
    _, lo = pivot_flags(s, left, right)
    v = s.to_numpy("float64")
    for i in np.flatnonzero(lo):
        w = v[i - left: i + right + 1]
        assert v[i] == w.min()
        assert (w == v[i]).sum() == 1


@pytest.mark.parametrize("n,left,right", CASES)
def test_every_true_maximum_is_found(n, left, right):
    """놓친 피봇이 없어야 한다 — 거짓음성도 결과를 바꾼다."""
    s = _series(n, seed=7)
    hi, lo = pivot_flags(s, left, right)
    v = s.to_numpy("float64")
    for i in range(left, n - right):
        w = v[i - left: i + right + 1]
        if not np.isfinite(w).all():
            continue
        unique_max = v[i] == w.max() and (w == v[i]).sum() == 1
        unique_min = v[i] == w.min() and (w == v[i]).sum() == 1
        assert hi[i] == unique_max, f"{i}번 봉 고점 판정이 틀렸다"
        assert lo[i] == unique_min, f"{i}번 봉 저점 판정이 틀렸다"


def test_a_window_touching_nan_yields_no_pivot():
    """NaN 구간에서 피봇이 나오면 지표 워밍업 구간이 결과에 섞인다."""
    s = _series(300, seed=3, nans=15)
    hi, lo = pivot_flags(s, 5, 5)
    v = s.to_numpy("float64")
    for i in np.flatnonzero(hi | lo):
        assert np.isfinite(v[i - 5: i + 6]).all()


def test_the_edges_cannot_be_pivots():
    """
    **이게 이 측정의 핵심 전제다.** 끝에서 right 봉 안쪽은 아직 피봇이 될 수
    없다 — 오른쪽 봉이 아직 안 왔기 때문이다. 여기가 새면 미래 참조가 된다.
    """
    n, left, right = 200, 5, 5
    hi, lo = pivot_flags(_series(n, seed=1), left, right)
    assert not hi[:left].any() and not lo[:left].any()
    assert not hi[n - right:].any(), "오른쪽 끝에서 피봇이 나왔다 — 미래 참조다"
    assert not lo[n - right:].any()


def test_a_series_shorter_than_the_window_is_handled():
    hi, lo = pivot_flags(_series(6, seed=2), 5, 5)
    assert not hi.any() and not lo.any()
    assert len(hi) == 6


def test_growing_the_series_never_changes_an_already_confirmed_pivot():
    """
    인과성. 봉이 더 와도 **이미 확정된** 피봇 판정은 바뀌지 않아야 한다.
    바뀐다면 신호가 사후에 다시 쓰이는 것이고, 백테스트가 거짓말을 시작한다.
    """
    full = _series(500, seed=9)
    right = 5
    hi_full, lo_full = pivot_flags(full, 5, right)
    for t in (150, 250, 350):
        hi_cut, lo_cut = pivot_flags(full.iloc[:t], 5, right)
        upto = t - right
        assert np.array_equal(hi_full[:upto], hi_cut[:upto])
        assert np.array_equal(lo_full[:upto], lo_cut[:upto])
