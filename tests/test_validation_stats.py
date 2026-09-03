"""
통계 검정 프리미티브 검증.

이 파일이 제일 중요하다. 여기 있는 함수들이 다른 모든 숫자를 판정하므로,
판정자가 틀리면 그 위의 모든 결론이 무의미해진다.

알려진 정답이 있는 경우로 검증한다 — 표준정규 분위수, 이항분포 손계산,
평균이 0인/양수인 분포에서의 t검정 결과.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.validation.stats import (
    binomial_p_at_least, bootstrap_ci, deflated_sharpe, gain_concentration,
    holm, benjamini_hochberg, norm_cdf, norm_ppf, sign_test, t_test,
)


# ─── 분포 ────────────────────────────────────────────────────────

def test_norm_cdf_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-4)
    assert norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-4)
    assert norm_cdf(2.576) == pytest.approx(0.995, abs=1e-4)


def test_norm_ppf_inverts_cdf():
    for p in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999):
        assert norm_cdf(norm_ppf(p)) == pytest.approx(p, abs=1e-9)


def test_norm_ppf_known_values():
    assert norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)


def test_norm_ppf_rejects_out_of_range():
    assert math.isnan(norm_ppf(0.0))
    assert math.isnan(norm_ppf(1.0))
    assert math.isnan(norm_ppf(1.5))


def test_binomial_matches_hand_calculation():
    # 8번 중 6번 이상 성공: (28 + 8 + 1) / 256
    assert binomial_p_at_least(6, 8) == pytest.approx(37 / 256)
    # 8번 중 8번: 1/256
    assert binomial_p_at_least(8, 8) == pytest.approx(1 / 256)
    # 0번 이상은 항상 1
    assert binomial_p_at_least(0, 8) == pytest.approx(1.0)


def test_binomial_seven_of_eight_is_significant():
    """실제로 나왔던 판정 — 7/8 양수는 p=0.035 로 유의하다."""
    assert binomial_p_at_least(7, 8) == pytest.approx(9 / 256, abs=1e-6)
    assert binomial_p_at_least(7, 8) < 0.05


# ─── t검정 ───────────────────────────────────────────────────────

def test_t_test_on_zero_mean_is_not_significant():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 4000)
    st = t_test(x)
    assert abs(st.t) < 3
    assert not st.significant
    assert st.verdict == "판정불가"


def test_t_test_on_positive_mean_is_significant():
    rng = np.random.default_rng(1)
    x = rng.normal(0.1, 1, 4000)      # 효과크기 0.1, n=4000 → t ≈ 6
    st = t_test(x)
    assert st.t > 4 and st.significant and st.verdict == "양의우위"
    assert st.ci_lo > 0


def test_t_test_detects_negative_edge():
    rng = np.random.default_rng(2)
    st = t_test(rng.normal(-0.1, 1, 4000))
    assert st.verdict == "음의우위" and st.ci_hi < 0
    assert not st.significant


def test_t_test_small_sample_returns_nan():
    st = t_test(np.array([0.1, 0.2, 0.3]))
    assert st.n == 3 and math.isnan(st.mean) and st.verdict == "표본부족"
    assert not st.significant


def test_t_test_ignores_nan_and_inf():
    x = np.array([1.0, 2.0, np.nan, np.inf, -np.inf] + [1.5] * 30)
    st = t_test(x)
    assert st.n == 32


def test_trades_needed_matches_formula():
    """관측 효과를 t=1.96 으로 만들 표본 수 = (1.96·sd/mean)²"""
    rng = np.random.default_rng(3)
    x = rng.normal(0.02, 0.5, 1200)
    st = t_test(x)
    expected = int((1.96 * st.sd / st.mean) ** 2)
    assert st.trades_needed() == pytest.approx(expected, rel=0.02)


def test_trades_needed_is_none_when_mean_is_zero():
    st = t_test(np.zeros(200))
    assert st.trades_needed() is None


def test_bootstrap_ci_brackets_the_true_mean():
    rng = np.random.default_rng(4)
    x = rng.normal(0.5, 1.0, 2000)
    lo, hi = bootstrap_ci(x, n_boot=2000, seed=7)
    assert lo < 0.5 < hi
    assert hi - lo < 0.2, "n=2000 에서 구간이 이렇게 넓을 수 없다"


def test_bootstrap_ci_is_deterministic_for_a_seed():
    x = np.random.default_rng(5).normal(0, 1, 500)
    assert bootstrap_ci(x, seed=11) == bootstrap_ci(x, seed=11)


def test_bootstrap_handles_fat_tails():
    """R배수 분포는 정규가 아니다 — 손절/목표에서 잘려 두 봉우리가 된다."""
    x = np.concatenate([np.full(700, -1.0), np.full(300, 2.0)])   # 기대값 -0.1
    st = t_test(x)
    assert st.mean == pytest.approx(-0.1)
    assert st.ci_hi < 0, "명확한 음의 기대값을 잡아야 한다"


# ─── 다중검정 보정 ───────────────────────────────────────────────

def test_holm_is_monotone_and_bounded():
    p = {"a": 0.001, "b": 0.02, "c": 0.3, "d": 0.9}
    adj = holm(p)
    assert all(0 <= v <= 1 for v in adj.values())
    # 원 p값 순서가 보정 후에도 유지된다
    order = sorted(p, key=lambda k: p[k])
    vals = [adj[k] for k in order]
    assert vals == sorted(vals)


def test_holm_is_at_least_as_strict_as_raw():
    p = {"a": 0.01, "b": 0.02, "c": 0.04}
    adj = holm(p)
    assert all(adj[k] >= p[k] for k in p)


def test_holm_kills_a_lone_borderline_finding():
    """8종목 중 하나가 p=0.16 이면 보정 후 유의할 수 없다 — 실제로 나왔던 상황."""
    p = {f"s{i}": v for i, v in enumerate([0.16, 0.20, 0.35, 0.45, 0.50, 0.59, 0.86, 0.98])}
    assert all(v >= 0.05 for v in holm(p).values())


def test_bh_is_less_conservative_than_holm():
    p = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04, "e": 0.05}
    h, b = holm(p), benjamini_hochberg(p)
    assert all(b[k] <= h[k] + 1e-12 for k in p)


def test_corrections_handle_nan_pvalues():
    p = {"a": 0.01, "b": float("nan")}
    assert all(np.isfinite(v) for v in holm(p).values())
    assert all(np.isfinite(v) for v in benjamini_hochberg(p).values())


# ─── Deflated Sharpe ────────────────────────────────────────────

def test_deflated_sharpe_shrinks_as_trials_grow():
    rng = np.random.default_rng(6)
    x = rng.normal(0.05, 0.5, 1200)
    trials = np.array([0.02, 0.05, 0.08, 0.11, -0.01])
    dsrs = [deflated_sharpe(x, n, trials).dsr for n in (5, 20, 100, 500)]
    assert all(np.isfinite(d) for d in dsrs)
    assert dsrs == sorted(dsrs, reverse=True), "시험 횟수가 늘면 DSR 은 내려가야 한다"


def test_deflated_sharpe_expected_max_grows_with_trials():
    rng = np.random.default_rng(7)
    x = rng.normal(0.05, 0.5, 800)
    trials = np.array([0.01, 0.04, 0.07, 0.1])
    e5 = deflated_sharpe(x, 5, trials).expected_max
    e100 = deflated_sharpe(x, 100, trials).expected_max
    assert e100 > e5 > 0


def test_deflated_sharpe_small_sample_is_nan():
    d = deflated_sharpe(np.array([0.1] * 10), 20)
    assert math.isnan(d.dsr) and not d.significant


def test_deflated_sharpe_flags_a_strong_result():
    rng = np.random.default_rng(8)
    x = rng.normal(0.4, 0.5, 3000)     # 거래샤프 0.8 — 매우 강함
    d = deflated_sharpe(x, 20, np.array([0.05, 0.1, 0.15, 0.2]))
    assert d.significant and d.dsr > 0.95


# ─── 집중도·부호 검정 ────────────────────────────────────────────

def test_gain_concentration_uses_positive_sum_not_total():
    """
    합계가 음수일 때 '전체 대비 비중'은 음수가 나와 무의미하다.
    실제로 -62%, -47% 같은 값을 뱉은 버그가 있었다.
    """
    vals = np.array([10.0, -5.0, -20.0, 2.0])     # 합계 -13, 이익 합계 12
    c = gain_concentration(vals)
    assert 0.0 <= c <= 1.0
    assert c == pytest.approx(10.0 / 12.0)


def test_gain_concentration_all_negative_is_nan():
    assert math.isnan(gain_concentration(np.array([-1.0, -2.0])))


def test_gain_concentration_single_source_is_one():
    assert gain_concentration(np.array([5.0, -1.0, -2.0])) == pytest.approx(1.0)


def test_sign_test_matches_binomial():
    r = sign_test({f"s{i}": v for i, v in enumerate([1, 1, 1, 1, 1, 1, 1, -1])})
    assert r["n"] == 8 and r["positive"] == 7
    assert r["p"] == pytest.approx(binomial_p_at_least(7, 8))
    assert r["significant"]


def test_sign_test_one_of_eight_is_not_significant():
    """5.9년 표본에서 실제로 나왔던 결과 — 1/8 양수."""
    r = sign_test({f"s{i}": v for i, v in enumerate([1, -1, -1, -1, -1, -1, -1, -1])})
    assert r["positive"] == 1 and r["p"] > 0.9 and not r["significant"]


def test_sign_test_drops_nan_entries():
    r = sign_test({"a": 1.0, "b": float("nan"), "c": -1.0})
    assert r["n"] == 2


def test_bootstrap_ci_handles_large_samples_without_exhausting_memory():
    """
    거래 5만 건에 재표본 4000회를 (n_boot, n) 한 덩어리로 잡으면 1.48 GiB 를
    요구하고 MemoryError 로 죽는다. 실제로 겪은 사고라 크기를 고정해 둔다.
    """
    import numpy as np
    from src.validation.stats import bootstrap_ci

    rng = np.random.default_rng(0)
    x = rng.normal(0.02, 0.5, 50_000)
    lo, hi = bootstrap_ci(x, n_boot=4000)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < x.mean() < hi


def test_bootstrap_ci_is_unchanged_by_chunking():
    """청크로 나눠도 통계적 결론이 같아야 한다 — 구간이 서로를 크게 벗어나지 않는다."""
    import numpy as np
    from src.validation.stats import bootstrap_ci

    rng = np.random.default_rng(1)
    x = rng.normal(0.05, 0.4, 3_000)
    a = bootstrap_ci(x, n_boot=2000, seed=7)
    b = bootstrap_ci(x, n_boot=2000, seed=7)
    assert a == b, "같은 시드면 완전히 같아야 한다"
    width = a[1] - a[0]
    c = bootstrap_ci(x, n_boot=2000, seed=8)
    assert abs(c[0] - a[0]) < width * 0.25
    assert abs(c[1] - a[1]) < width * 0.25
