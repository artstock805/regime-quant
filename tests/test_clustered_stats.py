"""
군집 강건 표준오차 — '종목을 늘려서 유의성을 만드는' 착시를 잡는가.

이 검정을 넣은 이유는 명확하다. 유니버스를 8종목에서 134종목으로 늘리면 거래 수가
16배가 되고 순진한 t 는 √16 = 4배로 뛴다. 그런데 암호화폐 알트는 서로 0.8 이상
상관돼 있어서, 같은 시각의 134개 거래는 독립 관측 134개가 아니라 사실상 한 개다.
**표본을 늘려 유의성을 얻었다고 착각하는 것이 이 프로젝트에서 가장 위험한 실수다.**

그래서 도구 자체를 먼저 검증한다 — 알려진 정답이 있는 인공 데이터로.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation.stats import clustered_mean_test, t_test, time_clusters


def test_independent_data_gives_same_se_as_naive():
    """관측이 실제로 독립이면 군집 SE 는 순진한 SE 와 거의 같아야 한다."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.02, 0.5, 4000)
    cluster = np.arange(4000) // 4          # 군집 안에도 상관이 없다
    r = clustered_mean_test(x, cluster)
    assert 0.85 < r.inflation < 1.15, f"팽창률 {r.inflation:.3f} — 독립인데 달라졌다"
    assert abs(r.t - t_test(x).t) < 0.5


def test_correlated_clusters_inflate_the_standard_error():
    """
    핵심 검정. 같은 군집이 공통 충격을 공유하면 SE 가 커지고 t 가 줄어야 한다.
    줄지 않으면 이 도구는 쓸모가 없다.
    """
    rng = np.random.default_rng(1)
    n_clusters, per = 200, 30
    shock = rng.normal(0, 0.45, n_clusters)          # 그 주의 시장 충격
    rows, labels = [], []
    for g in range(n_clusters):
        rows.append(shock[g] + rng.normal(0.02, 0.15, per))   # 개별 잡음은 작다
        labels.append(np.full(per, g))
    x = np.concatenate(rows)
    cluster = np.concatenate(labels)

    naive = t_test(x)
    r = clustered_mean_test(x, cluster)
    assert r.inflation > 2.0, f"팽창률이 {r.inflation:.2f} 뿐 — 상관을 못 잡았다"
    assert abs(r.t) < abs(naive.t), "군집 t 가 순진한 t 보다 작아야 한다"
    assert r.effective_n < len(x) / 3, (
        f"유효표본 {r.effective_n:.0f} / 실제 {len(x)} — 너무 관대하다")


def test_spurious_significance_from_duplicating_symbols_is_caught():
    """
    가장 중요한 검정 — **같은 신호를 종목 수만큼 복제**한 상황.
    거래 수는 16배가 되지만 정보는 하나도 늘지 않았다.
    순진한 t 는 4배로 뛰고, 군집 t 는 제자리여야 한다.
    """
    rng = np.random.default_rng(2)
    base = rng.normal(0.02, 0.5, 500)                # 한 종목의 거래
    weeks = np.arange(500)

    one = t_test(base)
    # 같은 주에 16종목이 사실상 같은 거래를 한다 (상관 1.0 극단)
    many = np.tile(base, 16)
    many_weeks = np.tile(weeks, 16)
    naive_many = t_test(many)
    clustered_many = clustered_mean_test(many, many_weeks)

    # 부호가 아니라 크기를 본다. 표본 평균이 음수로 나오는 시드도 있다.
    assert abs(naive_many.t) > 3.0 * abs(one.t), (
        f"복제하면 순진한 |t| 가 √16=4배 가까이 부푼다 — "
        f"{abs(one.t):.3f} → {abs(naive_many.t):.3f}")
    assert abs(clustered_many.t - one.t) < 0.35 * abs(one.t) + 0.3, (
        f"군집 t {clustered_many.t:.2f} vs 원본 {one.t:.2f} — "
        "복제로 만든 가짜 유의성을 못 걸렀다")


def test_too_few_clusters_refuses_to_judge():
    """군집이 적으면 추정 자체가 불안정하다. 틀린 답보다 '모른다'가 낫다."""
    rng = np.random.default_rng(3)
    x = rng.normal(0.02, 0.5, 300)
    cluster = np.arange(300) % 4
    r = clustered_mean_test(x, cluster, min_clusters=10)
    assert r.n_clusters == 4
    assert not np.isfinite(r.t)
    assert r.verdict == "표본부족"


def test_time_clusters_groups_by_week():
    times = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    labels = time_clusters(times, "W")
    assert len(set(labels)) in (5, 6), f"30일이 주 단위로 5~6개여야 하는데 {len(set(labels))}"
    assert len(labels) == 30


def test_clustered_test_matches_hand_computed_variance():
    """공식을 직접 계산해 맞춘다 — 구현이 표류하지 않도록."""
    x = np.array([1.0, 2.0, 3.0, 4.0] * 10)
    cluster = np.array([0, 0, 1, 1] * 10)
    cluster = cluster + np.repeat(np.arange(10) * 2, 4)   # 20개 군집
    r = clustered_mean_test(x, cluster)

    mean = x.mean()
    dev = x - mean
    codes, inv = np.unique(cluster, return_inverse=True)
    sums = np.bincount(inv, weights=dev)
    g, n = len(codes), len(x)
    var = (sums ** 2).sum() / n ** 2 * (g / (g - 1))
    assert abs(r.se_clustered - np.sqrt(var)) < 1e-12


def test_effective_sample_never_exceeds_actual_sample():
    """
    군집 내 상관이 음수면 군집 SE 가 순진한 SE 보다 작아지고, n/팽창² 이 n 을
    넘는다. '독립 관측이 실제 관측보다 많다'는 말은 성립하지 않는다.
    실측에서 4시간봉 실행이 '유효표본 556 / 실제 461 = 121%' 로 찍혔다.
    """
    import numpy as np
    from src.validation.stats import clustered_mean_test

    rng = np.random.default_rng(5)
    n_clusters, per = 150, 4
    rows, labels = [], []
    for g in range(n_clusters):
        # 군집 안에서 서로 상쇄되는(음의 상관) 값들
        base = rng.normal(0, 0.4)
        rows.append(np.array([base, -base, base * 0.5, -base * 0.5]) + 0.01)
        labels.append(np.full(per, g))
    x = np.concatenate(rows)
    r = clustered_mean_test(x, np.concatenate(labels))
    assert r.inflation < 1.0, "이 데이터는 군집 SE 가 더 작아야 한다"
    assert r.effective_n <= r.n, (
        f"유효표본 {r.effective_n:.0f} > 실제 {r.n} — 있을 수 없다")
