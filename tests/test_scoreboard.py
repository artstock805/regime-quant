"""
지표 성적표 — 순위를 내면서 그 순위를 의심하는가.

성적표는 만들기 쉽고 믿기 쉽다. 그래서 도구 자체를 먼저 검증한다.
알려진 정답이 있는 인공 데이터를 넣어 다음을 확인한다.

    · 진짜 예측력이 있는 지표를 상단에 올리는가
    · 순수 잡음만 넣으면 **지속성이 없다고 말하는가** ← 이게 핵심
    · 방향을 직전 기간에서만 배우는가 (미래 참조)
    · 지표 수만큼 다중검정을 보정하는가
    · 새로 추가된 지표를 표시하는가

두 번째가 가장 중요하다. 잡음에 대고 "순위가 유지된다"고 말하는 성적표는
없는 것보다 나쁘다 — 이 프로젝트는 종목 순위에서 이미 그 함정을 측정했다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation.scoreboard import (
    LEVEL_LIKE, ScoreboardConfig, Snapshot, aggregate, build_panel,
    diff_snapshots, make_snapshot, rank_persistence, score_periods, top_stability,
)


def _cfg(**kw):
    # 기간 단위를 월로 둔다. 6,000시간은 250일이라 분기로 나누면 2~3개뿐이고,
    # 그러면 지속성(기간 쌍)을 잴 수가 없다. 실제 실행은 분기가 기본값이다.
    base = dict(horizon=4, atr_stop_mult=4.0, cost_r=0.0, entry_quantile=0.7,
                z_period=48, period_freq="M", min_period_obs=200,
                min_direction_obs=200)
    base.update(kw)
    return ScoreboardConfig(**base)


def _make(n=4000, seed=0, signal_strength=0.0, n_noise=4):
    """
    합성 종목 하나. `signal` 열은 미래수익률을 signal_strength 만큼 예측한다.
    나머지 noise_* 는 순수 잡음이다.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC")
    shock = rng.normal(0, 1.0, n)
    # signal[t] 이 앞으로의 충격을 예측하도록 만든다 (미래 정보를 심는 게 아니라,
    # 지표가 t 시점에 알 수 있는 값이 t+ 의 움직임과 상관되게 만든다)
    sig = np.roll(shock, -1) * signal_strength + rng.normal(0, 1.0, n)
    close = 100.0 + np.cumsum(shock * 0.5)

    bars = pd.DataFrame({"open": close, "high": close + 0.3, "low": close - 0.3,
                         "close": close, "volume": rng.uniform(1, 10, n)}, index=idx)
    feats = pd.DataFrame({"atr": np.full(n, 0.5), "signal": sig}, index=idx)
    for i in range(n_noise):
        feats[f"noise_{i}"] = rng.normal(0, 1, n)
    return bars, feats


# ─── 패널 구성 ────────────────────────────────────────────────────

def test_build_panel_requires_atr_to_define_one_r():
    bars, feats = _make(500)
    with pytest.raises(KeyError, match="atr"):
        build_panel({"X": (bars, feats.drop(columns=["atr"]))}, _cfg())


def test_build_panel_drops_level_like_columns():
    """가격 수준에 가까운 열을 남기면 '예측력' 대신 '가격과 같은 것'을 잰다."""
    bars, feats = _make(800)
    feats["bb_upper"] = bars["close"] * 1.01     # LEVEL_LIKE 에 있는 이름
    panel = build_panel({"X": (bars, feats)}, _cfg())
    assert "bb_upper" not in panel.columns
    assert "bb_upper" in LEVEL_LIKE
    assert "signal" in panel.columns


def test_build_panel_normalises_by_one_r():
    """
    수익률을 그대로 비교하면 변동성 큰 종목이 무조건 이긴다.
    1R(=atr_stop_mult x ATR)로 나눠야 종목 간 비교가 성립한다.
    """
    bars, feats = _make(600)
    panel = build_panel({"X": (bars, feats)}, _cfg(atr_stop_mult=4.0))
    assert (panel["r_unit"] == 2.0).all(), "ATR 0.5 x 4 = 2.0 이어야 한다"


# ─── 미래 참조 ────────────────────────────────────────────────────

def test_direction_is_learned_only_from_the_previous_period():
    """
    지표값이 높을 때 오르는지 내리는지를 전 구간으로 정하면 그게 미래 참조다.
    뒤쪽 기간을 잘라내도 앞쪽 기간의 점수가 바뀌면 안 된다.
    """
    bars, feats = _make(6000, signal_strength=0.5)
    cfg = _cfg()
    full = score_periods(build_panel({"X": (bars, feats)}, cfg), cfg)
    cut = score_periods(build_panel({"X": (bars.iloc[:4000], feats.iloc[:4000])}, cfg), cfg)

    # 잘린 쪽의 마지막 기간은 데이터가 덜 들어와 값이 다를 수 있으므로 뺀다
    common = sorted(set(full["period"]) & set(cut["period"]))[:-1]
    assert common, "겹치는 기간이 있어야 검사가 성립한다"
    a = full[full["period"].isin(common)].set_index(["period", "indicator"])
    b = cut[cut["period"].isin(common)].set_index(["period", "indicator"])
    shared = a.index.intersection(b.index)
    assert len(shared) > 3
    np.testing.assert_allclose(a.loc[shared, "expectancy_r"],
                               b.loc[shared, "expectancy_r"], rtol=1e-9)


def test_forward_return_is_never_used_as_a_feature():
    bars, feats = _make(800)
    panel = build_panel({"X": (bars, feats)}, _cfg())
    indicators = set(panel.columns) - {"symbol", "time", "period", "r_unit", "fwd"}
    assert "fwd" not in indicators
    assert not any("fwd" in c for c in indicators)


# ─── 순위가 실제로 신호를 찾는가 ──────────────────────────────────

def test_a_genuinely_predictive_indicator_ranks_above_noise():
    bars, feats = _make(9000, signal_strength=0.8, n_noise=6)
    cfg = _cfg()
    per = score_periods(build_panel({"X": (bars, feats)}, cfg), cfg)
    scores = aggregate(per)
    names = [s.name for s in scores]
    assert names[0] == "signal", f"예측력 있는 지표가 1위여야 한다 (실제 {names[:3]})"


def test_noise_only_scoreboard_reports_no_persistence():
    """
    **가장 중요한 검정.** 잡음만 넣었을 때 "순위가 유지된다"고 말하면
    이 도구는 없는 것보다 나쁘다.
    """
    cfg = _cfg()
    frames = {f"S{i}": _make(6000, seed=100 + i, signal_strength=0.0, n_noise=8)
              for i in range(3)}
    per = score_periods(build_panel(frames, cfg), cfg)
    pers = rank_persistence(per, n_perm=1000, seed=1)
    assert pers["pairs"] >= 3
    assert pers["p"] >= 0.05, (
        f"잡음인데 지속성 p={pers['p']:.3f} — 없는 패턴을 찾았다고 말하고 있다")
    assert "무작위" in pers["verdict"] or "역전" in pers["verdict"]


def test_persistence_is_detected_when_it_really_exists():
    """반대 방향 — 진짜 지속성이 있으면 찾아내야 한다. 못 찾으면 둔한 도구다."""
    cfg = _cfg()
    frames = {}
    for i in range(3):
        # 같은 지표가 계속 예측력을 갖는 구조
        bars, feats = _make(9000, seed=7 + i, signal_strength=0.9, n_noise=6)
        frames[f"S{i}"] = (bars, feats)
    per = score_periods(build_panel(frames, cfg), cfg)
    pers = rank_persistence(per, n_perm=1000, seed=2)
    assert pers["mean_rho"] > 0.15, (
        f"진짜 지속성이 있는데 순위상관이 {pers['mean_rho']:.3f} 뿐이다")


# ─── 다중검정 ─────────────────────────────────────────────────────

def test_holm_correction_is_applied_across_all_indicators():
    """
    지표 96개를 각각 검정하면 5% 수준에서 우연히 5개쯤 통과한다.
    보정 없는 순위표는 그 우연을 발견으로 착각하게 만든다.
    """
    cfg = _cfg()
    frames = {f"S{i}": _make(6000, seed=200 + i, signal_strength=0.0, n_noise=12)
              for i in range(2)}
    per = score_periods(build_panel(frames, cfg), cfg)
    scores = aggregate(per)
    assert scores, "점수가 나와야 한다"
    for s in scores:
        assert s.p_holm >= s.p - 1e-12, "보정 p 는 원래 p 보다 작아질 수 없다"
    assert sum(s.significant for s in scores) == 0, (
        "잡음에서 Holm 통과 지표가 나왔다")


def test_direction_flips_are_counted():
    """방향이 기간마다 뒤집히는 지표는 순위 상단에 있어도 믿을 수 없다."""
    cfg = _cfg()
    frames = {f"S{i}": _make(6000, seed=300 + i, signal_strength=0.0, n_noise=6)
              for i in range(2)}
    per = score_periods(build_panel(frames, cfg), cfg)
    scores = aggregate(per)
    assert any(s.direction_flips > 0 for s in scores), (
        "잡음 지표라면 방향이 뒤집힌 것이 있어야 한다")


def test_top_stability_reports_its_random_baseline():
    """상위 k 유지율은 기준선(k/N) 과 함께 봐야 의미가 있다."""
    cfg = _cfg()
    frames = {f"S{i}": _make(6000, seed=400 + i, signal_strength=0.0, n_noise=10)
              for i in range(2)}
    per = score_periods(build_panel(frames, cfg), cfg)
    stab = top_stability(per, k=3)
    assert stab["pairs"] >= 1
    assert 0.0 <= stab["hit_rate"] <= 1.0
    assert 0.0 < stab["baseline"] < 1.0


# ─── 스냅샷과 새 지표 감지 ────────────────────────────────────────

def test_snapshot_records_the_trial_count():
    """
    지표 N개를 동시에 검정하면 시험 N회다. 그 숫자가 기록되지 않으면
    Deflated Sharpe 를 정직하게 계산할 수 없다.
    """
    cfg = _cfg()
    bars, feats = _make(6000, signal_strength=0.3, n_noise=5)
    per = score_periods(build_panel({"X": (bars, feats)}, cfg), cfg)
    scores = aggregate(per)
    snap = make_snapshot(scores, per, ["X"], cfg)
    assert snap.trials == len(scores) > 0
    assert any("시험" in n for n in snap.notes)


def test_snapshot_warns_when_persistence_is_absent():
    cfg = _cfg()
    frames = {f"S{i}": _make(6000, seed=500 + i, signal_strength=0.0, n_noise=8)
              for i in range(3)}
    per = score_periods(build_panel(frames, cfg), cfg)
    scores = aggregate(per)
    snap = make_snapshot(scores, per, list(frames), cfg)
    assert any("지속성" in n for n in snap.notes), (
        "지속성이 없는데 경고를 달지 않았다")


def test_diff_flags_newly_added_indicators():
    """
    지표를 추가하고 성적표를 다시 돌리는 것은 시험 횟수를 늘리는 행위다.
    그 사실이 보이지 않으면 조용히 과적합한다.
    """
    prev = {"indicators": [{"name": "a"}, {"name": "b"}], "trials": 2}
    cur = Snapshot(created="now", config={}, symbols=["X"],
                   indicators=[{"name": "b"}, {"name": "a"}, {"name": "c"}],
                   persistence={}, top_stability={}, trials=3)
    d = diff_snapshots(prev, cur)
    assert d["new"] == ["c"]
    assert d["gone"] == []
    assert d["trials_after"] == 5, "누적 시험 횟수가 더해져야 한다"
    moves = {m["name"]: m["delta"] for m in d["moves"]}
    assert moves["b"] == 1 and moves["a"] == -1


def test_diff_handles_the_first_run():
    cur = Snapshot(created="now", config={}, symbols=["X"],
                   indicators=[{"name": "a"}], persistence={}, top_stability={},
                   trials=1)
    d = diff_snapshots(None, cur)
    assert d["is_first"] and d["new"] == ["a"]


def test_diff_reports_dropped_indicators():
    prev = {"indicators": [{"name": "a"}, {"name": "z"}], "trials": 2}
    cur = Snapshot(created="now", config={}, symbols=["X"],
                   indicators=[{"name": "a"}], persistence={}, top_stability={},
                   trials=1)
    assert diff_snapshots(prev, cur)["gone"] == ["z"]


# ─── 빈 입력 ──────────────────────────────────────────────────────

def test_empty_inputs_do_not_crash():
    assert build_panel({}, _cfg()).empty
    assert score_periods(pd.DataFrame(), _cfg()).empty
    assert aggregate(pd.DataFrame()) == []
    assert rank_persistence(pd.DataFrame())["verdict"] == "표본부족"
    assert top_stability(pd.DataFrame())["pairs"] == 0
