"""
검증 도구 검증 — 감사·순열·민감도·효율·리포트.

순열 검정은 **기계 자체가 옳은지**를 확인할 수 있다. 미래를 보는 신호(완벽한 예지)를
넣으면 귀무분포를 압도해야 하고, 무작위 신호를 넣으면 압도하지 못해야 한다.
둘 다 맞으면 그 기계는 신호의 유무를 실제로 구분한다.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from conftest import make_candles
from src.config import Costs
from src.backtest.engine import ExecParams
from src.validation import (
    ValidationReport, audit_bars, audit_summary, block_permute, compare_reports,
    cost_sensitivity, load_report, parameter_plateau, permutation_test,
)

FREE = Costs(fee_rate=0.0, slippage_bps=0.0)
P = ExecParams(risk_per_trade=0.01, atr_stop_mult=2.0, reward_ratio=1.5, trailing=False)


# ═══ 데이터 감사 ═════════════════════════════════════════════════

def test_audit_passes_clean_data():
    df = make_candles(500, "1h")
    r = audit_bars(df, "T", "1h")
    assert r.ok and r.missing == 0 and r.duplicates == 0
    assert r.coverage == pytest.approx(1.0)
    assert r.ohlc_violations == 0 and r.nonpositive == 0


def test_audit_counts_missing_bars():
    df = make_candles(200, "1h")
    holed = pd.concat([df.iloc[:50], df.iloc[57:]])      # 7봉 삭제
    r = audit_bars(holed, "T", "1h")
    assert r.missing == 7
    assert r.coverage == pytest.approx(193 / 200, abs=1e-6)
    assert len(r.missing_spans) == 1 and r.missing_spans[0][2] == 7
    assert r.ok, "봉 누락은 거래소 점검으로도 생긴다 — 구조적 결함은 아니다"


def test_audit_detects_duplicates():
    df = make_candles(100, "1h")
    dupe = pd.concat([df, df.iloc[40:45]]).sort_index()
    r = audit_bars(dupe, "T", "1h")
    assert r.duplicates == 5 and not r.ok


def test_audit_detects_ohlc_violations():
    df = make_candles(100, "1h").copy()
    df.iloc[30, df.columns.get_loc("high")] = df["low"].iloc[30] - 1     # high < low
    df.iloc[60, df.columns.get_loc("low")] = df["close"].iloc[60] + 5    # low > close
    r = audit_bars(df, "T", "1h")
    assert r.ohlc_violations == 2 and not r.ok


def test_audit_detects_nonpositive_prices():
    df = make_candles(100, "1h").copy()
    df.iloc[20, df.columns.get_loc("low")] = 0.0
    r = audit_bars(df, "T", "1h")
    assert r.nonpositive >= 1 and not r.ok


def test_audit_detects_outlier_bar():
    """분할이나 잘못된 덤프는 한 봉의 수익률 이상치로 나타난다."""
    df = make_candles(400, "1h").copy()
    j = 200
    df.iloc[j, df.columns.get_loc("close")] = df["close"].iloc[j] * 5
    r = audit_bars(df, "T", "1h", outlier_z=8.0)
    assert df.index[j].isoformat() in r.outlier_times


def test_audit_outlier_detection_survives_its_own_spike():
    """
    표준편차 기반 z-점수는 자기모순적이다 — 극단값 하나가 표준편차를 부풀려
    곧바로 되돌아온 봉(-80%)을 스스로 가린다. MAD 를 쓰면 둘 다 잡힌다.
    """
    df = make_candles(400, "1h").copy()
    j = 200
    df.iloc[j, df.columns.get_loc("close")] = df["close"].iloc[j] * 5
    r = audit_bars(df, "T", "1h", outlier_z=8.0)
    assert r.outliers >= 2, "급등과 복귀가 각각 이상치로 잡혀야 한다"
    assert df.index[j + 1].isoformat() in r.outlier_times


def test_audit_does_not_cry_wolf_on_normal_data():
    """정상 데이터에서 이상치가 쏟아지면 경보가 무의미해진다."""
    r = audit_bars(make_candles(2000, "1h", seed=5), "T", "1h", outlier_z=8.0)
    assert r.outliers == 0


def test_audit_detects_naive_timezone():
    df = make_candles(100, "1h")
    naive = df.copy()
    naive.index = naive.index.tz_localize(None)
    assert audit_bars(naive, "T", "1h").naive_tz
    assert not audit_bars(naive, "T", "1h").ok


def test_audit_empty_input():
    r = audit_bars(pd.DataFrame(), "T", "1h")
    assert r.bars == 0 and r.coverage == 0.0


def test_audit_summary_folds_problems():
    df = make_candles(100, "1h")
    good = audit_bars(df, "A", "1h").as_dict()
    dupe = audit_bars(pd.concat([df, df.iloc[:3]]).sort_index(), "B", "1h").as_dict()
    s = audit_summary(pd.DataFrame([good, dupe]))
    assert not s["ok"] and any("중복" in p for p in s["problems"])
    s_ok = audit_summary(pd.DataFrame([good]))
    assert s_ok["ok"] and s_ok["problems"] == []


# ═══ 블록 순열 ═══════════════════════════════════════════════════

def _sig(direction: np.ndarray, index) -> pd.DataFrame:
    return pd.DataFrame(
        {"direction": direction.astype("float64"),
         "strength": np.where(direction != 0, 0.9, 0.0)},
        index=index,
    )


def test_block_permute_preserves_values_and_index():
    df = make_candles(300, "1h")
    rng = np.random.default_rng(0)
    d = np.where(rng.random(300) > 0.6, np.sign(rng.normal(size=300)), 0.0)
    sig = _sig(d, df.index)
    perm = block_permute(sig, 20, np.random.default_rng(1))

    assert perm.index.equals(sig.index)
    assert list(perm.columns) == list(sig.columns)
    # 값의 다중집합이 보존된다 (순서만 바뀐다)
    for col in sig.columns:
        assert sorted(perm[col].tolist()) == sorted(sig[col].tolist())


def test_block_permute_keeps_direction_and_strength_paired():
    """따로 섞으면 존재할 수 없는 신호(방향 0인데 확신도 0.9)가 만들어진다."""
    df = make_candles(240, "1h")
    rng = np.random.default_rng(2)
    d = np.where(rng.random(240) > 0.5, 1.0, 0.0)
    perm = block_permute(_sig(d, df.index), 12, np.random.default_rng(3))
    zero = perm["direction"] == 0
    assert (perm.loc[zero, "strength"] == 0.0).all()
    assert (perm.loc[~zero, "strength"] > 0).all()


def test_block_permute_rejects_bad_block_size():
    sig = _sig(np.ones(50), make_candles(50, "1h").index)
    with pytest.raises(ValueError, match="블록 크기"):
        block_permute(sig, 0, np.random.default_rng(0))
    with pytest.raises(ValueError, match="블록 크기"):
        block_permute(sig, 50, np.random.default_rng(0))


# ═══ 순열 검정 ═══════════════════════════════════════════════════

def _perm_case(direction: np.ndarray, bars: pd.DataFrame, n: int = 60):
    from src.indicators import compute_all
    feats = compute_all(bars, ["atr"])
    return permutation_test(
        bars, _sig(direction, bars.index), feats["atr"],
        interval="1h", costs=FREE, params=P, initial_equity=10_000,
        n_permutations=n, block_size=24, seed=0,
    )


@pytest.fixture(scope="module")
def perm_bars() -> pd.DataFrame:
    return make_candles(1600, "1h", seed=21)


def test_permutation_flags_a_foresight_signal(perm_bars):
    """
    다음 봉 방향을 아는 신호(미래 참조)는 귀무분포를 압도해야 한다.
    압도하지 못하면 이 검정은 신호의 유무를 구분하지 못한다는 뜻이다.
    """
    fwd = perm_bars["close"].shift(-1) - perm_bars["close"]
    d = np.sign(fwd.fillna(0.0).to_numpy())
    r = _perm_case(d, perm_bars)
    assert r.significant, f"예지 신호가 유의하지 않다 (p={r.p_value})"
    assert r.observed > r.null_mean
    assert r.percentile > 95


def test_permutation_does_not_flag_a_random_signal(perm_bars):
    rng = np.random.default_rng(31)
    d = np.where(rng.random(len(perm_bars)) > 0.7, np.sign(rng.normal(size=len(perm_bars))), 0.0)
    r = _perm_case(d, perm_bars)
    assert not r.significant, f"무작위 신호가 유의하다고 나왔다 (p={r.p_value})"
    assert r.verdict == "무작위 순서와 구분되지 않는다"


def test_permutation_p_value_is_never_zero(perm_bars):
    """유한 표본에서 p=0 은 성립할 수 없다 — +1 보정이 걸려 있어야 한다."""
    fwd = perm_bars["close"].shift(-1) - perm_bars["close"]
    r = _perm_case(np.sign(fwd.fillna(0.0).to_numpy()), perm_bars, n=30)
    assert r.p_value > 0
    assert r.p_value >= 1 / (r.n_permutations + 1) - 1e-12


def test_permutation_no_trades_returns_nan(perm_bars):
    r = _perm_case(np.zeros(len(perm_bars)), perm_bars, n=5)
    assert math.isnan(r.p_value) and r.verdict == "판정불가"


def test_permutation_default_block_follows_signal_persistence():
    from src.validation.permutation import _default_block
    idx = make_candles(1000, "1h").index
    # 40봉씩 방향이 유지되는 신호 → 블록은 그보다 커야 한다
    d = np.repeat(np.tile([1.0, -1.0], 13)[:25], 40)[:1000]
    assert _default_block(_sig(d, idx)) > 40


# ═══ 파라미터 고원 ═══════════════════════════════════════════════

def _fake_trades(expectancy: float, n: int = 400, seed: int = 0) -> pd.DataFrame:
    """
    평균이 **정확히** expectancy 인 거래 표.

    난수 그대로 쓰면 표준오차(sd 0.5, n 400 → 0.025)가 검정하려는 곡선의 차이를
    삼켜버려서, 고원 판정 로직이 아니라 샘플링 운을 테스트하게 된다.
    잡음의 평균을 0으로 맞춰 곡선을 지정한 대로 만든다.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 0.5, n)
    z -= z.mean()
    return pd.DataFrame({"r_multiple": z + expectancy})


def test_plateau_detected_when_neighbourhood_is_flat():
    curve = {1.0: 0.02, 2.0: 0.05, 3.0: 0.06, 4.0: 0.055, 5.0: 0.03}
    r = parameter_plateau(lambda v: _fake_trades(curve[v], seed=int(v * 10)),
                          "atr_stop_mult", list(curve), chosen=3.0)
    assert r.is_plateau and r.plateau_width >= 3
    assert r.verdict == "고원 위에 있다"
    assert not r.at_edge


def test_cliff_is_not_a_plateau():
    """최적값만 뾰족하고 옆칸이 급락하면 잡음에 맞춘 것이다."""
    curve = {1.0: -0.03, 2.0: -0.02, 3.0: 0.09, 4.0: -0.02, 5.0: -0.04}
    r = parameter_plateau(lambda v: _fake_trades(curve[v], n=1200, seed=int(v * 10)),
                          "atr_stop_mult", list(curve), chosen=3.0)
    assert not r.is_plateau
    assert "고원이 아니다" in r.verdict


def test_plateau_flags_optimum_at_search_edge():
    curve = {1.0: 0.01, 2.0: 0.03, 3.0: 0.05, 4.0: 0.07, 5.0: 0.09}
    r = parameter_plateau(lambda v: _fake_trades(curve[v], seed=int(v * 10)),
                          "atr_stop_mult", list(curve), chosen=3.0)
    assert r.at_edge and "범위를 넓혀야" in r.verdict


def test_plateau_records_chosen_and_best_separately():
    curve = {1.0: 0.01, 2.0: 0.06, 3.0: 0.02}
    r = parameter_plateau(lambda v: _fake_trades(curve[v], n=2000, seed=int(v * 10)),
                          "x", list(curve), chosen=3.0)
    assert r.best == 2.0
    assert r.chosen == 3.0
    assert r.chosen_value < r.best_value


def test_plateau_all_negative_is_not_a_plateau():
    curve = {1.0: -0.05, 2.0: -0.03, 3.0: -0.04}
    r = parameter_plateau(lambda v: _fake_trades(curve[v], n=2000, seed=int(v * 10)),
                          "x", list(curve), chosen=2.0)
    assert not r.is_plateau and r.plateau_width == 0


# ═══ 비용 민감도 ═════════════════════════════════════════════════

def test_cost_breakeven_is_interpolated():
    """비용 배수 1.0 에서 +0.02, 2.0 에서 -0.02 → 손익분기는 1.5 근처"""
    def runner(m: float) -> pd.DataFrame:
        return _fake_trades(0.06 - 0.04 * m, n=3000, seed=int(m * 100))
    r = cost_sensitivity(runner, [0.5, 1.0, 1.5, 2.0, 3.0])
    assert r.breakeven_multiple is not None
    assert r.breakeven_multiple == pytest.approx(1.5, abs=0.05)


def test_cost_breakeven_thin_margin_is_warned():
    """손익분기가 1.5배 미만이면 슬리피지 추정 오차에 뒤집힐 수 있다고 말해야 한다."""
    def runner(m: float) -> pd.DataFrame:
        return _fake_trades(0.05 - 0.04 * m, n=3000, seed=int(m * 100))
    r = cost_sensitivity(runner, [0.5, 1.0, 1.5, 2.0])
    assert 1.0 < r.breakeven_multiple < 1.5
    assert "여유가 거의 없다" in r.verdict


def test_cost_breakeven_wide_margin_is_reassuring():
    def runner(m: float) -> pd.DataFrame:
        return _fake_trades(0.12 - 0.04 * m, n=3000, seed=int(m * 100))
    r = cost_sensitivity(runner, [0.5, 1.0, 2.0, 3.0, 4.0])
    assert r.breakeven_multiple > 2.5
    assert "비용 추정 오차를 견딘다" in r.verdict


def test_cost_breakeven_below_one_is_flagged():
    """실제 비용(1.0배)에서 이미 음수면 '더 싸야만 통한다'고 말해야 한다."""
    def runner(m: float) -> pd.DataFrame:
        return _fake_trades(0.02 - 0.04 * m, n=3000, seed=int(m * 100))
    r = cost_sensitivity(runner, [0.25, 0.5, 1.0, 2.0])
    assert r.breakeven_multiple is not None and r.breakeven_multiple < 1.0
    assert "실제 비용보다 싸야만" in r.verdict


def test_cost_never_positive_is_reported():
    def runner(m: float) -> pd.DataFrame:
        return _fake_trades(-0.05, n=3000, seed=int(m * 100))
    r = cost_sensitivity(runner, [0.25, 1.0, 2.0])
    assert r.breakeven_multiple is None
    assert "양수가 되지 않는다" in r.verdict


# ═══ 리포트 ══════════════════════════════════════════════════════

def _report() -> ValidationReport:
    r = ValidationReport("테스트 검증", {"market": "futures_um", "interval": "1h"})
    r.add("통합 검정", {"net_mean": 0.01, "net_t": 1.26, "trades": 1009,
                     "gross_mean": 0.048, "gross_t": 3.03})
    r.add("창별", [{"창": 0, "합계R": 1.5}, {"창": 1, "합계R": -0.4}])
    r.check("통합 순기대R > 0", False)
    r.check("부호 검정", True)
    return r


def test_report_roundtrips_through_json(tmp_path):
    out = _report().save(tmp_path)
    d = load_report(out)
    assert d["title"] == "테스트 검증"
    assert d["checks"] == {"통합 순기대R > 0": False, "부호 검정": True}
    assert d["passed"] == 1 and d["total_checks"] == 2
    assert d["sections"]["통합 검정"]["net_t"] == pytest.approx(1.26)
    assert d["environment"]["python"]


def test_report_writes_html_and_latest_pointer(tmp_path):
    out = _report().save(tmp_path)
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "테스트 검증" in html and "부호 검정" in html
    assert "<table>" in html
    assert (tmp_path / "latest.txt").read_text(encoding="utf-8") == out.name


def test_report_converts_numpy_and_nan():
    r = ValidationReport("x", {})
    r.add("s", {"a": np.int64(3), "b": np.float64(1.5),
                "c": float("nan"), "d": float("inf"), "e": np.bool_(True)})
    s = r.as_dict()["sections"]["s"]
    assert s == {"a": 3, "b": 1.5, "c": None, "d": None, "e": True}
    json.dumps(r.as_dict())      # 직렬화가 터지지 않아야 한다


def test_report_verdict_wording():
    r = ValidationReport("x", {})
    r.check("a", True); r.check("b", True)
    assert "전부 통과" in r.verdict
    r2 = ValidationReport("x", {})
    for k, v in zip("abcd", [True, False, False, False]):
        r2.check(k, v)
    assert "우연과 구분되지 않는다" in r2.verdict


def test_compare_reports_lines_up_runs(tmp_path):
    a = _report().save(tmp_path, name="run-a")
    b = _report()
    b.check("통합 순기대R > 0", True)
    b_dir = b.save(tmp_path, name="run-b")
    cmp = compare_reports([a, b_dir])
    assert len(cmp) == 2
    assert set(cmp["실행"]) == {"run-a", "run-b"}
    assert "[통합 순기대R > 0]" in cmp.columns
    assert cmp.set_index("실행").loc["run-a", "[통합 순기대R > 0]"] == "X"
    assert cmp.set_index("실행").loc["run-b", "[통합 순기대R > 0]"] == "O"
    assert "net_t" in cmp.columns


def test_report_html_collapses_nested_lists():
    """
    중첩 리스트를 표 칸에 펼치면 표가 무너진다. 개수만 보이고 전체는 JSON 에 남는다.
    실제로 missing_spans 가 칸을 넘쳐 읽을 수 없었다.
    """
    r = ValidationReport("x", {})
    spans = [["2022-02-25T23:00", "2022-03-01T00:00", 72],
             ["2022-03-31T23:00", "2022-04-03T00:00", 48]]
    r.add("감사", [{"symbol": "SOL", "missing_spans": spans, "missing": 120}])
    html = r.render_html()
    cell = html.split("<table>")[1].split("</table>")[0]
    assert ">2건<" in cell, "칸에는 개수만 보여야 한다"
    # 자세한 내용은 title 툴팁으로만 — 칸 본문을 넘치지 않는다
    assert 'title="' in cell and "2022-03-01T00:00" in cell.split('title="')[1]
    # 원본은 JSON 에 온전히 남아야 한다
    assert r.as_dict()["sections"]["감사"][0]["missing_spans"] == spans


def test_report_html_escapes_markup():
    r = ValidationReport("x", {})
    r.add("s", [{"전략": "<script>alert(1)</script>"}])
    html = r.render_html()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_rerender_rebuilds_html_from_json(tmp_path):
    """
    JSON 이 원본이고 HTML 은 그것을 그린 것이다. 표현이 바뀌었을 때 비싼 계산을
    다시 돌리지 않고 HTML 만 갱신할 수 있어야 한다.
    """
    from src.validation import rerender

    out = _report().save(tmp_path, name="r1")
    (out / "report.html").write_text("깨진 내용", encoding="utf-8")
    path = rerender(out)
    html = path.read_text(encoding="utf-8")
    assert "깨진 내용" not in html
    assert "테스트 검증" in html and "부호 검정" in html
    # 체크 개수와 판정이 JSON 과 일치해야 한다
    assert load_report(out)["verdict"] in html
