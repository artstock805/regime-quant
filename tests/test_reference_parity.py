"""
지표 구현 대조 — 독립 라이브러리(`ta`)와 값이 같은가.

기존 검증이 못 잡는 구멍을 메운다.

    test_causality.py            미래를 보지 않는가
    test_vectorized_equivalence  최적화 구현이 자기 참조 구현과 같은가
    **이 파일**                   구현이 **표준 정의**와 같은가

앞의 두 개는 자기 자신과의 일관성만 본다. 정의를 잘못 구현하면 자기 자신과는
완벽히 일관되므로 전부 통과한다. 실제로 CCI 가 그랬다 — 평균편차를 각 봉의
자기 이동평균 기준으로 계산해서, 5분봉에서 18% 어긋난 채로 모든 내부 검증을
통과하고 있었다.

`ta` 가 없으면 건너뛴다. 필수 의존성으로 만들 만큼 무겁지는 않지만,
지표를 고칠 때는 반드시 이 파일을 돌려야 한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_candles
from src.indicators import compute_all

ta = pytest.importorskip("ta", reason="대조 기준 라이브러리 `ta` 가 필요합니다")

TOL = 1e-6      # 같은 정의라면 부동소수 오차만 남아야 한다


def _ref(df):
    """`ta` 로 계산한 기준값. 파라미터는 우리 기본값과 맞춘다."""
    return {
        "rsi": ta.momentum.RSIIndicator(df["close"], 14).rsi(),
        "atr": ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14)
               .average_true_range(),
        "adx": ta.trend.ADXIndicator(df["high"], df["low"], df["close"], 14).adx(),
        "macd": ta.trend.MACD(df["close"], 26, 12, 9).macd(),
        "macd_signal": ta.trend.MACD(df["close"], 26, 12, 9).macd_signal(),
        "cci": ta.trend.CCIIndicator(df["high"], df["low"], df["close"], 20).cci(),
        "williams_r": ta.momentum.WilliamsRIndicator(
            df["high"], df["low"], df["close"], 14).williams_r(),
        "mfi": ta.volume.MFIIndicator(
            df["high"], df["low"], df["close"], df["volume"], 14).money_flow_index(),
        "obv": ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume(),
    }


# Wilder 평활(RSI·ATR·ADX)은 초기값 시딩 방식이 라이브러리마다 다르다.
# 그 차이는 지수적으로 감쇠해서 실측상 200봉이면 0.000003%, 400봉이면 0이 된다.
# 정의 차이가 아니므로 수렴 구간에서 비교한다.
WARMUP = 400

# 레벨이 아니라 **차분**으로 비교해야 하는 지표. 시작점이 임의라 상수 오프셋이 생긴다.
# (OBV 는 실측에서 레벨 차이의 표준편차가 정확히 0 — 순수 오프셋이었다.)
DIFF_ONLY = {"obv"}


@pytest.fixture(scope="module")
def candles():
    # 합성 데이터로 충분하다. 정의가 다르면 어떤 데이터에서도 어긋난다.
    return make_candles(1200, "1h", seed=31)


@pytest.mark.parametrize("name", list(_ref(make_candles(50, "1h"))))
def test_matches_reference_library(name, candles):
    ours = compute_all(candles, [name])[name]
    theirs = _ref(candles)[name]

    if name in DIFF_ONLY:
        ours, theirs = ours.diff(), theirs.diff()

    both = ours.notna() & theirs.notna()
    a = ours[both].to_numpy()[WARMUP:]
    b = theirs[both].to_numpy()[WARMUP:]
    assert len(a) > 200, f"{name}: 수렴 구간이 부족합니다"

    err = np.abs(a - b) / np.maximum(np.abs(b), 1e-9)
    worst = float(err.max())

    assert worst < TOL, (
        f"{name}: 기준 라이브러리와 최대 상대오차 {worst * 100:.6f}% "
        f"(워밍업 {WARMUP}봉 제외) · 우리 {a[-1]:.6f} vs ta {b[-1]:.6f} — "
        "정의가 다릅니다. 평활 방식(Wilder vs 단순), 평균편차 기준, "
        "초기값 처리를 확인하세요."
    )


def test_wilder_indicators_converge_after_warmup(candles):
    """
    RSI·ATR·ADX 는 초기 몇백 봉에서 기준 라이브러리와 다르다 — 시딩 차이다.
    **수렴한다는 것 자체**를 고정한다. 수렴하지 않으면 정의가 다른 것이다.
    """
    for name in ("rsi", "atr", "adx"):
        ours = compute_all(candles, [name])[name]
        theirs = _ref(candles)[name]
        both = ours.notna() & theirs.notna()
        a, b = ours[both].to_numpy(), theirs[both].to_numpy()
        err = np.abs(a - b) / np.maximum(np.abs(b), 1e-9)
        assert err[50] > err[-1], f"{name}: 오차가 줄어들지 않는다"
        assert err[-1] < 1e-9, f"{name}: 끝까지 수렴하지 않는다 ({err[-1]:.3e})"


def test_obv_differs_only_by_a_constant_offset(candles):
    """OBV 는 시작점이 임의다. 레벨은 달라도 되지만 **변화량**은 같아야 한다."""
    ours = compute_all(candles, ["obv"])["obv"]
    theirs = _ref(candles)["obv"]
    gap = (ours - theirs).dropna()
    assert float(gap.std()) < 1e-9, "차이가 상수가 아니다 — 정의가 다르다"


def test_warmup_requirement_is_documented(candles):
    """
    수렴에 필요한 봉 수를 실제로 잰다. 백테스트가 그보다 짧은 워밍업을 쓰면
    지표값이 기준과 미묘하게 달라진다.
    """
    ours = compute_all(candles, ["adx"])["adx"]
    theirs = _ref(candles)["adx"]
    both = ours.notna() & theirs.notna()
    err = (np.abs(ours[both].to_numpy() - theirs[both].to_numpy())
           / np.maximum(np.abs(theirs[both].to_numpy()), 1e-9))
    converged = int(np.argmax(err < 1e-6))
    assert 0 < converged <= WARMUP, (
        f"ADX 수렴에 {converged}봉 필요 — WARMUP({WARMUP}) 을 넘었습니다"
    )


def test_cci_uses_window_mean_not_rolling_mean(candles):
    """
    CCI 회귀 테스트.

    평균편차는 **그 윈도우의 단일 평균**으로부터 재야 한다.
    각 봉의 자기 이동평균과의 편차를 다시 평균내던 옛 구현을 여기서 직접 재현해
    둘이 실제로 다르다는 것과, 우리가 올바른 쪽이라는 것을 함께 고정한다.
    """
    from src.indicators.core import sma, typical_price

    tp = typical_price(candles)
    ma = sma(tp, 20)
    wrong = (tp - ma) / (0.015 * (tp - ma).abs().rolling(20, min_periods=20).mean())
    ours = compute_all(candles, ["cci"])["cci"]
    theirs = ta.trend.CCIIndicator(candles["high"], candles["low"], candles["close"], 20).cci()

    valid = ours.notna() & theirs.notna() & wrong.notna()
    assert np.allclose(ours[valid], theirs[valid], rtol=1e-9), "우리 CCI 가 표준과 다릅니다"
    # 옛 구현은 실제로 눈에 띄게 달랐다 — 그래야 이 테스트가 의미가 있다
    gap = (wrong[valid] - theirs[valid]).abs().max()
    assert gap > 1.0, "옛 구현과 차이가 없다면 이 회귀 테스트는 무의미하다"
