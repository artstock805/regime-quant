"""
앙상블 — 확신도를 문턱과 같은 척도로 만드는가.

이 파일이 생긴 계기는 실측 버그다. `taker_flow_momentum` 은 진입 조건을
|z|>=0.15 로 선언하는데 확신도를 scale(|z|, 0.15, 2.0) 으로 만든다. 그 중위가
**0.104** 라서, 앙상블 문턱 0.35 는 실질적으로 |z|>=0.80 을 요구했다.
전략이 선언한 조건과 **5배 어긋나** 신호의 85~89% 가 설계 의도와 무관하게 버려졌다.

문턱 하나가 전략마다 다른 뜻을 갖는 구조는 비교가 성립하지 않는다.
확신도를 자기 최근 분포의 백분위로 바꾸면 "상위 몇 %"라는 같은 뜻이 된다.
"""

from __future__ import annotations

# ─── 확신도 백분위 캘리브레이션 ────────────────────────────────────

def test_calibrated_strength_is_causal():
    """
    백분위는 과거만 봐야 한다. 뒤를 잘라내도 앞의 값이 같아야 한다.
    이걸 틀리면 전 구간 분포를 알고 있는 셈이 되어 백테스트가 거짓이 된다.
    """
    import numpy as np
    import pandas as pd
    from src.backtest.ensemble import calibrated_strength

    rng = np.random.default_rng(0)
    n = 600
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    sig = pd.DataFrame({"direction": rng.choice([-1.0, 0.0, 1.0], n),
                        "strength": rng.uniform(0, 0.3, n)}, index=idx)
    full = calibrated_strength(sig, window=100)
    cut = calibrated_strength(sig.iloc[:400], window=100)
    pd.testing.assert_series_equal(full.iloc[:400], cut, check_names=False)


def test_calibrated_strength_spreads_a_compressed_scale():
    """
    핵심 목적. 확신도가 0 근처에 몰려 있어도 백분위는 0~1 로 펴져야 한다.
    실제로 taker_flow_momentum 확신도 중위가 0.104 라 문턱 0.35 를 거의 못 넘었다.
    """
    import numpy as np
    import pandas as pd
    from src.backtest.ensemble import calibrated_strength

    rng = np.random.default_rng(1)
    n = 1200
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    # 0~0.25 에 몰린 확신도 — 실측 분포를 흉내 낸다
    sig = pd.DataFrame({"direction": np.ones(n),
                        "strength": rng.beta(1.5, 12, n)}, index=idx)
    raw, cal = sig["strength"], calibrated_strength(sig, window=300)
    assert raw.median() < 0.2, "원래 확신도는 낮은 쪽에 몰려 있어야 한다"
    tail = cal.iloc[400:]
    assert 0.4 < tail.median() < 0.6, f"백분위 중위가 {tail.median():.2f} — 0.5 근처여야 한다"
    assert (raw > 0.35).mean() < (tail > 0.35).mean(), "문턱 통과 비율이 늘어야 한다"


def test_calibrated_strength_is_zero_where_no_signal():
    import numpy as np
    import pandas as pd
    from src.backtest.ensemble import calibrated_strength

    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    direction = np.zeros(n); direction[::4] = 1.0
    sig = pd.DataFrame({"direction": direction,
                        "strength": np.full(n, 0.2)}, index=idx)
    cal = calibrated_strength(sig, window=30)
    assert (cal[sig["direction"] == 0] == 0).all(), "무신호 봉은 0 이어야 한다"
    assert (cal[sig["direction"] != 0] > 0).any()


def test_combine_respects_the_calibrate_flag():
    """기본은 꺼져 있어야 한다 — 켜는 것은 새로운 시험 한 번이다."""
    from src.backtest.ensemble import WeightParams
    assert WeightParams().calibrate_strength is False
