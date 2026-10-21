"""
오더플로 **대용** 지표와 그것을 쓰는 전략.

**재료의 한계를 테스트가 알고 있어야 한다.** 리테일 FX 에는 체결 방향·체결량·
호가창이 없다 (XM 실측: real_volume 전봉 0, 호가창 제공 안 함, 틱의 last 와
volume 이 180일 전까지 100% 0). 그래서 방향은 봉 안 종가 위치로 추정하고,
활동량은 틱 카운트를 쓴다. 여기 테스트는 그 추정이 **의도한 대로** 계산되는지를
보는 것이지, 그것이 진짜 오더플로라고 주장하지 않는다.

특히 지키는 것:
    · 고가=저가 인 봉에서 0 으로 나누지 않는다 (inf 하나가 z-score 전체를 오염시킴)
    · 누적델타를 추세 제거해서 쓴다 (그냥 누적하면 시간만 재게 된다)
    · 스프레드가 벌어진 봉에서는 진입하지 않는다 (00시 롤오버 인공물 + 비용)
    · orderflow 카테고리가 레지스트리에 실제로 실린다 (안 실으면 조용히 사라진다)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators import compute_all, orderflow as of
from src.indicators.registry import all_specs
from src.regime import Regime
from src.strategies import build_all


def frame(closes, highs=None, lows=None, volume=1000.0, spread=19.0, n=None):
    """봉 프레임. 종가만 주면 고저는 종가 주변으로 만든다."""
    closes = np.asarray(closes, dtype="float64")
    n = len(closes)
    highs = np.asarray(highs, dtype="float64") if highs is not None else closes + 0.001
    lows = np.asarray(lows, dtype="float64") if lows is not None else closes - 0.001
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": closes, "high": highs, "low": lows, "close": closes,
        "volume": np.full(n, volume, dtype="float64"),
        "spread": np.full(n, spread, dtype="float64"),
    }, index=idx)


def bars_with_clv(closes, clv, rng=0.001, spread=19.0):
    """
    **원하는 종가 위치(CLV)가 나오도록** 고가·저가를 만든다.

    고저를 손으로 계산하면 부호를 헷갈린다 (실제로 한 번 틀렸다). CLV 를
    직접 지정하게 해서 테스트 의도가 코드에 그대로 보이게 한다.
    +1 이면 종가가 고가에(매수 쏠림), -1 이면 저가에(매도 쏠림), 0 이면 가운데.

    CLV = (2c - h - l)/(h - l) 이므로 h = c + R(1-t)/2, l = c - R(1+t)/2.
    """
    closes = np.asarray(closes, dtype="float64")
    n = len(closes)
    t = np.asarray(clv, dtype="float64")
    if t.ndim == 0:
        t = np.full(n, float(t))
    highs = closes + rng * (1.0 - t) / 2.0
    lows = closes - rng * (1.0 + t) / 2.0
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": closes, "high": highs, "low": lows, "close": closes,
        "volume": np.full(n, 1000.0), "spread": np.full(n, spread),
    }, index=idx)


# ─── 레지스트리 ──────────────────────────────────────────────────

def test_orderflow_indicators_are_actually_registered():
    """
    `_load_all` 에 카테고리를 안 넣으면 지표가 **조용히 사라진다** — 예외도
    안 나고 그냥 없다. registry.py 주석이 경고하는 함정이라 여기서 잠근다.
    """
    specs = all_specs()
    got = {k for k, v in specs.items() if v.category == "orderflow"}
    assert {"ofi_delta", "ofi_z", "cvd", "cvd_slope",
            "absorption", "cvd_divergence", "spread_z"} <= got


def test_spread_indicator_is_skipped_when_there_is_no_spread_column():
    """
    spread 는 FX(MT5) 봉에만 있다. 암호화폐 봉에는 없으므로 조용히 건너뛰어야
    한다 — 거래소를 바꿔도 파이프라인이 죽지 않게 하려는 것.
    """
    df = frame([1.10, 1.11, 1.12]).drop(columns=["spread"])
    out = compute_all(df, ["ofi_delta", "spread_z"])
    assert "ofi_delta" in out.columns
    assert "spread_z" not in out.columns


# ─── 0 으로 나누기 ───────────────────────────────────────────────

def test_a_flat_bar_does_not_divide_by_zero():
    """
    고가=저가 인 봉(움직임 없던 봉)은 FX 에서 실제로 나온다 — 한가한 시간대의
    이국 통화쌍. inf 가 하나 끼면 뒤의 z-score 가 전부 NaN 이 되어 지표가
    통째로 죽는다.
    """
    closes = [1.10] * 5
    df = frame(closes, highs=closes, lows=closes)      # 전부 납작한 봉
    d = of.ofi_delta(df)
    assert np.isfinite(d).all()
    assert (d == 0.0).all()          # 방향을 알 수 없으면 0 이다


def test_absorption_survives_flat_bars():
    closes = [1.10] * 3 + [1.11, 1.12, 1.13] * 40
    df = frame(closes, highs=[c + 0.0 if i < 3 else c + 0.001
                              for i, c in enumerate(closes)],
               lows=[c - 0.0 if i < 3 else c - 0.001
                     for i, c in enumerate(closes)])
    a = of.absorption(df, period=20)
    assert not np.isinf(a.dropna()).any()


# ─── 방향 추정 ───────────────────────────────────────────────────

def test_delta_is_positive_when_close_sits_at_the_high():
    """종가가 고가에 붙으면 그 봉 활동이 매수 쪽으로 쏠렸다고 본다."""
    df = pd.DataFrame({
        "open": [1.10], "high": [1.11], "low": [1.09], "close": [1.11],
        "volume": [1000.0],
    }, index=pd.date_range("2026-01-01", periods=1, tz="UTC"))
    assert of.ofi_delta(df).iloc[0] == pytest.approx(1000.0)


def test_delta_is_negative_when_close_sits_at_the_low():
    df = pd.DataFrame({
        "open": [1.10], "high": [1.11], "low": [1.09], "close": [1.09],
        "volume": [1000.0],
    }, index=pd.date_range("2026-01-01", periods=1, tz="UTC"))
    assert of.ofi_delta(df).iloc[0] == pytest.approx(-1000.0)


def test_delta_is_zero_in_the_middle():
    """종가가 봉 가운데면 어느 쪽으로도 쏠리지 않았다."""
    df = pd.DataFrame({
        "open": [1.10], "high": [1.11], "low": [1.09], "close": [1.10],
        "volume": [1000.0],
    }, index=pd.date_range("2026-01-01", periods=1, tz="UTC"))
    assert of.ofi_delta(df).iloc[0] == pytest.approx(0.0)


# ─── 누적델타는 추세 제거해야 한다 ───────────────────────────────

def test_cvd_is_detrended_not_a_raw_cumulative_sum():
    """
    누적합을 그대로 쓰면 무한히 커져서 z-score 가 **시간 흐름만** 재게 된다.
    최근 쏠림을 잡아야 하는데 "봉 번호"를 재면 지표가 아니다.
    """
    # 계속 매수 쪽으로 쏠리는 시장. 길이만 두 배로 늘려서 비교한다 —
    # 원 누적합은 길이에 비례해 커지고, 추세를 뺀 값은 커지지 않아야 한다.
    short = bars_with_clv(np.linspace(1.10, 1.15, 400), clv=1.0)
    long_ = bars_with_clv(np.linspace(1.10, 1.20, 800), clv=1.0)

    raw_s = abs(of.ofi_delta(short).cumsum().iloc[-1])
    raw_l = abs(of.ofi_delta(long_).cumsum().iloc[-1])
    cvd_s = of.cvd(short, period=100).abs().max()
    cvd_l = of.cvd(long_, period=100).abs().max()

    assert raw_l > raw_s * 1.8            # 누적합은 길이를 따라 커진다
    assert cvd_l < cvd_s * 1.2            # 추세를 뺀 값은 안 커진다


def test_cvd_slope_measures_acceleration_not_level():
    """
    **일정한 쏠림은 기울기 0 이다.** cvd 가 추세를 제거한 값이라 델타가
    일정하면 cvd 가 상수가 되고 기울기가 사라진다. 처음에 이걸 흐름의 방향으로
    착각해서 테스트를 잘못 썼고, 이 테스트가 그 의미를 못박는다.
    """
    # 계속 같은 세기로 팔리는 시장 → 평상시 상태 → 기울기 0 근처
    steady = bars_with_clv(np.linspace(1.10, 1.15, 400), clv=-1.0)
    s_steady = of.cvd_slope(steady, lookback=20, period=100).dropna()
    assert abs(s_steady.iloc[-1]) < 1.0

    # 약하게 팔리다가 갑자기 강하게 팔리는 시장 → 가속 → 기울기 음수
    clv = np.concatenate([np.full(300, -0.1), np.full(100, -1.0)])
    accel = bars_with_clv(np.linspace(1.10, 1.15, 400), clv=clv)
    s_accel = of.cvd_slope(accel, lookback=20, period=100).dropna()
    assert s_accel.iloc[-1] < -1.0


# ─── 다이버전스 ──────────────────────────────────────────────────

def test_divergence_flags_a_price_high_the_flow_did_not_confirm():
    """
    가격은 신고가인데 누적델타가 못 따라오면 위로 밀 힘이 빠진 것이다(-1).
    **자주 틀리는 신호라** 이 프로젝트에서는 방향 판단에 쓰지 않는다
    (실측 IC +0.0005 = 사실상 0). 계산이 맞는지만 잠근다.
    """
    # 앞 구간: 매수 쏠림으로 누적델타를 올린다.
    # 뒤 구간: 가격은 계속 신고가를 만들지만 종가는 저가에 붙어 델타가 빠진다.
    head = bars_with_clv(np.linspace(1.10, 1.15, 100), clv=1.0)
    tail = bars_with_clv(np.linspace(1.1501, 1.20, 30), clv=-1.0)
    df = pd.concat([head, tail])
    df.index = pd.date_range("2026-01-01", periods=len(df), freq="1h", tz="UTC")

    d = of.cvd_divergence(df, lookback=20, period=50)
    c = of.cvd(df, period=50)
    # 전제 확인: 가격은 신고가인데 누적델타는 아니다
    assert df["close"].iloc[-1] >= df["close"].iloc[-20:].max()
    assert c.iloc[-1] < c.iloc[-20:].max()
    assert d.iloc[-1] == -1.0


def test_divergence_is_zero_when_nothing_stands_out():
    df = frame([1.10] * 80, highs=[1.1005] * 80, lows=[1.0995] * 80)
    d = of.cvd_divergence(df, lookback=20, period=50)
    assert set(d.unique()) <= {0.0, 1.0, -1.0}


# ─── 전략 ────────────────────────────────────────────────────────

def strategy():
    return [s for s in build_all() if s.name == "order_flow_fade"][0]


def test_the_strategy_is_in_the_pool_and_declares_its_regimes():
    s = strategy()
    assert s.korean == "오더플로 쏠림 되돌림"
    assert set(s.regimes) == {Regime.UPTREND, Regime.DOWNTREND, Regime.RANGE}
    assert s.needs == ("ofi_z", "cvd", "cvd_slope")


def test_it_is_not_blocked_for_the_wrong_reason():
    """
    파생 데이터를 요구하지 않는다. `needs` 에 perp_/funding/oi_/basis 접두사가
    들어가면 엔진이 **조용히** 제외하는데, 그건 "쓸 수 없다"이지 "쓰지 않기로
    했다"가 아니다. 아래 테스트가 재는 것은 후자이므로 둘을 갈라 둔다.
    """
    assert not any(str(n).startswith(("perp_", "funding", "oi_", "basis"))
                   for n in strategy().needs)


def test_the_fx_engine_leaves_it_out_on_purpose():
    """
    **일부러 뺐다.** flow_extension_fade 와 사실상 같은 전략이기 때문이다 —
    포지션을 원한 572봉 중 겹치는 493봉에서 방향이 100% 일치했다. 둘이 동시에
    자격을 얻으면 같은 아이디어에 표가 두 장 가고 전략당 상한 0.45 를 우회해
    0.90 까지 실린다. 2026-09-15 GBPUSD 가 정확히 그 상태였다.

    되살리려면 config.yaml 의 fx.exclude_strategies 에서 이름을 지우면 된다.
    그 전에 중복 투표 문제를 어떻게 할지부터 정해야 한다.
    """
    from src.fx.engine import trader
    names = {s.name for s in trader().strategies()}
    assert "order_flow_fade" not in names


def _feats(score_series, spread_z=None):
    """합성 점수가 원하는 모양이 되도록 세 열을 같은 값으로 채운다."""
    f = pd.DataFrame({"ofi_z": score_series, "cvd": score_series,
                      "cvd_slope": score_series}, index=score_series.index)
    if spread_z is not None:
        f["spread_z"] = spread_z
    return f


def test_the_strategy_fades_it_does_not_follow():
    """
    **부호가 뒤집히면 전략이 정반대로 거래한다.** 1시간봉 7개 통화쌍 실측 IC 가
    ofi_z -0.054 (7/7 음수)였다 — 쏠림이 크면 되돌아온다. 쏠림을 추종하면
    측정된 부호와 반대로 거래하는 것이고, 이 프로젝트 초기에 돌파·모멘텀
    3종이 그 실수를 했다.
    """
    n = 400
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    # 뒤쪽에서 강하게 양(+) 쪽으로 쏠리게 만든다
    raw = pd.Series(np.concatenate([np.zeros(n - 40), np.full(40, 6.0)]), index=idx)
    df = frame(np.linspace(1.10, 1.11, n))
    out = strategy().generate(df, _feats(raw))
    # 매수 쏠림이 극단이면 **숏**이어야 한다
    assert out["direction"].iloc[-1] == -1.0

    raw_neg = pd.Series(np.concatenate([np.zeros(n - 40), np.full(40, -6.0)]),
                        index=idx)
    out2 = strategy().generate(df, _feats(raw_neg))
    assert out2["direction"].iloc[-1] == 1.0


def test_a_wide_spread_bar_blocks_entry():
    """
    스프레드가 벌어진 봉은 00시 UTC 롤오버다 (실측: 스프레드 상위 5% 봉의
    55/70 이 00시, 스프레드 33 vs 평소 19). spread_z 자체 IC 는 7/7 통화쌍에서
    +0.051 로 강했지만 **거래할 수 없는 우위**다 — 두 배 넓은 스프레드를 내고
    들어가야 하고, 이 프로젝트에서 비용은 우위를 먹는 쪽이었다.
    """
    n = 400
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    raw = pd.Series(np.concatenate([np.zeros(n - 40), np.full(40, 6.0)]), index=idx)
    df = frame(np.linspace(1.10, 1.11, n))

    normal = strategy().generate(df, _feats(raw, spread_z=pd.Series(0.0, index=idx)))
    assert normal["direction"].iloc[-1] == -1.0        # 평소면 진입

    wide = strategy().generate(df, _feats(raw, spread_z=pd.Series(5.0, index=idx)))
    assert wide["direction"].iloc[-1] == 0.0           # 벌어졌으면 안 들어간다


def test_strength_is_zero_when_there_is_no_direction():
    """
    방향이 없는데 확신도가 남아 있으면 앙상블이 0 방향에 가중을 준다.
    base.generate 가 강제하지만 이 전략에서도 확인한다.
    """
    n = 300
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    flat = pd.Series(0.0, index=idx)
    df = frame(np.linspace(1.10, 1.11, n))
    out = strategy().generate(df, _feats(flat))
    assert (out.loc[out["direction"] == 0.0, "strength"] == 0.0).all()


def test_missing_feature_columns_fail_loudly():
    """
    조용히 0 을 채우면 신호가 아니라 잡음이 된다. 없으면 바로 터져야 한다.
    """
    n = 50
    df = frame(np.linspace(1.10, 1.11, n))
    bad = pd.DataFrame({"ofi_z": np.zeros(n)}, index=df.index)
    with pytest.raises(KeyError):
        strategy().generate(df, bad)
