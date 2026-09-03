"""
평균회귀(fade) 전략 4종.

`scripts/screen_features.py` 결과가 방향을 정했다. 1시간봉에서 상위 정보계수(IC)가
**전부 음수**였다 — dc_pos −0.037, bb_pct_b −0.036, rsi −0.033, st_dir −0.034,
ema_spread −0.032, vwap_dev −0.037, pvt −0.040.

음의 IC는 "값이 높으면 미래 수익률이 낮다"는 뜻이다. 즉 위로 뻗은 것은 내려온다.
초기 전략 풀의 돌파·모멘텀 3종(돈치안 돌파 / EMA 크로스 / 켈트너 스퀴즈)은
이 부호와 **반대 방향**으로 거래하고 있었다. 여기 있는 전략들은 IC가 가리키는
방향으로 거래한다.

각 전략은 상관된 특징 여러 개를 합성해서 쓴다. 단일 특징의 IC가 0.03~0.04 수준이라
하나로는 잡음에 묻히고, 같은 계열을 평균하면 잡음이 상쇄된다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators.core import zscore
from ..regime import Regime
from .base import Strategy, empty_signals, latch, register, scale
from .library import _hold_until


def _fade_state(score: pd.Series, entry: float, exit_at: float) -> tuple[pd.Series, pd.Series]:
    """
    과열 점수가 극단에 닿으면 반대 방향, 중심으로 돌아오면 청산.
    (롱 상태, 숏 상태) 를 돌려준다. 점수가 높을 때 숏이다 — IC 부호가 음수이므로.
    """
    long_state = _hold_until(score <= -entry, score >= -exit_at)
    short_state = _hold_until(score >= entry, score <= exit_at)
    return long_state, short_state & ~long_state


def _composite_z(feats: pd.DataFrame, cols: list[str], period: int) -> pd.Series:
    """여러 특징의 z-점수 평균. 같은 계열의 잡음을 상쇄시킨다."""
    parts = [zscore(feats[c], period) for c in cols if c in feats.columns]
    if not parts:
        raise KeyError(f"합성에 필요한 열이 없습니다: {cols}")
    return pd.concat(parts, axis=1).mean(axis=1)


@register
class RangePositionFade(Strategy):
    name = "range_position_fade"
    korean = "채널 위치 되돌림"
    # 추세 국면 안에서도 IC가 강하게 음수다 (상승추세 dc_pos −0.064).
    # 그래서 추세·횡보 모두에서 발언권을 준다.
    regimes = (Regime.UPTREND, Regime.DOWNTREND, Regime.RANGE)
    needs = ("dc_pos", "bb_pct_b", "kc_pos", "williams_r")

    @classmethod
    def defaults(cls) -> dict:
        return {"entry": 0.35, "exit_at": 0.05}

    def _signals(self, df, feats):
        p = self.params
        out = empty_signals(df.index)

        # 네 특징 모두 '구간 안에서의 위치'다. 0~1 로 맞춘 뒤 중심(0.5)에서의 편차로.
        pos = pd.concat([
            feats["dc_pos"].clip(0, 1),
            feats["bb_pct_b"].clip(0, 1),
            feats["kc_pos"].clip(0, 1),
            (feats["williams_r"] + 100.0) / 100.0,
        ], axis=1).mean(axis=1)
        score = pos - 0.5          # +0.5 = 상단 극단, −0.5 = 하단 극단

        long_state, short_state = _fade_state(score, p["entry"], p["exit_at"])
        out.loc[long_state, "direction"] = 1.0
        out.loc[short_state, "direction"] = -1.0

        conf = scale(score.abs(), p["entry"], 0.5)
        out["strength"] = latch(conf, long_state) + latch(conf, short_state)
        return out


@register
class VwapReversion(Strategy):
    name = "vwap_reversion"
    korean = "VWAP 회귀"
    regimes = (Regime.UPTREND, Regime.DOWNTREND, Regime.RANGE)
    needs = ("vwap_dev",)

    @classmethod
    def defaults(cls) -> dict:
        # vwap_dev 의 IC는 장기 지평에서도 유지된다 (24봉 −0.036) → 오래 들고 간다
        return {"period": 200, "entry": 1.5, "exit_at": 0.25}

    def _signals(self, df, feats):
        p = self.params
        out = empty_signals(df.index)
        score = zscore(feats["vwap_dev"], p["period"])

        long_state, short_state = _fade_state(score, p["entry"], p["exit_at"])
        out.loc[long_state, "direction"] = 1.0
        out.loc[short_state, "direction"] = -1.0

        conf = scale(score.abs(), p["entry"], 3.0)
        out["strength"] = latch(conf, long_state) + latch(conf, short_state)
        return out


@register
class FlowExtensionFade(Strategy):
    name = "flow_extension_fade"
    korean = "플로 소진 되돌림"
    regimes = (Regime.UPTREND, Regime.DOWNTREND, Regime.RANGE)
    needs = ("pvt", "cmf", "ad_line", "force_index")

    @classmethod
    def defaults(cls) -> dict:
        return {"period": 200, "entry": 1.3, "exit_at": 0.2}

    def _signals(self, df, feats):
        p = self.params
        out = empty_signals(df.index)
        # 거래량 가중 플로 계열. IC가 −0.032 ~ −0.040 으로 이 집합에서 가장 강했다.
        score = _composite_z(feats, ["pvt", "cmf", "ad_line", "force_index"], p["period"])

        long_state, short_state = _fade_state(score, p["entry"], p["exit_at"])
        out.loc[long_state, "direction"] = 1.0
        out.loc[short_state, "direction"] = -1.0

        conf = scale(score.abs(), p["entry"], 2.8)
        out["strength"] = latch(conf, long_state) + latch(conf, short_state)
        return out


@register
class TrendExtensionFade(Strategy):
    name = "trend_extension_fade"
    korean = "추세 과열 되돌림"
    # 추세 국면 전용. 추세를 따라가는 대신 추세 안의 과열을 되돌린다.
    regimes = (Regime.UPTREND, Regime.DOWNTREND)
    needs = ("ppo", "ema_spread", "price_vs_ema200", "adx")

    @classmethod
    def defaults(cls) -> dict:
        return {"period": 200, "entry": 1.4, "exit_at": 0.2, "min_adx": 20.0}

    def _signals(self, df, feats):
        p = self.params
        out = empty_signals(df.index)
        score = _composite_z(feats, ["ppo", "ema_spread", "price_vs_ema200"], p["period"])
        # 추세가 실제로 있을 때만 — 방향 없는 구간의 과열은 의미가 없다
        score = score.where(feats["adx"] >= p["min_adx"], 0.0)

        long_state, short_state = _fade_state(score, p["entry"], p["exit_at"])
        out.loc[long_state, "direction"] = 1.0
        out.loc[short_state, "direction"] = -1.0

        conf = scale(score.abs(), p["entry"], 2.8)
        out["strength"] = latch(conf, long_state) + latch(conf, short_state)
        return out
