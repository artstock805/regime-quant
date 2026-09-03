"""
초기 전략 6종.

레짐별로 성격이 다른 전략을 배치했다.
    추세 국면  — 돈치안 돌파 / EMA 크로스+ADX / 켈트너 스퀴즈 이탈
    횡보 국면  — 볼린저 평균회귀 / RSI 다이버전스
    고변동     — VixFix 반등 (공포 정점 되돌림)

전략마다 유효 국면을 선언하고, 앙상블은 현재 국면에 발언권이 있는 전략만 투표에 넣는다.
어느 전략이 실제로 값을 하는지는 백테스트가 정하고, 못 하는 전략은 가중치 0으로 빠진다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators.core import crossover, crossunder
from ..regime import Regime
from .base import Strategy, empty_signals, held, latch, register, scale


@register
class DonchianBreakout(Strategy):
    name = "donchian_breakout"
    korean = "돈치안 채널 돌파"
    regimes = (Regime.UPTREND, Regime.DOWNTREND)
    needs = ("dc_upper", "dc_lower", "atr", "volume_ratio", "adx")

    @classmethod
    def defaults(cls) -> dict:
        return {"min_volume_ratio": 1.2, "min_adx": 20.0, "hold_bars": 6}

    def _signals(self, df, feats):
        p = self.params
        out = empty_signals(df.index)

        confirmed = (feats["volume_ratio"] >= p["min_volume_ratio"]) & (feats["adx"] >= p["min_adx"])
        # 돈치안 상·하단은 직전 봉까지의 극값이므로 현재 종가와 비교해도 미래 참조가 아니다
        broke_up = (df["close"] > feats["dc_upper"]) & confirmed
        broke_down = (df["close"] < feats["dc_lower"]) & confirmed

        # 돌파 후 hold_bars 동안 포지션을 유지한다. 채널 안으로 돌아오기 전까지.
        long_state = held(broke_up, p["hold_bars"]) & (df["close"] > feats["dc_lower"])
        short_state = held(broke_down, p["hold_bars"]) & (df["close"] < feats["dc_upper"])

        out.loc[long_state, "direction"] = 1.0
        out.loc[short_state & ~long_state, "direction"] = -1.0

        # 확신도: 돌파 폭을 ATR로 정규화 + 거래량 확인
        atr = feats["atr"].replace(0, np.nan)
        depth_up = scale((df["close"] - feats["dc_upper"]) / atr, 0.0, 1.0)
        depth_dn = scale((feats["dc_lower"] - df["close"]) / atr, 0.0, 1.0)
        vol_conf = scale(feats["volume_ratio"], p["min_volume_ratio"], 2.5)

        conf_up = 0.6 * depth_up + 0.4 * vol_conf
        conf_dn = 0.6 * depth_dn + 0.4 * vol_conf
        out["strength"] = latch(conf_up, long_state) + latch(conf_dn, short_state & ~long_state)
        return out


@register
class EmaCrossAdx(Strategy):
    name = "ema_cross_adx"
    korean = "EMA 크로스 + ADX"
    regimes = (Regime.UPTREND, Regime.DOWNTREND)
    needs = ("ema_spread", "adx", "adx_slope", "di_plus", "di_minus")

    @classmethod
    def defaults(cls) -> dict:
        return {"min_adx": 22.0, "min_spread": 0.05}

    def _signals(self, df, feats):
        p = self.params
        out = empty_signals(df.index)

        strong = feats["adx"] >= p["min_adx"]
        # ADX가 오르는 중일 때만 — 추세가 소진되는 국면에서 뒤늦게 타지 않도록
        rising = feats["adx_slope"] > 0
        gate = strong & rising

        long_state = gate & (feats["ema_spread"] >= p["min_spread"]) & (feats["di_plus"] > feats["di_minus"])
        short_state = gate & (feats["ema_spread"] <= -p["min_spread"]) & (feats["di_minus"] > feats["di_plus"])

        out.loc[long_state, "direction"] = 1.0
        out.loc[short_state, "direction"] = -1.0

        adx_conf = scale(feats["adx"], p["min_adx"], 45.0)
        spread_conf = scale(feats["ema_spread"].abs(), p["min_spread"], 1.5)
        out["strength"] = (0.5 * adx_conf + 0.5 * spread_conf).where(
            long_state | short_state, 0.0
        )
        return out


@register
class KeltnerSqueezeRelease(Strategy):
    name = "keltner_squeeze"
    korean = "켈트너 스퀴즈 이탈"
    regimes = (Regime.UPTREND, Regime.DOWNTREND)
    needs = ("squeeze_on", "kc_upper", "kc_lower", "kc_width", "volume_ratio")

    @classmethod
    def defaults(cls) -> dict:
        return {"lookback": 12, "min_volume_ratio": 1.1, "hold_bars": 8}

    def _signals(self, df, feats):
        p = self.params
        out = empty_signals(df.index)

        squeezed = feats["squeeze_on"] > 0.5
        # 최근에 압축돼 있었고 지금은 풀렸다
        released = held(squeezed, p["lookback"]).shift(1).fillna(False) & ~squeezed
        confirmed = released & (feats["volume_ratio"] >= p["min_volume_ratio"])

        up = confirmed & (df["close"] > feats["kc_upper"])
        down = confirmed & (df["close"] < feats["kc_lower"])

        long_state = held(up, p["hold_bars"]) & (df["close"] > feats["kc_lower"])
        short_state = held(down, p["hold_bars"]) & (df["close"] < feats["kc_upper"])

        out.loc[long_state, "direction"] = 1.0
        out.loc[short_state & ~long_state, "direction"] = -1.0

        # 압축이 깊었다가 풀릴수록 확신이 높다.
        # 앵커는 BTC/ETH/SOL 2년 5분봉 실측 분위수 (p05 0.32 ~ p50 0.89).
        tight = scale(-feats["kc_width"], -0.90, -0.35)
        vol_conf = scale(feats["volume_ratio"], p["min_volume_ratio"], 2.5)
        conf = 0.5 * tight + 0.5 * vol_conf
        out["strength"] = latch(conf, long_state) + latch(conf, short_state & ~long_state)
        return out


@register
class BollingerMeanReversion(Strategy):
    name = "bollinger_reversion"
    korean = "볼린저 평균회귀"
    regimes = (Regime.RANGE,)
    needs = ("bb_pct_b", "bb_width", "adx")

    @classmethod
    def defaults(cls) -> dict:
        return {"entry_band": 0.05, "exit_band": 0.5, "max_adx": 25.0, "min_width": 0.5}

    def _signals(self, df, feats):
        p = self.params
        out = empty_signals(df.index)
        pct_b = feats["bb_pct_b"]

        # 추세가 붙어 있으면 평균회귀를 하지 않는다 — 밴드 밖에서 계속 밖으로 간다
        calm = (feats["adx"] <= p["max_adx"]) & (feats["bb_width"] >= p["min_width"])

        touched_low = (pct_b <= p["entry_band"]) & calm
        touched_high = (pct_b >= 1.0 - p["entry_band"]) & calm

        # 밴드를 건드린 뒤 중심선까지 돌아오면 청산. 상태를 앞으로 채워 표현한다.
        long_state = _hold_until(touched_low, pct_b >= p["exit_band"])
        short_state = _hold_until(touched_high, pct_b <= p["exit_band"])

        out.loc[long_state, "direction"] = 1.0
        out.loc[short_state & ~long_state, "direction"] = -1.0

        # 진입 시점 pct_b 실측: 중위 -0.038, p05 -0.255 → 그 범위를 0~1로 편다
        conf_long = scale(-pct_b, -p["entry_band"], 0.25)
        conf_short = scale(pct_b, 1.0 - p["entry_band"], 1.25)
        out["strength"] = latch(conf_long, long_state) + latch(conf_short, short_state & ~long_state)
        return out


@register
class RsiDivergenceStrategy(Strategy):
    name = "rsi_divergence"
    korean = "RSI 다이버전스"
    regimes = (Regime.RANGE,)
    needs = ("rsi_divergence", "rsi", "adx")

    @classmethod
    def defaults(cls) -> dict:
        return {"min_signal": 0.15, "max_adx": 28.0, "hold_bars": 10}

    def _signals(self, df, feats):
        p = self.params
        out = empty_signals(df.index)
        sig = feats["rsi_divergence"]
        calm = feats["adx"] <= p["max_adx"]

        # 강세 다이버전스는 과매도에서, 약세는 과매수에서만 인정한다
        bull = (sig >= p["min_signal"]) & (feats["rsi"] <= 45.0) & calm
        bear = (sig <= -p["min_signal"]) & (feats["rsi"] >= 55.0) & calm

        long_state = held(bull, p["hold_bars"])
        short_state = held(bear, p["hold_bars"])

        out.loc[long_state, "direction"] = 1.0
        out.loc[short_state & ~long_state, "direction"] = -1.0
        # |rsi_divergence| 실측: 중위 0.151, p90 0.386 → 상한을 1.0이 아니라 0.40으로
        conf = scale(sig.abs(), p["min_signal"], 0.40)
        out["strength"] = latch(conf, long_state) + latch(conf, short_state & ~long_state)
        return out


@register
class VixFixReversal(Strategy):
    name = "vixfix_reversal"
    korean = "VixFix 반등"
    regimes = (Regime.HIGH_VOL, Regime.RANGE)
    needs = ("vix_fix", "rsi", "atr_pct")

    @classmethod
    def defaults(cls) -> dict:
        return {"percentile": 90.0, "lookback": 200, "hold_bars": 12}

    def _signals(self, df, feats):
        p = self.params
        out = empty_signals(df.index)

        # VixFix 가 자기 과거 대비 상위 구간 = 공포 정점
        from ..indicators.core import percent_rank
        rank = percent_rank(feats["vix_fix"], p["lookback"])
        panic = rank >= p["percentile"]
        # 정점을 지나 종가가 되돌아서기 시작한 봉
        reclaim = panic.shift(1).fillna(False) & (df["close"] > df["close"].shift(1))

        long_state = held(reclaim, p["hold_bars"])
        out.loc[long_state, "direction"] = 1.0
        out["strength"] = latch(scale(rank, p["percentile"], 100.0), long_state)
        return out


# ─── 도구 ────────────────────────────────────────────────────────

def _hold_until(enter: pd.Series, exit_cond: pd.Series) -> pd.Series:
    """
    enter 가 True인 봉부터 exit_cond 가 True가 되는 봉 전까지 True를 유지한다.

    앞으로 채우는 방식이지만 미래를 보지 않는다 — 각 시점의 값은 그 시점까지의
    enter/exit 이력만으로 결정된다. cummax 로 벡터화한다.
    """
    e = enter.fillna(False).to_numpy()
    x = exit_cond.fillna(False).to_numpy()
    idx = np.arange(len(e))

    last_enter = np.where(e, idx, -1)
    np.maximum.accumulate(last_enter, out=last_enter)
    last_exit = np.where(x, idx, -1)
    np.maximum.accumulate(last_exit, out=last_exit)

    # 마지막 진입이 마지막 청산보다 뒤면 보유 중
    return pd.Series((last_enter >= 0) & (last_enter > last_exit), index=enter.index)
