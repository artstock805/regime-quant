"""
파생 시장 전략 — 캔들 밖 정보로 짠 것들.

만든 근거는 `scripts/orthogonality.py` 의 실측이다. 단독 IC 만 보면 파생은
캔들을 못 넘는다(0.0308 vs 0.0441). 그런데 **캔들 상위 8개 특징의 유효 차원이
1.50 / 8** 이었다 — 평균 상관 0.777, 최대 0.983. 여덟 개가 사실상 한 개다.

그래서 물어야 할 것은 "IC 가 더 높은가"가 아니라 "이미 아는 것을 걷어내도 남는가"
였고, 답은 남는다는 것이었다.

    perp_taker_ls   단독 +0.0158 → 부분 +0.0203 (겹침 0.120)  더 강해진다
    oi_zscore       단독 -0.0179 → 부분 -0.0145 (겹침 0.142)
    oi_change       단독 -0.0180 → 부분 -0.0126 (겹침 0.165)
    oi_price_regime 단독 -0.0361 → 부분 -0.0088 (겹침 0.766)  가격 모멘텀이었다

캔들 8개에 파생 11개를 더하면 R² 가 0.182% → 0.322% 로 오른다. 무작위 11개를
더했을 때가 +0.007%p 이므로 **과적합 기준선의 19배**다.

경고 — 부호만 보고 방향을 정하지 않았다
    이 프로젝트는 이미 한 번 그 실수를 했다. 특징 IC 가 전부 음수인 걸 보고
    평균회귀 전략 4종을 짰는데, 승률은 54.8% 로 높았지만 기대R 은 -0.0158 로
    모멘텀(+0.0325)보다 나빴다. **IC 는 평균 방향만 알려 주고 손익 분포의
    비대칭을 반영하지 못한다.** 그래서 여기서도 IC 부호는 가설의 출발점일 뿐이고,
    채택 여부는 백테스트의 기대R 과 검증 스위트가 정한다.

입력 열이 없으면(2021-12 이전, 또는 파생을 안 받은 경우) 전략이 터진다.
`SuiteConfig(use_derivatives=True)` 로 켜야 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..regime import Regime
from .base import Strategy, empty_signals, held, latch, register, scale


@register
class TakerFlowMomentum(Strategy):
    """
    테이커 흐름 추종.

    부분 IC 가 단독보다 **커지는** 유일한 특징이다(+0.0158 → +0.0203). 캔들과 겹침이
    0.120 뿐이라, 가격이 아직 반영하지 못한 체결 압력을 본다는 해석이 가능하다.

    시장가로 밀어붙이는 쪽이 이긴다는 가설이므로 방향은 순추종이다. 추세 국면에만
    발언권을 준 이유는, 횡보에서 체결 압력은 대개 되돌려지기 때문이다.
    """

    name = "taker_flow_momentum"
    korean = "테이커 흐름 추종"
    regimes = (Regime.UPTREND, Regime.DOWNTREND)
    needs = ("perp_taker_ls", "atr_pct", "adx")

    @classmethod
    def defaults(cls) -> dict:
        return {"entry_z": 0.15, "min_adx": 18.0, "hold_bars": 6, "z_period": 168}

    def _signals(self, df, feats):
        p = self.params
        out = empty_signals(df.index)

        flow = feats["perp_taker_ls"]
        # 종목·시기마다 기준선이 달라 절대값으로 문턱을 잡을 수 없다.
        # 자기 최근 분포로 표준화한다 — 과거만 보므로 인과적이다.
        mu = flow.rolling(p["z_period"], min_periods=p["z_period"]).mean()
        sd = flow.rolling(p["z_period"], min_periods=p["z_period"]).std(ddof=0)
        z = (flow - mu) / sd.replace(0, np.nan)

        trending = feats["adx"] >= p["min_adx"]
        buy = (z >= p["entry_z"]) & trending
        sell = (z <= -p["entry_z"]) & trending

        long_state = held(buy, p["hold_bars"]) & (z > 0)
        short_state = held(sell, p["hold_bars"]) & (z < 0)

        out.loc[long_state, "direction"] = 1.0
        out.loc[short_state & ~long_state, "direction"] = -1.0

        conf = scale(z.abs(), p["entry_z"], 2.0)
        out["strength"] = latch(conf, long_state) + latch(conf, short_state & ~long_state)
        return out


@register
class CrowdedPositionFade(Strategy):
    """
    쏠린 포지션 되돌림.

    OI 가 자기 분포 상단이고 펀딩까지 한쪽으로 치우쳐 있으면, 그 방향으로 이미
    들어갈 사람은 다 들어갔다는 뜻이다. 청산 연쇄의 연료가 쌓인 상태다.

    OI 만으로는 방향을 알 수 없다 — 롱이 쌓였는지 숏이 쌓였는지는 **펀딩이
    말해 준다.** 펀딩이 양수면 롱이 숏에게 돈을 내고 있으니 롱 쏠림이다.
    그래서 두 조건을 AND 로 묶는다. OI 단독 전략을 만들지 않은 이유다.

    횡보와 고변동에만 발언권을 준다. 추세 국면에서 쏠림을 역행하면 추세에 깔린다.
    """

    name = "crowded_position_fade"
    korean = "쏠린 포지션 되돌림"
    regimes = (Regime.RANGE, Regime.HIGH_VOL)
    needs = ("oi_zscore", "funding_zscore", "bb_pct_b")

    @classmethod
    def defaults(cls) -> dict:
        return {"oi_z": 1.0, "funding_z": 0.8, "hold_bars": 8}

    def _signals(self, df, feats):
        p = self.params
        out = empty_signals(df.index)

        crowded = feats["oi_zscore"] >= p["oi_z"]
        long_crowded = crowded & (feats["funding_zscore"] >= p["funding_z"])
        short_crowded = crowded & (feats["funding_zscore"] <= -p["funding_z"])

        # 쏠림이 풀리는 쪽으로 선다
        short_state = held(long_crowded, p["hold_bars"]) & (feats["oi_zscore"] > 0)
        long_state = held(short_crowded, p["hold_bars"]) & (feats["oi_zscore"] > 0)

        out.loc[long_state, "direction"] = 1.0
        out.loc[short_state & ~long_state, "direction"] = -1.0

        crowd_conf = scale(feats["oi_zscore"], p["oi_z"], 3.0)
        fund_conf = scale(feats["funding_zscore"].abs(), p["funding_z"], 3.0)
        conf = 0.5 * crowd_conf + 0.5 * fund_conf
        out["strength"] = latch(conf, long_state) + latch(conf, short_state & ~long_state)
        return out


@register
class OpenInterestThrust(Strategy):
    """
    신규 자금 유입 추종.

    OI 가 빠르게 늘면서 가격이 움직이면 **새 돈이 들어오는 것**이고, OI 가 줄면서
    움직이면 포지션이 청산되는 것이다. 방향이 같아도 지속성이 다르다 —
    청산발 움직임은 연료가 없어 이어지기 어렵다.

    `oi_price_regime` 은 단독 IC 가 가장 높았지만(0.0361) 캔들을 걷어내면
    0.0088 로 무너졌다(겹침 0.766). 즉 그 지표는 가격 모멘텀을 다시 부른 것에
    가깝다. 그래서 여기서는 **OI 변화 자체**를 조건으로 쓰고 방향만 가격에서
    받아, 겹치는 부분을 줄였다.
    """

    name = "oi_thrust"
    korean = "미결제약정 급증 추종"
    regimes = (Regime.UPTREND, Regime.DOWNTREND)
    needs = ("oi_change", "oi_zscore", "atr", "adx")

    @classmethod
    def defaults(cls) -> dict:
        return {"min_oi_change": 0.02, "min_adx": 20.0, "hold_bars": 6, "lookback": 24}

    def _signals(self, df, feats):
        p = self.params
        out = empty_signals(df.index)

        # 같은 구간의 가격 변화. shift 없이 pct_change 는 과거만 본다.
        price_chg = df["close"].pct_change(p["lookback"])
        oi_up = feats["oi_change"] >= p["min_oi_change"]
        trending = feats["adx"] >= p["min_adx"]

        thrust_up = oi_up & (price_chg > 0) & trending
        thrust_dn = oi_up & (price_chg < 0) & trending

        long_state = held(thrust_up, p["hold_bars"]) & (price_chg > 0)
        short_state = held(thrust_dn, p["hold_bars"]) & (price_chg < 0)

        out.loc[long_state, "direction"] = 1.0
        out.loc[short_state & ~long_state, "direction"] = -1.0

        conf = 0.5 * scale(feats["oi_change"], p["min_oi_change"], 0.10) \
             + 0.5 * scale(feats["oi_zscore"].abs(), 0.5, 2.5)
        out["strength"] = latch(conf, long_state) + latch(conf, short_state & ~long_state)
        return out
