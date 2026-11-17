"""
유튜브에서 뽑은 전략을 이 앱의 Strategy 로 감싼다.

**레지스트리에 등록하지 않는다.** build_all() 은 백테스트·검증·성적표가 전부
쓰는 전략 풀이고, 최종 판정(129종목 7,496거래)이 그 풀에서 나왔다. 영상에서
가져온 전략이 거기 섞이면 그 숫자들이 조용히 바뀐다. 그래서 유튜브 전략은
**FX 자동매매 엔진의 투표에만** 들어간다 (FxTrader.strategies).

**투표에 들어가도 발언권은 성적이 정한다.** signal_for 가 최근 학습 구간
(기본 6개월)에서 이 전략을 단독으로 돌려 레짐별 기대 R 을 뽑는다. 거래가
20건에 못 미치거나 기대 R 이 0 이하면 가중치는 0 이다 — 켜 놓았는데 주문에
아무 영향이 없다면 고장이 아니라 그 결과다. 화면이 가중치를 보여 준다.

**조건(이벤트)을 상태로 바꾼다.** 이 앱의 전략은 봉마다 "원하는 포지션"을
내놓는데 영상의 규칙은 대개 "교차가 일어나면 진입"이다. 조건이 참이 된 봉부터
hold_bars 봉 동안 그 방향을 유지하고, 그 사이 반대 조건이 참이 되면 뒤집는다.
같은 봉에서 롱·숏 조건이 함께 참이면 판단하지 않는다.

**확신도는 1 로 고정이다.** 규칙이 참/거짓뿐이라 강약이 없다. 엔진이
calibrate_strength(FX 기본값)를 켜면 확신도를 자기 분포의 백분위로 바꾸는데,
전부 같은 값의 백분위는 약 0.5 다(실측 0.501). 즉 이 전략의 한 표는
`가중치 × 약 0.5` 로 들어간다.

**국면은 전부 선언한다.** 영상은 국면을 말하지 않는다. 어느 국면에서 값을
하는지는 레짐별 가중치가 가른다.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Config
from ..regime import ALL_REGIMES
from ..strategies.base import Strategy, empty_signals
from . import indicators, store
from .expr import evaluate, names_in

logger = logging.getLogger(__name__)
PREFIX = "yt_"


class YoutubeStrategy(Strategy):
    regimes = ALL_REGIMES
    needs = ()

    def __init__(self, spec: dict, **params) -> None:
        super().__init__(**params)
        self.spec = spec
        self.name = PREFIX + spec["strategy_name"]
        self.korean = f"유튜브 · {spec['strategy_name']}"

    @classmethod
    def defaults(cls) -> dict:
        return {"hold_bars": 6}

    def _condition(self, side: str, cols: dict, index: pd.Index) -> pd.Series:
        expr = self.spec["entry_conditions"][side]
        if not expr:
            return pd.Series(False, index=index)
        return evaluate(expr, indicators.with_previous(names_in(expr), cols), index)

    def _signals(self, df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
        """
        **계산이 터지면 기권한다.** signal_for 는 전략을 전부 돌린 뒤에야 판단하고,
        엔진은 신호가 없는 종목의 청산까지 건너뛴다. 선택 사항인 전략 하나가
        터졌다고 다른 전략의 판단과 포지션 관리가 멈추면 안 된다. 저장 전에
        합성 봉으로 한 번 돌려 보지만(spec.validate) 실제 봉은 다를 수 있다.
        """
        try:
            return self._compute(df)
        except Exception:
            logger.exception("%s: 신호 계산 실패 — 이번 봉은 기권합니다", self.name)
            return empty_signals(df.index)

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = indicators.columns(df, self.spec["indicators_needed"])
        long = self._condition("long", cols, df.index)
        short = self._condition("short", cols, df.index)

        trigger = pd.Series(np.nan, index=df.index)
        trigger[long & ~short] = 1.0
        trigger[short & ~long] = -1.0
        hold = int(self.params["hold_bars"])
        # ffill 은 과거 값을 앞으로 끌고 올 뿐이라 미래를 보지 않는다
        state = (trigger.ffill(limit=hold - 1) if hold > 1 else trigger).fillna(0.0)

        out = empty_signals(df.index)
        out["direction"] = state
        out["strength"] = (state != 0.0).astype("float64")
        return out


def enabled_strategies(cfg: Config | None = None) -> list[YoutubeStrategy]:
    """켜 둔 유튜브 전략. 호출할 때마다 파일을 다시 읽으므로 토글이 다음 바퀴에 반영된다."""
    return [YoutubeStrategy(record["spec"])
            for record in store.load_all(cfg) if record["enabled"]]
