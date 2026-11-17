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

**영상이 말한 시간봉으로 계산한다.** 영상이 "4시간봉 전략"이라고 했으면 조건을
4시간봉 위에서 본다. 엔진은 자기 봉(fx.interval, 기본 1시간)으로 돌기 때문에,
받은 봉을 영상의 봉으로 묶어서(리샘플) 계산한 뒤 엔진 봉으로 되돌린다.

    · 영상 봉 = 엔진 봉      그대로 계산한다
    · 영상 봉이 더 길다      묶어서 계산한다. hold_bars 도 **영상 봉** 기준이다
    · 영상 봉이 더 짧다      **적용할 수 없다.** 1시간봉으로는 5분봉 조건을 계산할 수
                             없다. 엉뚱한 봉으로 돌리지 않고 투표에서 뺀다
    · 영상이 말하지 않았다   엔진 봉을 쓴다 (timeframe 이 null)

**긴 봉의 판단은 그 봉이 닫힌 뒤에만 쓴다.** 4시간봉 T(00:00~04:00)의 종가는
03:00 에 열린 1시간봉이 닫혀야 정해진다. 그 값을 00:00 봉에 붙이면 세 시간
뒤의 가격으로 진입하는 백테스트가 된다. 그래서 T 의 판단을 T+3h 봉에 놓고,
다음 긴 봉이 닫힐 때까지 그 상태를 이어 간다. 아직 다 차지 않은 마지막 긴 봉은
쓰지 않는다.

긴 봉의 경계는 UTC 기준이다(4시간봉 = 0·4·8시…, 일봉 = UTC 자정). 브로커의
4시간봉·일봉은 서버 시간(GMT+2/+3) 기준이라 경계가 2~3시간 어긋난다. 영상이
본 차트와 봉 모양이 조금 다를 수 있다.

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

from ..config import Config, load_config
from ..regime import ALL_REGIMES
from ..strategies.base import Strategy, empty_signals
from . import indicators, store
from .expr import evaluate, names_in

logger = logging.getLogger(__name__)
PREFIX = "yt_"

MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
_OHLCV = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

SAME, RESAMPLED, UNSUPPORTED = "same", "resampled", "unsupported"


def timeframe_mode(timeframe: str | None, base: str) -> str:
    """
    영상의 봉(timeframe)을 엔진의 봉(base) 위에서 어떻게 다룰지.

        same         그대로 계산 (같은 봉이거나 영상이 봉을 말하지 않았다)
        resampled    영상 봉이 더 길다 — 묶어서 계산
        unsupported  영상 봉이 더 짧거나 배수가 아니다 — 계산할 수 없다
    """
    if timeframe is None:
        return SAME
    want, have = MINUTES.get(timeframe), MINUTES.get(base)
    if want is None or have is None:
        return UNSUPPORTED
    if want == have:
        return SAME
    return RESAMPLED if want > have and want % have == 0 else UNSUPPORTED


def _step_minutes(index: pd.Index) -> int | None:
    """봉 간격(분). 주말 공백이 있어도 중앙값은 정상 간격이다."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return None
    step = index.to_series().diff().median()
    return int(step.total_seconds() // 60) if pd.notna(step) else None


def _resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """
    더 긴 봉으로 묶는다. 라벨은 봉이 **여는 시각**이다 (src/data/resample.py 와 같은
    규칙). origin 을 epoch 로 고정해야 데이터가 어디서 시작하든 경계가 같다 —
    엔진이 받아 오는 창은 매 바퀴 한 봉씩 밀린다.
    """
    out = df[list(_OHLCV)].resample(f"{minutes}min", label="left", closed="left",
                                    origin="epoch").agg(_OHLCV)
    return out.dropna(subset=["close"])


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
            direction = self._direction(df)
        except Exception:
            logger.exception("%s: 신호 계산 실패 — 이번 봉은 기권합니다", self.name)
            return empty_signals(df.index)
        out = empty_signals(df.index)
        out["direction"] = direction
        out["strength"] = (direction != 0.0).astype("float64")
        return out

    def _direction(self, df: pd.DataFrame) -> pd.Series:
        """받은 봉의 인덱스에 맞춘 방향(-1/0/+1). 영상의 봉으로 계산한다."""
        have = _step_minutes(df.index)
        want = MINUTES.get(self.spec.get("timeframe"))
        if have is None or want is None or want == have:
            return self._state(df)
        if want < have or want % have:
            # 받은 봉이 영상의 봉보다 굵다. 엔진은 이런 전략을 애초에 넣지 않는다
            # (enabled_strategies). 다른 경로로 불렸을 때도 엉뚱한 봉으로 답하지 않는다.
            return pd.Series(0.0, index=df.index)

        state = self._state(_resample(df, want))
        # 긴 봉 T 의 판단은 그 봉의 마지막 작은 봉(T + want - have)이 닫혀야 알 수 있다.
        # 거기에 놓고 앞으로 채운다. 아직 다 안 찬 마지막 긴 봉은 놓일 자리가
        # 인덱스 밖(미래)이라 저절로 빠진다.
        known = state.set_axis(state.index + pd.Timedelta(minutes=want - have))
        return known.reindex(df.index, method="ffill").fillna(0.0)

    def _state(self, df: pd.DataFrame) -> pd.Series:
        """조건이 참이 된 봉부터 hold_bars 봉 동안 그 방향을 유지한다."""
        cols = indicators.columns(df, self.spec["indicators_needed"])
        long = self._condition("long", cols, df.index)
        short = self._condition("short", cols, df.index)

        trigger = pd.Series(np.nan, index=df.index)
        trigger[long & ~short] = 1.0
        trigger[short & ~long] = -1.0
        hold = int(self.params["hold_bars"])
        # ffill 은 과거 값을 앞으로 끌고 올 뿐이라 미래를 보지 않는다
        return (trigger.ffill(limit=hold - 1) if hold > 1 else trigger).fillna(0.0)


# 단기 군이 다루는 가장 짧은 봉(분). **1분봉은 넣지 않는다.** MT5 가 학습 구간만큼의
# 1분봉을 주지 않고(실측: 6개월치 요청이 실패한다), 준다 해도 28종목을 1분 안에
# 훑을 수 없다. 5분봉은 6개월치를 받고 종목당 3.1초였다(28종목 약 1분 30초).
MIN_SHORT_MINUTES = 5

MAIN_GROUP, SHORT_GROUP = "main", "short"


def strategy_group(timeframe: str | None, base: str) -> str | None:
    """
    이 전략이 들어가는 투표 군.

        main   엔진 봉(base) 이상 — 기존 전략과 함께 투표한다
        short  엔진 봉보다 짧다 — 단기 전략끼리 따로 투표하고 따로 진입·청산한다
        None   적용할 수 없다 (1분봉 등)

    **짧은 봉 전략을 기본 군에 넣지 않는 이유.** 기본 군은 엔진 봉으로 돌고, 그
    봉으로는 더 짧은 봉의 조건을 계산할 수 없다. 반대로 기본 군 전체를 짧은 봉으로
    내리면 1시간봉으로 검증한 기존 전략이 전부 다른 것이 된다. 그래서 군을 가른다.
    """
    if timeframe_mode(timeframe, base) != UNSUPPORTED:
        return MAIN_GROUP
    want, have = MINUTES.get(timeframe), MINUTES.get(base)
    if want is not None and have is not None and MIN_SHORT_MINUTES <= want < have:
        return SHORT_GROUP
    return None


def _enabled(cfg: Config, group: str) -> list[YoutubeStrategy]:
    base = str(cfg.get("fx.interval", "1h"))
    return [YoutubeStrategy(record["spec"])
            for record in store.load_all(cfg)
            if record["enabled"]
            and strategy_group(record["spec"]["timeframe"], base) == group]


def enabled_strategies(cfg: Config | None = None) -> list[YoutubeStrategy]:
    """
    기본 군 엔진이 투표에 넣을 유튜브 전략. 호출할 때마다 파일을 다시 읽으므로
    토글이 다음 바퀴에 반영된다.

    켜 두었어도 **엔진의 봉으로 계산할 수 없는 전략은 뺀다** (영상 봉이 엔진 봉보다
    짧은 경우). 그런 전략은 단기 군으로 간다 (short_strategies).
    """
    return _enabled(cfg or load_config(), MAIN_GROUP)


def short_strategies(cfg: Config | None = None) -> list[YoutubeStrategy]:
    """단기 군 엔진이 투표에 넣을 유튜브 전략 — 엔진 봉보다 짧은 시간봉의 전략."""
    return _enabled(cfg or load_config(), SHORT_GROUP)


def short_interval(cfg: Config | None = None) -> str | None:
    """
    단기 군이 도는 봉 — 켜 둔 단기 전략 중 **가장 짧은 시간봉**. 없으면 None.

    더 긴 전략(예: 5분봉으로 도는데 15분봉 전략)은 YoutubeStrategy 가 묶어서
    계산한다. 5·15·30분은 서로 배수라 항상 묶을 수 있다.
    """
    frames = [s.spec["timeframe"] for s in short_strategies(cfg)]
    return min(frames, key=MINUTES.__getitem__) if frames else None
