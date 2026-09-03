"""
전략 인터페이스.

전략은 **원하는 포지션 상태**를 봉마다 내놓는다. 진입 이벤트가 아니라 상태다.
    direction  -1(숏) / 0(무포지션) / +1(롱)
    strength   0~1, 그 판단의 확신도. 앙상블 투표 가중에 쓴다.

이벤트(교차 발생)로 만들면 그 봉을 놓쳤을 때 진입 기회가 사라지고, 재진입 규칙을
전략마다 따로 써야 한다. 상태로 두면 엔진이 전환만 감지해서 처리하면 된다.

**신호는 봉이 닫힌 뒤 계산되고, 체결은 다음 봉 시가에서 일어난다.** 신호가 나온 봉의
종가로 체결하는 것은 미래 참조다 — 종가를 알기 전에는 주문을 낼 수 없다.
엔진이 이 지연을 강제하므로 전략은 신경 쓰지 않아도 된다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from ..regime import Regime

SIGNAL_COLUMNS = ("direction", "strength")


class Strategy(ABC):
    """모든 전략의 기반."""

    name: str = ""
    korean: str = ""
    # 이 전략이 유효하다고 보는 국면. 앙상블은 현재 국면에 맞는 전략만 투표에 넣는다.
    regimes: tuple[Regime, ...] = ()
    # 필요한 특징열. 없으면 바로 터진다.
    needs: tuple[str, ...] = ()

    def __init__(self, **params):
        self.params = {**self.defaults(), **params}

    @classmethod
    def defaults(cls) -> dict:
        return {}

    @abstractmethod
    def _signals(self, df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
        """direction/strength 두 열을 돌려준다."""

    def generate(self, df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.needs if c not in feats.columns]
        if missing:
            raise KeyError(f"{self.name}: 특징열이 없습니다 {missing}")

        out = self._signals(df, feats)
        if list(out.columns) != list(SIGNAL_COLUMNS):
            raise ValueError(f"{self.name}: 반환 열이 {list(out.columns)} 입니다")

        out["direction"] = out["direction"].fillna(0.0).clip(-1, 1).astype("float64")
        out["strength"] = out["strength"].fillna(0.0).clip(0.0, 1.0).astype("float64")
        # 방향이 없으면 확신도도 0이어야 한다 — 앙상블이 0방향에 가중을 주지 않도록
        out.loc[out["direction"] == 0.0, "strength"] = 0.0
        return out

    def __repr__(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"


REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    if not cls.name:
        raise ValueError(f"{cls.__name__}: name 이 비어 있습니다")
    if cls.name in REGISTRY:
        raise ValueError(f"전략 이름 중복: {cls.name}")
    if not cls.regimes:
        raise ValueError(f"{cls.name}: 유효 국면(regimes)을 선언해야 합니다")
    REGISTRY[cls.name] = cls
    return cls


def all_strategies() -> dict[str, type[Strategy]]:
    """
    등록된 전략 전부.

    로딩을 `if not REGISTRY` 로 감싸면 안 된다. 누군가 library 를 직접 임포트해서
    레지스트리를 먼저 채우면 mean_reversion 을 건너뛰고, 전략이 소리 없이 사라진다.
    파이썬이 모듈을 캐시하므로 매번 임포트해도 @register 는 한 번만 실행된다.
    """
    from . import derivatives, library, mean_reversion  # noqa: F401
    return dict(REGISTRY)


def build_all(**overrides) -> list[Strategy]:
    """등록된 전략을 기본 파라미터로 전부 인스턴스화한다."""
    return [cls(**overrides.get(name, {})) for name, cls in sorted(all_strategies().items())]


def strategies_for(regime: Regime | str | None) -> list[str]:
    """해당 국면에서 발언권이 있는 전략 이름."""
    if regime is None:
        return []
    return sorted(
        name for name, cls in all_strategies().items()
        if any(str(r) == str(regime) for r in cls.regimes)
    )


def required_features(strategies: Iterable[Strategy]) -> list[str]:
    """전략들이 요구하는 특징열 합집합 — 필요한 지표만 계산하기 위해."""
    out: set[str] = set()
    for s in strategies:
        out.update(s.needs)
    return sorted(out)


def strategy_table() -> pd.DataFrame:
    rows = []
    for name, cls in sorted(all_strategies().items()):
        rows.append({
            "name": name,
            "korean": cls.korean,
            "regimes": ", ".join(str(r) for r in cls.regimes),
            "needs": len(cls.needs),
        })
    return pd.DataFrame(rows)


# ─── 전략 구현에 쓰는 공용 도구 ───────────────────────────────────

def scale(s: pd.Series, lo: float, hi: float) -> pd.Series:
    """lo~hi 를 0~1로 자른다. 확신도를 만드는 데 쓴다."""
    return ((s - lo) / (hi - lo)).clip(0.0, 1.0)


def empty_signals(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {"direction": np.zeros(len(index)), "strength": np.zeros(len(index))}, index=index
    )


def held(cond: pd.Series, bars: int) -> pd.Series:
    """조건이 지난 bars 봉 안에 한 번이라도 True였는지. 과거만 본다."""
    return cond.fillna(False).rolling(bars, min_periods=1).max().astype(bool)


def latch(value: pd.Series, active: pd.Series) -> pd.Series:
    """
    active 가 True인 구간에서, 그 구간이 시작된 봉의 value 를 유지한다.

    확신도는 **판단이 내려진 시점의 값**이어야 한다. 보유 중에는 진입 조건이 이미
    깨져 있으므로, 현재 봉으로 확신도를 다시 계산하면 0으로 붕괴하고 그 전략은
    실력과 무관하게 앙상블에서 계속 저평가된다.

    구간 시작 봉의 값을 ffill 로 끌고 가므로 미래를 보지 않는다.
    """
    a = active.fillna(False).astype(bool)
    starts = a & ~a.shift(1, fill_value=False)
    return value.where(starts).ffill().where(a, 0.0).fillna(0.0)
