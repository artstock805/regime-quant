"""
역행 대응 — 들고 있는 포지션이 불리하게 갈 때 무엇을 할지.

기본 동작은 "손절 거리(1R)만큼 역행하면 손절"이다. 이 모듈을 켜면 그 자리에서
손절하는 대신 **대응한다.**

    마팅게일   손절 거리 1배마다 직전의 multiplier 배 수량으로 같은 방향에 추가한다.
               최대 max_adds 회. 평균 단가가 가격을 따라 내려와, 조금만 되돌려도
               묶음 전체가 처음 노렸던 수익을 낸다.
    양방 대응  추가할 자리에 왔는데 방금 마감한 봉이 **급변동**이면(최근 평균의
               shock_mult 배 이상 크게, 불리한 쪽으로) 물량을 늘리지 않는다. 묶음과
               같은 수량의 반대 포지션을 열어 손실을 그 자리에 묶는다.
    최종 손절  마지막 추가 뒤 한 단계 더 역행하면 묶음 전체가 손절된다. 이 손절은
               첫 진입 때부터 브로커에 걸려 있다 — 앱이 꺼져 있어도 지켜진다.

**최악의 손실은 1R 이 아니다.** 첫 진입은 손절 거리 1배에 잔고의 risk_per_trade
(1%)를 거는 크기 그대로다. 2배씩 3회 추가하고 최종 손절에 닿으면
1×4 + 2×3 + 4×2 + 8×1 = **26R (잔고의 약 26%)** 를 잃는다. 양방 대응이 열리면
그 시점의 손실에서 멈춘다.

**봉 마감에만 판단한다** (엔진의 규칙과 같다). 추가 진입은 그 단계 가격이 아니라
단계를 넘긴 봉의 마감 가격에 나간다. 한 봉에 여러 단계를 건너뛰어도 한 번에 한
단계만 추가한다.

**역할은 주문 태그 끝에 남긴다.** `m0` 첫 진입, `m1`~`m9` 추가, `h` 양방 대응.
기억에만 두면 재시작한 순간 묶음을 잃는다 — 엔진은 매 바퀴 MT5 의 포지션에서
묶음을 다시 읽는다. 표식이 없는 포지션(이 기능 전에 연 것)은 예전처럼 다룬다.
단계 간격은 첫 진입의 손절에서 거꾸로 읽으므로(손절 = 진입가 ∓ (max_adds+1) 간격),
**묶음이 열려 있는 동안 max_adds 를 바꾸면 간격이 틀어진다.**
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

import pandas as pd

BASE = "m0"
HEDGE = "h"
_ROLE = re.compile(r"(?:m(\d)|h)$")
SHOCK_LOOKBACK = 14


@dataclass(frozen=True)
class Rules:
    enabled: bool = False
    max_adds: int = 3
    multiplier: float = 2.0
    shock_mult: float = 2.0

    def as_dict(self) -> dict:
        return asdict(self)


def rules(cfg) -> Rules:
    def get(key: str, default):
        return cfg.get(f"fx.defense.{key}", default)

    return Rules(enabled=bool(get("enabled", False)),
                 # 단계는 태그에 한 자리 숫자로 남는다
                 max_adds=min(max(int(get("max_adds", 3)), 0), 9),
                 multiplier=max(float(get("multiplier", 2.0)), 1.0),
                 shock_mult=float(get("shock_mult", 2.0)))


def role(pos: dict) -> tuple[str, int] | None:
    """이 포지션의 역할 ("base"|"add"|"hedge", 단계). 표식이 없으면 None."""
    found = _ROLE.search(str(pos.get("comment", "")).strip())
    if not found:
        return None
    if found.group(1) is None:
        return "hedge", 0
    level = int(found.group(1))
    return ("base", 0) if level == 0 else ("add", level)


def round_step(volume: float, step: float) -> float:
    """랏 단위로 내림한다 (broker 와 같은 규칙 — 올리면 의도보다 커진다)."""
    if step <= 0:
        return volume
    return round(math.floor(volume / step + 1e-9) * step, 8)


def shock(df: pd.DataFrame, mult: float) -> int:
    """
    마지막 마감봉이 급변동이면 그 방향(+1 상승, -1 하락), 아니면 0.

    급변동 = 봉의 고저 폭이 그 앞 14봉 평균 폭의 mult 배 이상. 천천히 밀린 것과
    한 봉에 크게 밀린 것을 가른다 — 앞의 것은 되돌림을 기다릴 만하지만 뒤의 것에
    물량을 두 배로 싣는 것은 떨어지는 칼을 잡는 일이다.
    """
    if mult <= 0 or len(df) < SHOCK_LOOKBACK + 1:
        return 0
    spans = (df["high"] - df["low"]).to_numpy("float64")
    usual = spans[-SHOCK_LOOKBACK - 1:-1].mean()
    if not usual > 0 or spans[-1] < mult * usual:
        return 0
    move = float(df["close"].iloc[-1] - df["open"].iloc[-1])
    return (move > 0) - (move < 0)


@dataclass(frozen=True)
class Basket:
    """한 종목에 쌓인 묶음 — 첫 진입, 추가 진입들, (있으면) 양방 대응."""
    base: dict
    adds: tuple[dict, ...]
    hedge: dict | None
    rules: Rules

    @property
    def side(self) -> str:
        return self.base["side"]

    @property
    def sign(self) -> int:
        return 1 if self.side == "LONG" else -1

    @property
    def final_stop(self) -> float:
        return float(self.base["stop_loss"])

    @property
    def step(self) -> float:
        """단계 간격 = 손절 거리 1배 (1R 의 가격)."""
        return abs(float(self.base["entry_price"]) - self.final_stop) / (self.rules.max_adds + 1)

    @property
    def level(self) -> int:
        """
        지금까지 간 단계 — 열려 있는 추가분 중 **가장 높은 차수.** 개수로 세면
        중간 것이 사라졌을 때(1차는 닫히고 2차만 남음) 다음 주문이 이미 있는
        태그(m2)로 나가 매번 거부되고, 대응이 조용히 멈춘다.
        """
        return max((role(p)[1] for p in self.adds), default=0)

    @property
    def legs(self) -> tuple[dict, ...]:
        return (self.base, *self.adds)

    @property
    def volume(self) -> float:
        return sum(float(p["volume"]) for p in self.legs)

    def reached(self, price: float) -> bool:
        """가격이 다음 추가 단계에 닿았는가."""
        trigger = float(self.base["entry_price"]) - self.sign * (self.level + 1) * self.step
        # 단계 가격은 나눗셈으로 나온다. 정확히 닿은 가격이 소수 오차로 빠지지 않게.
        return self.sign * (trigger - price) >= -1e-9

    def tag(self, suffix: str) -> str:
        return str(self.base["comment"]).strip()[:-len(BASE)] + suffix

    def next_volume(self, volume_step: float) -> float:
        return round_step(float(self.base["volume"]) * self.rules.multiplier ** (self.level + 1),
                          volume_step)

    def target(self, reward_ratio: float, added: tuple[float, float]) -> float:
        """
        (수량, 가격)을 추가한 뒤의 묶음 목표가 — **처음 노렸던 수익 금액**이 나는 가격.

        첫 진입은 손절 거리의 reward_ratio 배를 노렸다. 묶음이 커진 뒤에도 같은
        돈을 벌면 되므로, 평균 단가에서 그 거리를 (첫 수량 / 전체 수량)만큼만 간다.
        """
        fills = [(float(p["volume"]), float(p["entry_price"])) for p in self.legs] + [added]
        total = sum(v for v, _ in fills)
        average = sum(v * price for v, price in fills) / total
        return average + self.sign * reward_ratio * self.step * float(self.base["volume"]) / total


def basket_of(held: list[dict], rule: Rules) -> Basket | None:
    """
    이 종목에 들고 있는 것에서 묶음을 읽는다. 대응할 묶음이 아니면 None —
    기능이 꺼져 있거나, 첫 진입 표식이 없거나(예전 포지션), 손절이 없다.
    """
    if not rule.enabled:
        return None
    bases = [p for p in held if role(p) == ("base", 0)]
    if len(bases) != 1:
        return None
    base = bases[0]
    if not base.get("stop_loss") or not base.get("entry_price"):
        return None
    adds = sorted((p for p in held if (role(p) or ("", 0))[0] == "add"
                   and p["side"] == base["side"]), key=lambda p: role(p)[1])
    hedges = [p for p in held if role(p) == ("hedge", 0) and p["side"] != base["side"]]
    return Basket(base, tuple(adds), hedges[0] if hedges else None, rule)
