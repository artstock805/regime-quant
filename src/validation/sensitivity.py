"""
민감도 검정 — 결론이 파라미터 한 칸에 매달려 있는가.

두 가지를 본다.

**1) 파라미터 고원(plateau) 검사**
    최적값 주변이 평평한 고원이면 그 값은 실재하는 구조를 잡은 것이다.
    최적값만 뾰족하고 옆칸이 급락하면 잡음에 맞춘 것이다. 고원이 아니면
    실전에서 그 뾰족한 봉우리에 다시 서 있을 확률이 없다.

**2) 비용 손익분기 배수**
    비용을 0.25배~4배로 바꿔가며 순기대값이 0을 지나는 지점을 찾는다.
    손익분기 배수가 1.0 미만이면 **실제 비용보다 싸야만 통하는 전략**이다.
    1.0 을 넉넉히 넘어야 슬리피지 추정이 조금 틀려도 견딘다.

두 검정 모두 `runner` 를 받는다 — 파라미터를 받아 거래 표를 돌려주는 함수다.
단독 백테스트든 walk-forward 든 같은 검정을 붙일 수 있다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .stats import t_test

# 파라미터 값 → 거래 표
Runner = Callable[[float], pd.DataFrame]


@dataclass(frozen=True)
class PlateauResult:
    param: str
    chosen: float
    best: float
    best_value: float
    chosen_value: float
    plateau_width: int
    n_points: int
    is_plateau: bool
    at_edge: bool
    table: list[dict]

    @property
    def verdict(self) -> str:
        if self.at_edge:
            return "최적값이 탐색 범위 끝 — 범위를 넓혀야 한다"
        if not self.is_plateau:
            return "고원이 아니다 — 잡음에 맞춘 값일 수 있다"
        return "고원 위에 있다"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict
        return d


def parameter_plateau(
    runner: Runner,
    param: str,
    values: list[float],
    chosen: float,
    metric: str = "r_multiple",
    tolerance: float = 0.5,
    progress=None,
) -> PlateauResult:
    """
    파라미터를 values 로 훑고 고원인지 판정한다.

    tolerance=0.5 는 "최고값의 50% 이상을 내는 칸"을 고원의 일부로 본다.
    고원 폭이 3칸 이상이면 고원으로 인정한다 — 최적값 하나와 양옆이 살아있어야
    실전에서 그 근처에 서 있어도 비슷한 결과를 기대할 수 있다.
    """
    rows = []
    for i, v in enumerate(values):
        trades = runner(v)
        st = t_test(trades[metric].to_numpy()) if not trades.empty else None
        rows.append({
            param: v,
            "거래": 0 if trades.empty else len(trades),
            "기대R": np.nan if st is None else round(st.mean, 5),
            "t": np.nan if st is None else round(st.t, 3),
        })
        if progress:
            progress(i + 1, len(values))

    tbl = pd.DataFrame(rows)
    vals = tbl["기대R"].to_numpy(dtype="float64")
    if not np.isfinite(vals).any():
        return PlateauResult(param, chosen, np.nan, np.nan, np.nan, 0,
                             len(values), False, False, tbl.to_dict("records"))

    best_i = int(np.nanargmax(vals))
    best_v = float(vals[best_i])
    chosen_i = int(np.argmin([abs(v - chosen) for v in values]))

    # 최고값의 tolerance 배 이상을 내는 **연속 구간**의 길이
    if best_v <= 0:
        width = 0            # 최고값도 음수면 고원을 논할 것이 없다
    else:
        ok = vals >= best_v * tolerance
        width = 1
        k = best_i - 1
        while k >= 0 and ok[k]:
            width += 1
            k -= 1
        k = best_i + 1
        while k < len(ok) and ok[k]:
            width += 1
            k += 1

    return PlateauResult(
        param=param, chosen=float(chosen),
        best=float(values[best_i]), best_value=best_v,
        chosen_value=float(vals[chosen_i]) if np.isfinite(vals[chosen_i]) else float("nan"),
        plateau_width=int(width), n_points=len(values),
        is_plateau=bool(width >= 3 and best_v > 0),
        at_edge=bool(best_i in (0, len(values) - 1)),
        table=tbl.to_dict("records"),
    )


@dataclass(frozen=True)
class CostSensitivity:
    multiples: list[float]
    expectancies: list[float]
    breakeven_multiple: float | None
    actual_multiple: float
    table: list[dict]

    @property
    def verdict(self) -> str:
        if self.breakeven_multiple is None:
            return ("어떤 비용 수준에서도 양수가 되지 않는다"
                    if all(e <= 0 for e in self.expectancies if np.isfinite(e))
                    else "손익분기 지점을 탐색 범위에서 찾지 못했다")
        if self.breakeven_multiple < 1.0:
            return (f"실제 비용의 {self.breakeven_multiple:.2f}배에서만 손익분기 — "
                    "실제 비용보다 싸야만 통한다")
        if self.breakeven_multiple < 1.5:
            return (f"손익분기 {self.breakeven_multiple:.2f}배 — 여유가 거의 없다. "
                    "슬리피지 추정이 조금 틀리면 뒤집힌다")
        return f"손익분기 {self.breakeven_multiple:.2f}배 — 비용 추정 오차를 견딘다"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict
        return d


def cost_sensitivity(
    runner: Runner,
    multiples: list[float] | None = None,
    metric: str = "r_multiple",
    progress=None,
) -> CostSensitivity:
    """
    비용 배수를 바꿔가며 순기대값을 재고, 0을 지나는 지점을 선형보간으로 찾는다.

    runner 는 **비용 배수**를 받아 거래 표를 돌려준다.
    """
    multiples = multiples or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    rows, exps = [], []
    for i, m in enumerate(multiples):
        trades = runner(m)
        e = float(trades[metric].mean()) if not trades.empty else float("nan")
        exps.append(e)
        rows.append({"비용배수": m, "거래": 0 if trades.empty else len(trades),
                     "순기대R": None if not np.isfinite(e) else round(e, 5)})
        if progress:
            progress(i + 1, len(multiples))

    # 양수 → 음수로 바뀌는 첫 구간에서 보간
    breakeven = None
    for (m0, e0), (m1, e1) in zip(zip(multiples, exps), zip(multiples[1:], exps[1:])):
        if np.isfinite(e0) and np.isfinite(e1) and e0 > 0 >= e1:
            breakeven = m0 + (m1 - m0) * e0 / (e0 - e1)
            break

    return CostSensitivity(
        multiples=list(multiples), expectancies=exps,
        breakeven_multiple=None if breakeven is None else float(breakeven),
        actual_multiple=1.0, table=rows,
    )
