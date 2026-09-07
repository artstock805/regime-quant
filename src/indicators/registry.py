"""
지표 레지스트리.

전략과 레짐 분류기는 지표를 이름으로 꺼내 쓴다. 하드코딩된 import 대신 레지스트리를
두는 이유는 두 가지다.
  · 지표 풀 전체를 열거할 수 있어야 앙상블이 후보를 훑을 수 있다.
  · 지표를 추가할 때 전략 코드를 건드리지 않게 된다.

각 지표 함수는 (df, **params) 를 받아 Series 하나 또는 DataFrame 하나를 돌려준다.
DataFrame을 돌려주는 경우 열 이름이 spec.outputs 와 정확히 일치해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import pandas as pd

OHLCV = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class Spec:
    name: str
    fn: Callable[..., pd.Series | pd.DataFrame]
    category: str
    params: dict = field(default_factory=dict)
    outputs: tuple[str, ...] = ()
    needs: tuple[str, ...] = OHLCV

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.outputs or (self.name,)


REGISTRY: dict[str, Spec] = {}


def register(
    name: str,
    category: str,
    params: dict | None = None,
    outputs: Iterable[str] | None = None,
    needs: Iterable[str] = OHLCV,
):
    def deco(fn: Callable[..., pd.Series | pd.DataFrame]):
        if name in REGISTRY:
            raise ValueError(f"지표 이름 중복: {name}")
        REGISTRY[name] = Spec(
            name=name, fn=fn, category=category,
            params=dict(params or {}),
            outputs=tuple(outputs or (name,)),
            needs=tuple(needs),
        )
        return fn
    return deco


def _load_all() -> None:
    """
    카테고리 모듈을 임포트해서 데코레이터가 실행되게 한다.

    `if not REGISTRY` 같은 조건으로 감싸면 안 된다. 누군가 trend 만 직접 임포트해서
    레지스트리를 먼저 채우면 나머지 카테고리를 건너뛰고, 지표가 소리 없이 사라진다.
    파이썬이 모듈을 캐시하므로 매번 불러도 @register 는 한 번만 실행된다.
    """
    from . import (  # noqa: F401
        derivatives, momentum, orderflow, trend, volatility, volume,
    )


def all_specs() -> dict[str, Spec]:
    _load_all()
    return dict(REGISTRY)


def column_owners() -> dict[str, str]:
    """출력 열 이름 → 그 열을 만드는 지표 이름."""
    out: dict[str, str] = {}
    for spec in all_specs().values():
        for col in spec.output_names:
            out[col] = spec.name
    return out


def resolve(names: Iterable[str]) -> list[str]:
    """
    지표 이름과 출력 열 이름을 섞어서 받아 지표 이름 목록으로 정리한다.

    전략은 needs 를 출력 열('bb_pct_b')로 선언하고 지표는 이름('bollinger')으로
    등록되어 있다. 둘을 호출부마다 손으로 변환하게 두면 반드시 어긋나므로
    여기서 한 번에 흡수한다. 순서는 유지하고 중복은 제거한다.
    """
    specs = all_specs()
    owners = column_owners()
    out: list[str] = []
    unknown: list[str] = []
    for name in names:
        target = name if name in specs else owners.get(name)
        if target is None:
            unknown.append(name)
        elif target not in out:
            out.append(target)
    if unknown:
        raise KeyError(f"지표도 출력 열도 아닌 이름: {unknown}")
    return out


def by_category() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for spec in all_specs().values():
        out.setdefault(spec.category, []).append(spec.name)
    return {k: sorted(v) for k, v in sorted(out.items())}


def compute(df: pd.DataFrame, name: str, **overrides) -> pd.Series | pd.DataFrame:
    """지표 하나를 계산한다."""
    specs = all_specs()
    if name not in specs:
        raise KeyError(f"등록되지 않은 지표: {name}")
    spec = specs[name]
    missing = [c for c in spec.needs if c not in df.columns]
    if missing:
        raise KeyError(f"{name}: 입력 열이 없습니다 {missing}")

    params = {**spec.params, **overrides}
    out = spec.fn(df, **params)

    if isinstance(out, pd.Series):
        return out.rename(spec.output_names[0])
    if set(out.columns) != set(spec.output_names):
        raise ValueError(
            f"{name}: 반환 열 {list(out.columns)} 이 선언 {list(spec.output_names)} 과 다릅니다"
        )
    return out


def compute_all(
    df: pd.DataFrame,
    names: Iterable[str] | None = None,
    skip_missing_inputs: bool = True,
) -> pd.DataFrame:
    """
    지표 풀 전체(또는 지정한 일부)를 계산해 넓은 특징 프레임으로 돌려준다.

    names 에는 지표 이름('bollinger')과 출력 열 이름('bb_pct_b')을 섞어 넣어도 된다.

    skip_missing_inputs=True 면 taker_buy_base 처럼 소스에 따라 없을 수 있는 열을
    요구하는 지표는 조용히 건너뛴다. 거래소를 바꿔도 파이프라인이 죽지 않게 하려는 것.
    """
    specs = all_specs()
    wanted = resolve(names) if names is not None else list(specs)
    pieces: list[pd.Series | pd.DataFrame] = []

    for name in wanted:
        spec = specs[name]
        if skip_missing_inputs and any(c not in df.columns for c in spec.needs):
            continue
        pieces.append(compute(df, name))

    if not pieces:
        return pd.DataFrame(index=df.index)
    return pd.concat(pieces, axis=1)


def summary() -> pd.DataFrame:
    """지표 풀 현황 표 — 몇 개가 어떤 카테고리에 있는지 확인용."""
    rows = []
    for spec in all_specs().values():
        rows.append({
            "name": spec.name,
            "category": spec.category,
            "outputs": len(spec.output_names),
            "params": ", ".join(f"{k}={v}" for k, v in spec.params.items()),
        })
    return pd.DataFrame(rows).sort_values(["category", "name"]).reset_index(drop=True)
