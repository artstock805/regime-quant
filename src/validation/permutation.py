"""
순열 검정 — 신호 타이밍이 무작위보다 나은가.

t검정은 "기대값이 0인가"를 묻는다. 순열 검정은 더 날카로운 질문을 한다:
**같은 신호를 시간 순서만 뒤섞어도 이 성과가 나오는가.**

우리 신호가 실제로 시장의 무언가를 잡아낸다면, 순서를 흩으면 성과가 사라져야 한다.
흩어도 비슷하게 나온다면 그 성과는 신호가 아니라 손절·목표·사이징 같은
**실행 구조가 만들어낸 산물**이다. 그건 우위가 아니다.

**블록 순열을 쓴다.** 신호를 한 봉씩 무작위로 섞으면 자기상관이 파괴되어
귀무분포가 지나치게 쉬워진다(포지션이 한 봉만 유지되는 신호는 비용만 낸다).
연속된 블록 단위로 순서를 바꾸면 신호의 지속성과 진입 빈도를 보존한 채
**시장 국면과의 정렬만** 끊을 수 있다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..backtest.engine import ExecParams, run
from ..config import Costs


@dataclass(frozen=True)
class PermutationResult:
    observed: float
    null_mean: float
    null_sd: float
    p_value: float
    n_permutations: int
    block_size: int
    percentile: float
    null_samples: list[float] = field(default_factory=list)

    @property
    def significant(self) -> bool:
        return bool(np.isfinite(self.p_value) and self.p_value < 0.05)

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.p_value):
            return "판정불가"
        if self.p_value < 0.05:
            return "신호가 무작위보다 낫다"
        return "무작위 순서와 구분되지 않는다"

    def as_dict(self) -> dict:
        d = asdict(self)
        # 표본 전체를 리포트에 넣으면 무거우니 히스토그램용으로만 남긴다
        d["null_samples"] = [round(x, 6) for x in self.null_samples[:500]]
        d["significant"] = self.significant
        d["verdict"] = self.verdict
        return d


def block_permute(
    signals: pd.DataFrame, block_size: int, rng: np.random.Generator
) -> pd.DataFrame:
    """
    신호를 블록 단위로 순서를 바꾼다. 인덱스는 원본 그대로 유지한다.

    direction 과 strength 를 **함께** 옮긴다. 따로 섞으면 방향과 확신도의
    짝이 깨져서 존재할 수 없는 신호가 만들어진다.
    """
    n = len(signals)
    if block_size <= 0 or block_size >= n:
        raise ValueError(f"블록 크기가 부적절합니다: {block_size} (봉 {n})")

    starts = np.arange(0, n, block_size)
    order = rng.permutation(len(starts))
    pieces = []
    for k in order:
        lo = starts[k]
        pieces.append(signals.iloc[lo:lo + block_size].to_numpy())
    arr = np.concatenate(pieces)[:n]
    return pd.DataFrame(arr, index=signals.index, columns=signals.columns)


def permutation_test(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    atr: pd.Series,
    *,
    interval: str,
    costs: Costs,
    params: ExecParams,
    initial_equity: float,
    minute_df: pd.DataFrame | None = None,
    regimes: pd.Series | None = None,
    n_permutations: int = 200,
    block_size: int | None = None,
    metric: str = "r_multiple",
    seed: int = 0,
    progress=None,
) -> PermutationResult:
    """
    관측 기대값이 블록 순열 귀무분포에서 어디에 있는지 본다.

    block_size 기본값은 신호의 평균 지속 길이 × 3 이다. 지속 길이보다 짧게 잡으면
    신호가 잘려 진입 빈도가 달라지고, 너무 길게 잡으면 순열이 거의 원본이 된다.
    """
    base = run(df, signals, atr, interval=interval, costs=costs, params=params,
               initial_equity=initial_equity, minute_df=minute_df, regimes=regimes)
    if base.trades.empty:
        nan = float("nan")
        return PermutationResult(nan, nan, nan, nan, 0, 0, nan)
    observed = float(base.trades[metric].mean())

    if block_size is None:
        block_size = _default_block(signals)

    rng = np.random.default_rng(seed)
    nulls: list[float] = []
    for k in range(n_permutations):
        perm = block_permute(signals, block_size, rng)
        r = run(df, perm, atr, interval=interval, costs=costs, params=params,
                initial_equity=initial_equity, minute_df=minute_df, regimes=regimes)
        if not r.trades.empty:
            nulls.append(float(r.trades[metric].mean()))
        if progress and (k + 1) % 20 == 0:
            progress(k + 1, n_permutations)

    if not nulls:
        nan = float("nan")
        return PermutationResult(observed, nan, nan, nan, 0, block_size, nan)

    arr = np.array(nulls)
    # 단측 p값. +1 보정은 관측값 자신을 귀무표본에 포함시키는 표준 처리로,
    # p=0 이 나오는 것을 막는다 (유한 표본에서 p=0 은 성립할 수 없다).
    p = (int((arr >= observed).sum()) + 1) / (len(arr) + 1)
    return PermutationResult(
        observed=observed,
        null_mean=float(arr.mean()),
        null_sd=float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
        p_value=float(p),
        n_permutations=len(arr),
        block_size=block_size,
        percentile=float((arr < observed).mean() * 100),
        null_samples=[float(x) for x in arr],
    )


def _default_block(signals: pd.DataFrame) -> int:
    """신호가 한 방향으로 유지되는 평균 길이 × 3."""
    d = signals["direction"].to_numpy()
    changes = np.flatnonzero(np.diff(d) != 0)
    if changes.size == 0:
        return max(8, len(d) // 50)
    runs = np.diff(np.concatenate([[0], changes + 1, [len(d)]]))
    mean_run = float(runs.mean())
    return int(max(8, min(len(d) // 10, round(mean_run * 3))))
