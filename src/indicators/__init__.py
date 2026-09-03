"""
지표 풀.

    from src.indicators import compute_all, summary
    feats = compute_all(df)                      # 등록된 지표 전부
    feats = compute_all(df, ["rsi", "adx"])      # 지표 이름으로
    feats = compute_all(df, ["bb_pct_b"])        # 출력 열 이름으로도 된다
"""

from .registry import (
    REGISTRY, Spec, all_specs, by_category, column_owners, compute, compute_all,
    resolve, summary,
)

__all__ = [
    "REGISTRY", "Spec", "all_specs", "by_category", "column_owners",
    "compute", "compute_all", "resolve", "summary",
]
