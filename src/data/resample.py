"""
OHLCV 리샘플링.

신호용 5m/15m/1h 는 바이낸스에서 네이티브로 받는 게 정확하므로 기본은 그쪽을 쓴다.
이 모듈은 (1) 4h 처럼 네이티브로 안 받아둔 상위 타임프레임을 파생시킬 때,
(2) 멀티 타임프레임 정렬을 검증할 때 쓴다.

봉 라벨은 바이낸스와 같은 규칙 — 봉의 **여는 시각**으로 라벨링한다(label='left').
이걸 틀리면 상위 타임프레임 값이 한 봉 앞당겨져서 미래를 훔쳐보는 백테스트가 된다.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from .binance import INTERVAL_MS, interval_timedelta

# 리샘플 시 열마다 다른 집계 규칙. volume 계열은 합, 가격은 OHLC.
AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "quote_volume": "sum",
    "trades": "sum",
    "taker_buy_base": "sum",
}

# pandas offset alias. 'H'/'T'는 pandas 2.2에서 폐기되어 'h'/'min'을 쓴다.
_PANDAS_RULE = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h", "1d": "1D",
}


def infer_interval(df: pd.DataFrame) -> timedelta:
    """인덱스 간격의 최빈값으로 원본 인터벌을 추정한다."""
    if len(df) < 2:
        raise ValueError("인터벌을 추정하려면 최소 2봉이 필요합니다")
    return df.index.to_series().diff().dropna().mode().iloc[0]


def resample_ohlcv(
    df: pd.DataFrame,
    target: str,
    drop_partial: bool = True,
) -> pd.DataFrame:
    """
    df 를 target 인터벌로 올린다. 하위 → 상위만 가능하고, 배수 관계여야 한다.

    drop_partial=True 면 원본이 다 채우지 못한 마지막 봉을 버린다. 예를 들어 1h 로
    올릴 때 원본 5m 이 11:00~11:40 까지만 있으면 11:00 봉은 아직 미완성이므로 버린다.
    이 봉을 남기면 종가가 실제와 다른 봉으로 신호를 만들게 된다.
    """
    if df.empty:
        return df.copy()
    if target not in _PANDAS_RULE:
        raise ValueError(f"지원하지 않는 타깃 인터벌: {target}")

    src = infer_interval(df)
    tgt = interval_timedelta(target)
    if tgt < src:
        raise ValueError(f"상위 → 하위 리샘플은 불가능합니다 ({src} → {target})")
    if INTERVAL_MS[target] % int(src.total_seconds() * 1000) != 0:
        raise ValueError(f"배수 관계가 아닙니다 ({src} → {target})")

    cols = {k: v for k, v in AGG.items() if k in df.columns}
    out = df.resample(_PANDAS_RULE[target], label="left", closed="left").agg(cols)
    # 원본에 구멍이 있으면 전부 NaN인 봉이 생긴다 — 없는 봉이므로 지운다
    out = out.dropna(subset=["open", "high", "low", "close"])

    if drop_partial and not out.empty:
        bars_per_target = INTERVAL_MS[target] // int(src.total_seconds() * 1000)
        counts = df.resample(_PANDAS_RULE[target], label="left", closed="left").size()
        last = out.index[-1]
        if counts.get(last, 0) < bars_per_target:
            out = out.iloc[:-1]

    return out


def build_timeframes(
    base: pd.DataFrame,
    targets: list[str],
    drop_partial: bool = True,
) -> dict[str, pd.DataFrame]:
    """하나의 기준 프레임에서 여러 타임프레임을 한 번에 파생시킨다."""
    return {t: resample_ohlcv(base, t, drop_partial=drop_partial) for t in targets}


def align_higher(lower: pd.DataFrame, higher: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    상위 타임프레임 값을 하위 타임프레임 인덱스에 붙인다. 미래 참조 없이.

    핵심: 상위 봉은 **닫힌 뒤에야** 하위 봉에서 쓸 수 있다. 1h 봉 11:00 은 12:00 에
    닫히므로, 11:05 의 5m 봉이 볼 수 있는 최신 1h 값은 10:00 봉이다.
    merge_asof 로 그냥 붙이면 11:00 봉이 11:05 에 보여서 미래를 훔쳐본다.
    그래서 상위 인덱스를 '닫힘 시각'으로 옮긴 뒤에 붙인다.
    """
    if higher.empty or lower.empty:
        return pd.DataFrame(index=lower.index, columns=columns, dtype="float64")

    span = infer_interval(higher)
    shifted = higher[columns].copy()
    shifted.index = shifted.index + span  # 닫힘 시각으로 이동

    joined = pd.merge_asof(
        pd.DataFrame(index=lower.index).reset_index(),
        shifted.reset_index(),
        left_on="open_time", right_on="open_time",
        direction="backward",
    ).set_index("open_time")
    return joined[columns]
