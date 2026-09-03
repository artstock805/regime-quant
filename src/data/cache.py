"""
캔들 로컬 캐시.

(시장, 종목, 인터벌)마다 parquet 한 장을 두고 증분으로 채운다.
백테스트를 수십 번 돌리는 게 정상인 작업이라, 매번 다시 내려받으면
개발 속도가 데이터 다운로드에 묶인다.

정책
  · 캐시에 있는 구간은 다시 받지 않는다. 요청 구간에서 앞/뒤로 빠진 부분만 받는다.
  · 캐시 중간에 구멍이 있어도 앞뒤 경계만 보고 판단한다. 구멍이 의심되면
    refresh=True 로 해당 구간을 강제로 다시 받는다.
  · 저장 직전에 반드시 정렬·중복제거한다. 중복 봉은 백테스트에서 같은 신호를
    두 번 내는 사고로 이어진다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from .binance import KEEP_COLUMNS, _as_utc, fetch_klines, interval_timedelta

Logger = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def cache_path(klines_dir: str | Path, market: str, symbol: str, interval: str) -> Path:
    return Path(klines_dir) / market / f"{symbol}_{interval}.parquet"


def load_cached(path: str | Path) -> pd.DataFrame:
    """캐시를 읽는다. 없으면 빈 프레임."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=KEEP_COLUMNS)
    df = pd.read_parquet(p)
    if df.empty:
        return df
    if df.index.name != "open_time":
        raise ValueError(f"캐시 인덱스가 open_time이 아닙니다: {p}")
    # parquet 왕복 뒤에도 tz가 UTC로 유지되는지 확인한다
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone.utc)
    return df.sort_index(kind="stable")


def save_cached(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # 안정 정렬 — 같은 타임스탬프의 입력 순서를 보존해야 keep="last" 가 뜻대로 동작한다
    clean = df.sort_index(kind="stable")
    clean = clean[~clean.index.duplicated(keep="last")]
    clean.index.name = "open_time"
    clean.to_parquet(p, compression="zstd")
    return p


def get_klines(
    symbol: str,
    interval: str,
    start: datetime | str,
    end: datetime | str | None = None,
    market: str = "futures_um",
    klines_dir: str | Path = "data/klines",
    refresh: bool = False,
    log: Logger = _noop,
) -> pd.DataFrame:
    """
    캐시를 우선 쓰고 빠진 구간만 받아서 [start, end) 를 돌려준다.

    refresh=True 면 요청 구간을 캐시를 무시하고 새로 받는다. **구간 밖의 캐시는
    그대로 보존한다** — 좁은 구간에 refresh 를 걸었다고 5.9년치가 날아가면 안 된다.
    (실제로 120일 구간에 refresh 를 걸어 51,840봉이 2,855봉으로 잘린 사고가 있었다.)
    """
    start_dt = _as_utc(start)
    end_dt = _as_utc(end) if end is not None else datetime.now(timezone.utc)
    path = cache_path(klines_dir, market, symbol, interval)
    span = interval_timedelta(interval)

    # 디스크에 있는 것은 항상 읽어 둔다. refresh 는 '무엇을 다시 받을지'만 정하고,
    # '무엇을 버릴지'는 정하지 않는다.
    existing = load_cached(path)
    cached = pd.DataFrame(columns=KEEP_COLUMNS) if refresh else existing

    if cached.empty:
        missing = [(start_dt, end_dt)]
    else:
        missing = []
        # 캐시 앞쪽으로 모자란 구간
        if start_dt < cached.index[0]:
            missing.append((start_dt, cached.index[0]))
        # 캐시 뒤쪽으로 모자란 구간. 마지막 봉의 닫힘 시각부터 이어 받는다.
        tail_from = cached.index[-1] + span
        if tail_from < end_dt:
            missing.append((tail_from, end_dt))

    # 기존 캐시를 먼저 넣고 새로 받은 것을 뒤에 붙인다.
    # 중복 제거가 keep="last" 이므로 겹치는 구간은 새 데이터가 이긴다.
    frames = [existing] if not existing.empty else []
    for lo, hi in missing:
        log(f"[{symbol} {interval}] 신규 구간 {lo:%Y-%m-%d} → {hi:%Y-%m-%d}")
        fresh = fetch_klines(symbol, interval, lo, hi, market=market, log=log)
        if not fresh.empty:
            frames.append(fresh)

    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS)

    # 안정 정렬이라야 뒤에 붙인 새 데이터가 겹치는 구간에서 이긴다.
    # 기본 quicksort 로는 순서가 뒤섞여 옛 값이 남는 일이 생긴다.
    merged = pd.concat(frames).sort_index(kind="stable")
    merged = merged[~merged.index.duplicated(keep="last")]

    if missing:
        # merged 는 항상 existing 을 포함하므로 캐시는 줄어들 수 없다.
        # 별도 안전장치를 두지 않는 이유: 도달 불가능한 검사이고, 중복 정리처럼
        # 정당하게 줄어드는 경우에 잘못 발동할 수 있다.
        save_cached(merged, path)
        log(f"[{symbol} {interval}] 캐시 저장 {len(merged):,}봉 → {path.name}")

    return merged.loc[(merged.index >= start_dt) & (merged.index < end_dt)]


def cache_summary(klines_dir: str | Path, market: str) -> pd.DataFrame:
    """캐시에 무엇이 얼마나 들어있는지 표로 보여준다."""
    root = Path(klines_dir) / market
    rows = []
    for p in sorted(root.glob("*.parquet")):
        df = load_cached(p)
        if df.empty:
            continue
        symbol, interval = p.stem.rsplit("_", 1)
        rows.append({
            "symbol": symbol,
            "interval": interval,
            "bars": len(df),
            "start": df.index[0],
            "end": df.index[-1],
            "MB": round(p.stat().st_size / 1e6, 1),
        })
    return pd.DataFrame(rows)
