"""
틱을 봉 단위 특징으로 집계한다 — 봉 종가 위치보다 **해상도가 높은** 오더플로 대용.

왜 필요한가
    `indicators/orderflow.py` 의 대용은 봉 하나당 숫자 하나(종가 위치)로 방향을
    추정한다. 7개 통화쌍 실측에서 그 방식은 기대R 이 25칸 중 19칸 음수였고
    횡보에서는 7/7 전부 음수였다. 재료가 너무 거칠다는 뜻이다.

    틱은 같은 1시간 안에 **2,500~5,000번의 실제 움직임**을 담고 있다. 종가가
    어디 붙었는지 하나로 뭉개는 대신 그 경로를 세면 훨씬 많은 것을 볼 수 있다.

**여기에도 체결 정보는 없다.** 실측으로 확인한 것 (XM, EURUSD, 활발한 시간대
4,933틱):

    BUY(32)·SELL(64) 비트    둘 다 100% 켜짐 → 공격자 방향 정보가 아니다
    LAST(8)·VOLUME(16) 비트  0% → 체결가·체결량이 아예 안 온다
    bid 갱신 4,346 / ask 갱신 4,313 → 거의 대칭, 한쪽 쏠림을 못 읽는다

그래서 쓸 수 있는 것은 **mid 가 몇 번 올랐고 몇 번 내렸나**(틱 규칙)와 그
경로의 모양뿐이다. 체결 방향을 모를 때 쓰는 고전적인 대용이고, 그 이상을
주장하지 않는다.

용량
    EURUSD 1시간에 약 2,500~5,000틱, 60일이면 300만 틱쯤 된다. C 드라이브는
    만성적으로 부족하므로 캐시는 **데이터 드라이브**(REGIME_QUANT_DATA, 기본
    config 의 data_dir)로 보낸다. 주 단위로 나눠 받아 조각마다 파케이로
    저장한다 — 한 번에 60일을 요청하면 메모리에 다 올라오고, 중간에 끊기면
    처음부터 다시 받아야 한다.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config
from .mt5 import MT5Error, _lock, _mt5, initialise

# MT5 틱 플래그. 우리가 쓰는 것은 없지만, 왜 안 쓰는지 남겨 두려고 적는다.
_FLAG_BID, _FLAG_ASK = 2, 4
_FLAG_LAST, _FLAG_VOLUME = 8, 16

# 파케이 한 조각의 기간. 너무 짧으면 파일이 많아지고, 너무 길면 재시도가 비싸다.
CHUNK_DAYS = 7


def cache_root() -> Path:
    """틱 캐시 위치. C 가 만성 부족이라 데이터 드라이브로 보낸다."""
    cfg = load_config()
    base = Path(str(cfg.get("paths.data_dir", "data")))
    return base / "fx-ticks"


def _fetch_range(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """구간의 원본 틱. 없으면 빈 프레임 (주말은 정상적으로 비어 있다)."""
    with _lock:
        initialise()
        mt5 = _mt5()
        if not mt5.symbol_select(symbol, True):
            raise MT5Error(f"{symbol} 을 Market Watch 에 추가하지 못했습니다")
        raw = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_ALL)
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["time_msc", "bid", "ask", "flags"])
    df = pd.DataFrame(raw)
    keep = [c for c in ("time_msc", "bid", "ask", "flags") if c in df.columns]
    return df[keep]


def load_ticks(symbol: str, days: int = 60, *,
               refresh: bool = False) -> pd.DataFrame:
    """
    최근 days 일의 틱. 주 단위 조각으로 캐시한다.

    **진행 중인 주는 캐시하지 않는다.** 아직 틱이 더 들어올 구간을 파일로
    굳혀 두면 다음에 그 조각을 그대로 읽어 최근 데이터가 영구히 빠진다.
    """
    root = cache_root() / symbol
    root.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)

    pieces: list[pd.DataFrame] = []
    cursor = start
    while cursor < now:
        stop = min(cursor + timedelta(days=CHUNK_DAYS), now)
        # 조각 경계가 안 겹치게 파일명을 시작일로 고정한다
        path = root / f"{cursor:%Y%m%d}-{CHUNK_DAYS}d.parquet"
        fresh_edge = stop >= now - timedelta(hours=1)

        if path.exists() and not refresh and not fresh_edge:
            pieces.append(pd.read_parquet(path))
        else:
            part = _fetch_range(symbol, cursor, stop)
            if not part.empty and not fresh_edge:
                part.to_parquet(path, index=False)
            if not part.empty:
                pieces.append(part)
        cursor = stop

    if not pieces:
        return pd.DataFrame(columns=["time_msc", "bid", "ask", "flags"])
    out = pd.concat(pieces, ignore_index=True)
    out = out.drop_duplicates(subset=["time_msc", "bid", "ask"])
    return out.sort_values("time_msc").reset_index(drop=True)


def bar_features(ticks: pd.DataFrame, interval: str = "1h") -> pd.DataFrame:
    """
    틱을 봉 단위 특징으로 접는다.

    **봉 라벨은 봉의 시작 시각이다** — `mt5.bars()` 의 `open_time` 인덱스와
    맞춰야 특징이 엉뚱한 봉에 붙지 않는다. 한 칸만 밀려도 미래 참조가 된다.
    """
    if ticks.empty:
        return pd.DataFrame()

    t = ticks.copy()
    t["at"] = pd.to_datetime(t["time_msc"], unit="ms", utc=True)
    t["mid"] = (t["bid"].astype("float64") + t["ask"].astype("float64")) / 2.0
    t["spread"] = t["ask"].astype("float64") - t["bid"].astype("float64")

    # 틱 규칙: mid 가 올랐나 내렸나. 보합은 어느 쪽도 아니다.
    d = t["mid"].diff()
    t["up"] = (d > 0).astype("float64")
    t["down"] = (d < 0).astype("float64")
    t["ret"] = t["mid"].pct_change()

    rule = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
            "1h": "1h", "4h": "4h", "1d": "1D"}.get(interval)
    if rule is None:
        raise MT5Error(f"지원하지 않는 봉: {interval}")

    g = t.set_index("at").resample(rule, label="left", closed="left")
    out = pd.DataFrame({
        "tick_count": g["mid"].count(),
        "tick_up": g["up"].sum(),
        "tick_down": g["down"].sum(),
        "tick_spread_mean": g["spread"].mean(),
        "tick_spread_max": g["spread"].max(),
        # 틱 단위 실현변동성. 봉 고저폭과 달리 경로의 거칠기를 잡는다.
        "tick_rv": g["ret"].std(),
    })

    moved = (out["tick_up"] + out["tick_down"]).replace(0.0, np.nan)
    # 상승틱 비중 - 하락틱 비중. -1 ~ +1.
    out["tick_imbalance"] = (out["tick_up"] - out["tick_down"]) / moved
    # 움직인 틱이 하나도 없는 봉(장 닫힘 경계)은 방향을 알 수 없다 → 0
    out["tick_imbalance"] = out["tick_imbalance"].fillna(0.0)

    out.index.name = "open_time"
    return out.dropna(subset=["tick_count"]).loc[out["tick_count"] > 0]


def attach(bars: pd.DataFrame, symbol: str, interval: str = "1h",
           days: int = 60, *, refresh: bool = False) -> pd.DataFrame:
    """
    봉 프레임에 틱 특징을 붙인다. 겹치지 않는 구간은 NaN 으로 남는다.

    NaN 을 0 으로 채우지 않는다 — 0 은 "쏠림이 없다"는 뜻이고 NaN 은 "모른다"는
    뜻이다. 섞으면 틱을 못 받은 구간이 중립 신호로 읽혀 지표가 조용히 희석된다.
    """
    feats = bar_features(load_ticks(symbol, days, refresh=refresh), interval)
    if feats.empty:
        return pd.DataFrame(index=bars.index)
    return feats.reindex(bars.index)
