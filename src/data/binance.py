"""
바이낸스 캔들(kline) 수집.

두 경로를 함께 쓴다.
  1) data.binance.vision 벌크 덤프 — 월간/일간 ZIP. API 키 불필요, 레이트리밋 없음, 압도적으로 빠름.
     지나간 달은 월간 ZIP, 이번 달 지나간 날은 일간 ZIP으로 받는다.
  2) REST /klines — 오늘 치 꼬리 구간만. 덤프는 하루 늦게 올라오기 때문.

실측으로 확인한 함정 두 개를 여기서 흡수한다.
  · CSV 덤프에 헤더 줄이 있는 파일과 없는 파일이 섞여 있다 (바이낸스가 중간에 추가했다).
  · open_time 단위가 파일에 따라 밀리초 / 마이크로초로 다르다.
둘 다 값의 크기를 보고 자동 판별한다. 판별하지 않으면 1970년이나 55000년에 찍힌
인덱스가 조용히 섞여 들어와서 백테스트 전체가 어긋난다.
"""

from __future__ import annotations

import io
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Iterable

import pandas as pd
import requests

VISION_BASE = "https://data.binance.vision/data"
REST_HOSTS = {
    "spot": ("https://api.binance.com/api/v3/klines", 1000),
    "futures_um": ("https://fapi.binance.com/fapi/v1/klines", 1500),
}
VISION_SEGMENT = {"spot": "spot", "futures_um": "futures/um"}

# 덤프 CSV의 12개 열 (바이낸스 문서 순서)
RAW_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]
# 우리가 실제로 들고 다니는 열
KEEP_COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"]

INTERVAL_MS: dict[str, int] = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000,
}

Logger = Callable[[str], None]


def _noop(_: str) -> None:
    pass


class BinanceDataError(RuntimeError):
    pass


# ─────────────────────────────────────────────────────────────────
#  공통 유틸
# ─────────────────────────────────────────────────────────────────

def interval_timedelta(interval: str) -> timedelta:
    if interval not in INTERVAL_MS:
        raise BinanceDataError(f"지원하지 않는 인터벌: {interval} (가능: {', '.join(INTERVAL_MS)})")
    return timedelta(milliseconds=INTERVAL_MS[interval])


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "regime-quant/0.1"})
    adapter = requests.adapters.HTTPAdapter(max_retries=3, pool_maxsize=8)
    s.mount("https://", adapter)
    return s


def _epoch_to_utc(values: pd.Series) -> pd.Series:
    """
    정수 타임스탬프를 UTC datetime으로. 단위(초/밀리/마이크로/나노)를 크기로 판별한다.
    2020~2030년대를 기준으로 ms는 ~1.6e12, us는 ~1.6e15 규모다.
    """
    nums = pd.to_numeric(values, errors="coerce")
    med = float(nums.dropna().abs().median()) if nums.notna().any() else 0.0
    if med > 1e17:
        unit = "ns"
    elif med > 1e14:
        unit = "us"
    elif med > 1e11:
        unit = "ms"
    else:
        unit = "s"
    return pd.to_datetime(nums, unit=unit, utc=True)


def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """12열 원본을 표준 스키마(UTC 인덱스 + 8개 숫자열)로 정리한다."""
    df = raw.copy()
    df.columns = RAW_COLUMNS[: len(df.columns)]

    df["open_time"] = _epoch_to_utc(df["open_time"])
    for col in KEEP_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.loc[:, ["open_time", *[c for c in KEEP_COLUMNS if c in df.columns]]]
    df = df.dropna(subset=["open_time", "open", "high", "low", "close"])
    # kind="stable" 이 필수다. 기본 quicksort 는 불안정 정렬이라 같은 타임스탬프의
    # 원래 순서가 보존되지 않고, 그러면 아래 keep="last" 가 임의의 행을 고른다.
    df = df.set_index("open_time").sort_index(kind="stable")
    # 같은 봉이 두 소스에서 겹쳐 들어오는 경우가 있다 — 나중 것(더 신선한 것)을 남긴다
    df = df[~df.index.duplicated(keep="last")]
    return df


def _read_dump_zip(content: bytes) -> pd.DataFrame:
    """ZIP 안의 CSV 한 장을 읽는다. 헤더 줄 유무를 자동 판별한다."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise BinanceDataError("ZIP 안에 CSV가 없습니다")
        with zf.open(names[0]) as fh:
            df = pd.read_csv(fh, header=None, dtype=str)

    if df.empty:
        return df
    # 첫 줄 첫 칸이 숫자로 안 읽히면 헤더 줄이다
    if pd.isna(pd.to_numeric(pd.Series([df.iat[0, 0]]), errors="coerce").iloc[0]):
        df = df.iloc[1:].reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────
#  벌크 덤프
# ─────────────────────────────────────────────────────────────────

def _vision_url(market: str, symbol: str, interval: str, period: str, daily: bool) -> str:
    seg = VISION_SEGMENT.get(market)
    if seg is None:
        raise BinanceDataError(f"지원하지 않는 시장: {market} (가능: {', '.join(VISION_SEGMENT)})")
    grain = "daily" if daily else "monthly"
    return f"{VISION_BASE}/{seg}/{grain}/klines/{symbol}/{interval}/{symbol}-{interval}-{period}.zip"


def _fetch_dump(
    sess: requests.Session, market: str, symbol: str, interval: str, period: str, daily: bool
) -> pd.DataFrame | None:
    """
    덤프 한 장을 받는다. 없으면(404) None — 종목 상장 이전 달이거나 아직 안 올라온 날이다.
    이건 오류가 아니라 정상적인 '없음'이므로 조용히 넘긴다.
    """
    url = _vision_url(market, symbol, interval, period, daily)
    resp = sess.get(url, timeout=60)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    raw = _read_dump_zip(resp.content)
    if raw.empty:
        return None
    # 소스마다 다른 원본 모양을 여기서 바로 표준 스키마로 접는다.
    # 정규화를 미루고 원본 12열 모양으로 되돌리려 하면 타임스탬프 해상도에서 사고가 난다.
    return _normalise(raw)


def _months_between(start: date, end: date) -> Iterable[str]:
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m > 12:
            y, m = y + 1, 1


# ─────────────────────────────────────────────────────────────────
#  REST (꼬리 구간)
# ─────────────────────────────────────────────────────────────────

def fetch_rest(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    market: str = "futures_um",
    sess: requests.Session | None = None,
    log: Logger = _noop,
) -> pd.DataFrame:
    """REST /klines 를 페이지네이션해서 [start, end) 구간을 받는다."""
    if market not in REST_HOSTS:
        raise BinanceDataError(f"지원하지 않는 시장: {market}")
    url, page_limit = REST_HOSTS[market]
    sess = sess or _session()
    step = INTERVAL_MS[interval]

    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    frames: list[pd.DataFrame] = []

    while cursor < end_ms:
        params = {
            "symbol": symbol, "interval": interval,
            "startTime": cursor, "endTime": end_ms, "limit": page_limit,
        }
        resp = sess.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            log("  레이트리밋 — 10초 대기")
            time.sleep(10)
            continue
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        last_open = int(rows[-1][0])
        cursor = last_open + step
        if len(rows) < page_limit:
            break
        time.sleep(0.12)  # 가중치 여유를 남긴다

    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS)
    return _normalise(pd.concat(frames, ignore_index=True))


# ─────────────────────────────────────────────────────────────────
#  공개 API
# ─────────────────────────────────────────────────────────────────

def fetch_klines(
    symbol: str,
    interval: str,
    start: datetime | date | str,
    end: datetime | date | str | None = None,
    market: str = "futures_um",
    closed_only: bool = True,
    log: Logger = _noop,
) -> pd.DataFrame:
    """
    [start, end) 구간의 캔들을 받아 표준 스키마 DataFrame으로 돌려준다.

    반환: UTC tz-aware DatetimeIndex(open_time) + open/high/low/close/volume/
          quote_volume/trades/taker_buy_base

    closed_only=True(기본)면 아직 닫히지 않은 마지막 봉을 버린다. 미완성 봉으로
    신호를 만들면 실전에서 재현되지 않는 백테스트가 되기 때문에 기본값을 이렇게 둔다.
    """
    start_dt = _as_utc(start)
    end_dt = _as_utc(end) if end is not None else datetime.now(timezone.utc)
    if end_dt <= start_dt:
        raise BinanceDataError(f"end({end_dt})가 start({start_dt})보다 앞입니다")

    interval_timedelta(interval)  # 유효성 검사
    sess = _session()
    frames: list[pd.DataFrame] = []

    today = datetime.now(timezone.utc).date()
    # 이번 달은 월간 덤프가 아직 없다 → 월간은 지난달까지만
    monthly_cutoff = date(today.year, today.month, 1)

    # 1) 월간 덤프
    for period in _months_between(start_dt.date(), end_dt.date()):
        first_of_month = datetime.strptime(period, "%Y-%m").date()
        if first_of_month >= monthly_cutoff:
            break
        df = _fetch_dump(sess, market, symbol, interval, period, daily=False)
        if df is None:
            log(f"  {symbol} {interval} {period} 월간 덤프 없음 — 건너뜀")
            continue
        frames.append(df)
        log(f"  {symbol} {interval} {period} 월간 {len(df):,}봉")

    # 2) 이번 달 일간 덤프 (어제까지 — 오늘 치는 아직 안 올라온다)
    day = max(monthly_cutoff, start_dt.date())
    last_dump_day = min(end_dt.date(), today) - timedelta(days=1)
    daily_count = 0
    while day <= last_dump_day:
        df = _fetch_dump(sess, market, symbol, interval, day.isoformat(), daily=True)
        if df is not None:
            frames.append(df)
            daily_count += len(df)
        day += timedelta(days=1)
    if daily_count:
        log(f"  {symbol} {interval} 일간 덤프 {daily_count:,}봉")

    # 3) 꼬리 구간은 REST
    tail_start = max(datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc), start_dt)
    if tail_start < end_dt:
        tail = fetch_rest(symbol, interval, tail_start, end_dt, market, sess, log)
        if not tail.empty:
            frames.append(tail)
            log(f"  {symbol} {interval} REST 꼬리 {len(tail):,}봉")

    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS)

    # 모든 프레임이 이미 표준 스키마(UTC 인덱스)다 — 인덱스째로 합치고 중복만 정리한다.
    # frames 순서가 월간덤프 → 일간덤프 → REST 이므로, 안정 정렬이어야
    # 겹치는 봉에서 가장 신선한 REST 값이 이긴다.
    out = pd.concat(frames).sort_index(kind="stable")
    out = out[~out.index.duplicated(keep="last")]
    out = out.loc[(out.index >= start_dt) & (out.index < end_dt)]

    if closed_only and not out.empty:
        # open_time T 봉은 T + interval 에 닫힌다. 그 시점이 지나지 않은 봉은 버린다.
        span = interval_timedelta(interval)
        cutoff = min(end_dt, datetime.now(timezone.utc)) - span
        dropped = int((out.index > cutoff).sum())
        if dropped:
            out = out.loc[out.index <= cutoff]
            log(f"  {symbol} {interval} 미완성 봉 {dropped}개 제외")
    return out


def _as_utc(value: datetime | date | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)
