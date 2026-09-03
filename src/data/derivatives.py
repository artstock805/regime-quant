"""
파생 시장 데이터 — 캔들 밖의 정보원.

검증에서 막힌 지점이 여기를 만들게 했다. 총우위 +0.0165R 을 비용 0.0252R 이
152% 먹는다. 종목을 바꾸거나 파라미터를 돌려서는 넘을 수 없는 벽이다.
특징 스크리닝에서 캔들 기반 67개가 전부 |IC|<0.05 였으므로 **우위를 캔들 밖에서
가져오지 못하면 이 시스템은 끝난다.** 그 시도가 이 모듈이다.

받는 것
    미결제약정(OI)        포지션이 쌓이는가 풀리는가 — 가격만 봐서는 알 수 없다
    상위 트레이더 롱숏     계정 기준과 포지션 기준이 갈리면 큰손이 반대로 서 있다는 뜻
    전체 계정 롱숏         군중의 방향
    테이커 매수매도 비율    누가 시장가로 밀어붙이는가
    펀딩비율              롱숏 중 어느 쪽이 비용을 내는가 = 포지션 쏠림의 가격

**REST 로는 못 받는다.** `/futures/data/*` 는 30일치만 준다(실측 확인).
5.9년 백테스트에는 쓸 수 없다. 벌크 아카이브
`data.binance.vision/data/futures/um/daily/metrics/` 에 **2021-01 부터 5분 간격**으로
전 종목이 올라와 있고, 이쪽은 제한이 없다. 펀딩비율은 2019-09 부터 있다.

주의 — 미래 참조
    metrics 의 create_time 은 '그 시점의 상태'다. 1시간봉에 붙일 때는 **봉이 닫히는
    시각까지의 마지막 값**만 써야 한다. 봉 안의 미래 값이 섞이면 백테스트가 거짓말이
    된다. `align_to_bars()` 가 그 경계를 강제한다.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import requests

from .binance import (
    BinanceDataError, _as_utc, _epoch_to_utc, _session, interval_timedelta,
)

VISION_BASE = "https://data.binance.vision/data"

# metrics 아카이브가 시작되는 날. 이보다 앞은 아예 없다 — 요청해도 404 다.
METRICS_START = date(2021, 1, 1)

# 아카이브 원본 열 → 우리 이름. 원본 이름을 그대로 쓰지 않는 이유는
# `sum_toptrader_long_short_ratio` 같은 이름이 지표 코드에서 읽히지 않기 때문이다.
METRICS_COLUMNS = {
    "sum_open_interest": "open_interest",
    "sum_open_interest_value": "open_interest_value",
    "count_toptrader_long_short_ratio": "top_trader_account_ls",
    "sum_toptrader_long_short_ratio": "top_trader_position_ls",
    "count_long_short_ratio": "global_account_ls",
    "sum_taker_long_short_vol_ratio": "taker_ls_ratio",
}
METRICS_FIELDS = tuple(METRICS_COLUMNS.values())
FUNDING_FIELDS = ("funding_rate",)
ALL_FIELDS = METRICS_FIELDS + FUNDING_FIELDS

# 값 하나를 얼마나 오래 끌고 써도 되는가. 소스 간격의 2~3배로 잡는다.
# 데이터가 끊긴 구간에서 옛날 값을 조용히 재사용하는 것이 가장 위험한 실패다 —
# 백테스트는 멀쩡히 돌아가고 결과만 거짓이 된다.
STALENESS = {f: pd.Timedelta("2h") for f in METRICS_FIELDS}
STALENESS["funding_rate"] = pd.Timedelta("24h")


def _noop(_: str) -> None:
    return None


Log = Callable[[str], None]


# ─────────────────────────────────────────────────────────────────
#  벌크 아카이브
# ─────────────────────────────────────────────────────────────────

def _read_zip_csv(content: bytes) -> pd.DataFrame:
    """ZIP 안 CSV 한 장. metrics 는 헤더가 있고 fundingRate 는 있을 때도 없을 때도 있다."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise BinanceDataError("ZIP 안에 CSV가 없습니다")
        with zf.open(names[0]) as fh:
            return pd.read_csv(fh, dtype=str)


# 아카이브 서버는 간헐적으로 5xx 를 낸다. 134종목 수집에서 4종목이 503 으로
# 죽었고, 재시도로 전부 복구됐다. 한 장 실패가 전체 수집을 죽이면 안 된다.
RETRY_STATUS = (500, 502, 503, 504, 429)
MAX_RETRIES = 4


def _fetch_zip(sess: requests.Session, url: str) -> pd.DataFrame | None:
    """
    없으면(404) None. 상장 이전이거나 아직 안 올라온 날이라 오류가 아니다.

    5xx·429 는 **일시적**이므로 지수 백오프로 재시도한다. 재시도가 없으면
    한 장의 503 이 129종목 수집 전체를 날린다(실제로 겪었다).
    """
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = sess.get(url, timeout=60)
            if resp.status_code == 404:
                return None
            if resp.status_code in RETRY_STATUS:
                raise requests.HTTPError(f"{resp.status_code} (일시적)", response=resp)
            resp.raise_for_status()
            df = _read_zip_csv(resp.content)
            return None if df.empty else df
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
            resp_obj = getattr(exc, "response", None)
            status = getattr(resp_obj, "status_code", None)
            # 재시도 대상이 아닌 오류는 바로 올린다 — 조용히 삼키면 원인을 못 찾는다
            if status is not None and status not in RETRY_STATUS:
                raise
            last = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.5 * (2 ** attempt))
    raise BinanceDataError(f"{MAX_RETRIES}회 재시도 실패: {url} ({last})")


def _days_between(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _months_between(start: date, end: date) -> Iterable[str]:
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m > 12:
            y, m = y + 1, 1


def _normalise_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=METRICS_COLUMNS)
    if "create_time" not in df.columns:
        raise BinanceDataError(f"metrics 에 create_time 이 없습니다: {list(raw.columns)[:5]}")
    # create_time 은 'YYYY-MM-DD HH:MM:SS' 문자열이다. 정수 epoch 가 아니다.
    t = pd.to_datetime(df["create_time"], utc=True, errors="coerce")
    df = df.assign(time=t)
    keep = [c for c in METRICS_FIELDS if c in df.columns]
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.loc[:, ["time", *keep]].dropna(subset=["time"])
    # kind="stable" — 기본 quicksort 는 불안정 정렬이라 같은 시각의 원래 순서가
    # 보존되지 않고, 그러면 keep="last" 가 임의의 행을 고른다. 캔들 쪽에서 겪은 사고다.
    df = df.set_index("time").sort_index(kind="stable")
    return df[~df.index.duplicated(keep="last")]


def _normalise_funding(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("calc_time") or cols.get("fundingtime") or cols.get("funding_time")
    rcol = cols.get("last_funding_rate") or cols.get("fundingrate") or cols.get("funding_rate")
    if tcol is None or rcol is None:
        raise BinanceDataError(f"펀딩 아카이브 열을 못 찾았습니다: {list(df.columns)}")
    t = pd.to_numeric(df[tcol], errors="coerce")
    time = _epoch_to_utc(t) if t.notna().any() else pd.to_datetime(df[tcol], utc=True,
                                                                   errors="coerce")
    out = pd.DataFrame({"time": time,
                        "funding_rate": pd.to_numeric(df[rcol], errors="coerce")})
    out = out.dropna(subset=["time"]).set_index("time").sort_index(kind="stable")
    return out[~out.index.duplicated(keep="last")]


def fetch_metrics(
    symbol: str, start: datetime | date | str, end: datetime | date | str,
    *, workers: int = 8, log: Log = _noop,
) -> pd.DataFrame:
    """미결제약정·롱숏비율·테이커비율. 5분 간격, 2021-01 부터."""
    s, e = _as_utc(start).date(), _as_utc(end).date()
    s = max(s, METRICS_START)
    if s > e:
        return pd.DataFrame(columns=list(METRICS_FIELDS))

    days = list(_days_between(s, e))
    sess = _session()

    def one(d: date) -> pd.DataFrame | None:
        url = (f"{VISION_BASE}/futures/um/daily/metrics/{symbol}/"
               f"{symbol}-metrics-{d:%Y-%m-%d}.zip")
        raw = _fetch_zip(sess, url)
        return None if raw is None else _normalise_metrics(raw)

    frames = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, part in enumerate(pool.map(one, days)):
            if part is not None and not part.empty:
                frames.append(part)
            if (i + 1) % 200 == 0:
                log(f"    metrics {i + 1}/{len(days)}일")
    if not frames:
        return pd.DataFrame(columns=list(METRICS_FIELDS))
    out = pd.concat(frames).sort_index(kind="stable")
    return out[~out.index.duplicated(keep="last")]


def fetch_funding(
    symbol: str, start: datetime | date | str, end: datetime | date | str,
    *, log: Log = _noop,
) -> pd.DataFrame:
    """펀딩비율. 8시간 간격, 2019-09 부터. 월간 아카이브를 쓴다."""
    s, e = _as_utc(start).date(), _as_utc(end).date()
    sess = _session()
    frames = []
    for period in _months_between(s, e):
        url = (f"{VISION_BASE}/futures/um/monthly/fundingRate/{symbol}/"
               f"{symbol}-fundingRate-{period}.zip")
        raw = _fetch_zip(sess, url)
        if raw is not None:
            frames.append(_normalise_funding(raw))
    if not frames:
        return pd.DataFrame(columns=list(FUNDING_FIELDS))
    out = pd.concat(frames).sort_index(kind="stable")
    out = out[~out.index.duplicated(keep="last")]
    return out.loc[str(s):str(e + timedelta(days=1))]


# ─────────────────────────────────────────────────────────────────
#  캐시
# ─────────────────────────────────────────────────────────────────

def cache_path(deriv_dir: str | Path, market: str, symbol: str, kind: str) -> Path:
    return Path(deriv_dir) / market / f"{symbol}-{kind}.parquet"


def load_cached(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()
    return df.sort_index(kind="stable")


def save_cached(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    return p


def _probe_path(cache: Path) -> Path:
    return Path(cache).with_suffix(".probe.json")


def _load_probe(cache: str | Path) -> datetime | None:
    """이 종목에 대해 '여기보다 앞은 없다'고 확인한 시각."""
    p = _probe_path(Path(cache))
    if not p.exists():
        return None
    try:
        return datetime.fromisoformat(json.loads(p.read_text())["probed_start"])
    except Exception:
        return None          # 손상된 사이드카는 없는 것으로 본다 (다시 훑으면 된다)


def _save_probe(cache: str | Path, probed_start: datetime) -> None:
    p = _probe_path(Path(cache))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"probed_start": probed_start.isoformat()}), encoding="utf-8")


def get_derivatives(
    symbol: str,
    start: datetime | date | str,
    end: datetime | date | str,
    *,
    market: str = "futures_um",
    deriv_dir: str | Path = "data/derivatives",
    refresh: bool = False,
    log: Log = _noop,
) -> pd.DataFrame:
    """
    metrics + funding 을 합쳐 하나의 시계열로 돌려준다. 캐시를 쓴다.

    캔들 캐시에서 배운 것: **refresh 는 '무엇을 다시 받을지'만 정하고 '무엇을 버릴지'는
    정하지 않는다.** 좁은 구간에 refresh 를 걸어 5.9년치를 날린 사고가 있었다.
    디스크에 있는 것은 항상 읽어서 병합한다.
    """
    s, e = _as_utc(start), _as_utc(end)
    frames = {}
    for kind, fetch in (("metrics", fetch_metrics), ("funding", fetch_funding)):
        path = cache_path(deriv_dir, market, symbol, kind)
        existing = load_cached(path)
        have_start = existing.index.min() if not existing.empty else None
        have_end = existing.index.max() if not existing.empty else None
        probed = _load_probe(path)

        need: list[tuple[datetime, datetime]] = []
        if refresh or existing.empty:
            need.append((s, e))
        else:
            # **없다고 확인한 구간을 다시 훑지 않는다.**
            # 대부분 종목의 아카이브는 2021-12 부터인데 백테스트 시작은 2020-10 이다.
            # 그대로 두면 실행마다 존재하지 않는 430일을 다시 요청한다. 404 는
            # 오류가 아니라 조용히 넘어가므로 눈에 띄지 않고, 종목당 수백 번의
            # 헛요청이 수집 시간을 늘리고 간헐적 503 을 맞을 확률만 키운다.
            head_from = s if probed is None else max(s, probed)
            if head_from < have_start:
                need.append((head_from, have_start))
            if e > have_end:
                need.append((have_end, e))

        parts = [existing] if not existing.empty else []
        for a, b in need:
            log(f"  {symbol} {kind} {a:%Y-%m-%d}~{b:%Y-%m-%d} 내려받는 중")
            got = fetch(symbol, a, b, log=log)
            if not got.empty:
                parts.append(got)
        if parts:
            merged = pd.concat(parts).sort_index(kind="stable")
            merged = merged[~merged.index.duplicated(keep="last")]
            save_cached(merged, path)
            frames[kind] = merged
            # 요청했던 가장 이른 시각을 남긴다. 다음 실행은 여기보다 앞을 묻지 않는다.
            asked = min([s] + [a for a, _ in need])
            _save_probe(path, min(asked, merged.index.min()))

    if not frames:
        return pd.DataFrame(columns=list(ALL_FIELDS))
    out = pd.concat(frames.values(), axis=1, sort=True).sort_index(kind="stable")
    return out.loc[(out.index >= s) & (out.index <= e)]


# ─────────────────────────────────────────────────────────────────
#  봉에 붙이기 — 미래 참조가 생기는 곳
# ─────────────────────────────────────────────────────────────────

def align_to_bars(bars: pd.DataFrame, deriv: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    파생 시계열을 봉 인덱스에 맞춰 붙인다.

    **경계가 전부다.** 봉의 인덱스는 여는 시각이고 지표는 닫는 값(close)으로 계산한다.
    그러니 파생값도 **그 봉이 닫히는 시각까지**의 마지막 값을 써야 종가와 같은 시점이
    된다. 봉 안쪽 미래 값이 섞이면 백테스트 전체가 거짓말이 된다.

    metrics 는 5분, 펀딩은 8시간 간격이라 봉보다 성기다. 뒤로 채우면(bfill) 미래를
    보게 되므로 앞의 값만 끌어온다(merge_asof direction="backward").
    """
    if deriv.empty or bars.empty:
        return bars

    step = interval_timedelta(interval)
    left = pd.DataFrame({"_close": bars.index + step}).sort_values("_close", kind="stable")

    # **열마다 따로 붙인다.** metrics 는 5분, 펀딩은 8시간이라 옆으로 합쳐 두면
    # 대부분의 행에서 펀딩이 비어 있다. merge_asof 는 직전 '행'을 가져오므로
    # 한 번에 붙이면 펀딩이 거의 다 NaN 이 된다(실측 커버리지 2.08%).
    out = {}
    for col in (c for c in ALL_FIELDS if c in deriv.columns):
        s = deriv[col].dropna()
        if s.empty:
            continue
        right = (s.rename(col).rename_axis("time").reset_index()
                 .sort_values("time", kind="stable"))
        joined = pd.merge_asof(left, right, left_on="_close", right_on="time",
                               direction="backward",
                               # 오래된 값을 무한정 끌고 오지 않는다. 데이터가 끊긴
                               # 구간에서 조용히 옛날 값을 쓰는 게 가장 위험하다.
                               tolerance=STALENESS[col])
        out[col] = joined[col].to_numpy()

    if not out:
        return bars
    return bars.join(pd.DataFrame(out, index=bars.index))


def coverage(bars: pd.DataFrame, interval: str = "1h") -> pd.DataFrame:
    """붙은 뒤에 실제로 몇 %가 채워졌는지. 조용한 결측을 눈에 보이게 한다."""
    rows = []
    for f in ALL_FIELDS:
        if f not in bars.columns:
            rows.append({"필드": f, "커버리지_%": 0.0, "첫값": None})
            continue
        s = bars[f]
        first = s.first_valid_index()
        rows.append({"필드": f,
                     "커버리지_%": round(float(s.notna().mean()) * 100, 2),
                     "첫값": None if first is None else first.strftime("%Y-%m-%d")})
    return pd.DataFrame(rows)
