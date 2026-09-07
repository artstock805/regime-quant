"""
과거 풋프린트 복원 — 바이낸스 aggTrades 아카이브에서.

왜 필요한가
    `src/live/orderbook.py` 는 실시간으로 가격대별 델타를 쌓지만, 그것만으로는
    **검증을 할 수 없다.** 우위가 있는지 재려면 과거가 필요하다. 다행히
    바이낸스는 체결 하나하나를 공격자 방향까지 담아 무료로 공개한다.

    앞서 봉 단위로 합산한 테이커 불균형은 우위가 없었다 (7개 코인 IC -0.021,
    taker_flow_momentum 5.9년 검증 1/4 관문 "우연과 구분되지 않는다").
    **봉으로 합산하는 순간 실무자들이 보는 구조가 사라진다** — 어느 가격
    레벨에서 물량이 흡수됐는지, 봉 끝이 한쪽으로만 밀렸는지가 전부 뭉개진다.
    여기서 만드는 특징들은 그 구조를 봉 하나당 숫자로 남기려는 시도다.

    값을 낼지는 모른다. 이 파일의 목적은 **재는 것**이다.

아카이브
    https://data.binance.vision/data/futures/um/daily/aggTrades/<SYM>/...zip
    열: agg_trade_id, price, quantity, first_trade_id, last_trade_id,
        transact_time, is_buyer_maker

    **is_buyer_maker=true 는 "매수자가 메이커"** — 즉 공격자는 매도자다.
    실시간 스트림의 m 필드와 같은 의미다. 이 부호를 뒤집으면 델타가 정반대로
    쌓이고, 실시간과 과거가 서로 다른 것을 재게 된다.

용량
    BTCUSDT 하루 zip 이 3~24MB, 20만 행쯤 된다. 180일이면 2~4GB 다.
    C 는 만성 부족이라 캐시는 데이터 드라이브로 보낸다.

    **하루씩 처리해서 봉 특징만 남긴다.** 180일 원본을 한 번에 메모리에
    올리면 3천만 행이 되어 터진다. 원본은 파일로 두고, 계산 결과(봉당 한 줄)만
    모은다.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..config import load_config

VISION = "https://data.binance.vision/data/futures/um/daily/aggTrades"

# 특징 정의가 바뀌면 캐시를 재사용하면 안 된다. 파일명에 이 번호가 들어가므로
# **특징을 고칠 때 반드시 올려야 한다** — 안 올리면 옛 값이 조용히 재사용된다.
FEATURE_VERSION = 3

COLUMNS = ("agg_trade_id", "price", "quantity", "first_trade_id",
           "last_trade_id", "transact_time", "is_buyer_maker")

RULES = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
         "1h": "1h", "4h": "4h", "1d": "1D"}

# 레벨이 "강하게 기울었다"고 볼 기준. 0.6 이면 대략 4:1 이다.
# 실무에서 흔히 쓰는 3:1(0.5)~4:1 범위 안이고, 이 값을 성적에 맞춰 고르면
# 그 순간 과적합이므로 관례값을 그대로 쓴다.
IMBALANCE_RATIO = 0.6


class ArchiveError(RuntimeError):
    """아카이브를 못 받았거나 형식이 다르다."""


def cache_root() -> Path:
    cfg = load_config()
    return Path(str(cfg.get("paths.data_dir", "data"))) / "aggtrades"


# ─── 원본 받기 ───────────────────────────────────────────────────

def _read_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ArchiveError("ZIP 안에 CSV 가 없습니다")
        with zf.open(names[0]) as fh:
            df = pd.read_csv(fh, header=None, dtype=str,
                             names=list(COLUMNS))
    if df.empty:
        return df
    # 아카이브에 헤더 줄이 있는 날과 없는 날이 섞여 있다. 첫 칸이 숫자로
    # 안 읽히면 헤더다.
    first = pd.to_numeric(pd.Series([df.iat[0, 0]]), errors="coerce").iloc[0]
    if pd.isna(first):
        df = df.iloc[1:].reset_index(drop=True)
    return df


def raw_day(symbol: str, day: date, *, session: requests.Session | None = None,
            keep: bool = True) -> pd.DataFrame:
    """
    하루치 원본 체결. 없으면 빈 프레임 (아직 안 올라온 날은 오류가 아니다).

    받은 zip 을 그대로 캐시한다 — 특징 정의를 바꿔 다시 계산할 때 재다운로드가
    가장 비싼 단계다.
    """
    sym = symbol.upper()
    root = cache_root() / sym
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{sym}-aggTrades-{day:%Y-%m-%d}.zip"

    if path.exists():
        content = path.read_bytes()
    else:
        url = f"{VISION}/{sym}/{sym}-aggTrades-{day:%Y-%m-%d}.zip"
        sess = session or requests.Session()
        resp = sess.get(url, timeout=180)
        if resp.status_code == 404:
            return pd.DataFrame(columns=list(COLUMNS))
        resp.raise_for_status()
        content = resp.content
        if keep:
            path.write_bytes(content)

    df = _read_zip(content)
    if df.empty:
        return df
    # 숫자로 바꾸기 **전에** 문자열에서 눈금을 구한다. 바꾼 뒤에는 이미 늦다.
    tick = infer_tick(df["price"])
    out = pd.DataFrame({
        "px": pd.to_numeric(df["price"], errors="coerce"),
        "qty": pd.to_numeric(df["quantity"], errors="coerce"),
        "ms": pd.to_numeric(df["transact_time"], errors="coerce"),
        # 문자열 "true"/"false" 로 온다. bool() 로 감싸면 "false" 가 True 다.
        "buyer_maker": df["is_buyer_maker"].astype(str).str.strip().str.lower()
                        .isin(("true", "1")),
    }).dropna(subset=["px", "qty", "ms"])
    # **공격자는 매수자인가.** is_buyer_maker=true 면 매수자가 메이커이므로
    # 공격자는 매도자다. 뒤집으면 델타가 정반대가 된다.
    out["buy"] = ~out["buyer_maker"]
    out = out.drop(columns=["buyer_maker"])
    # pandas 의 attrs 로 실어 보낸다. 별도 반환값으로 만들면 호출부가 전부
    # 튜플을 풀어야 하고, 눈금을 안 쓰는 곳에서도 신경 써야 한다.
    out.attrs["tick"] = tick
    return out


def infer_tick(px: pd.Series) -> float:
    """
    호가 눈금. 실제로 체결된 가격들의 **최소 간격**으로 잡는다.
    하루 20만 체결이면 인접 눈금이 반드시 나타나므로 참값에 수렴한다.

    **문자열로 계산한다.** float 로 빼면 0.1 이 0.09999999999854481 로 나오고,
    그 값으로 가격을 눈금에 맞추면 같은 가격이 여러 칸으로 갈라진다.
    아카이브는 "63014.9" 처럼 정확한 십진수를 주므로 그걸 그대로 쓴다.
    (숫자로 들어오면 어쩔 수 없이 float 로 계산하고 그 사실을 감춘다.)
    """
    vals = px.dropna().unique()
    if len(vals) < 2:
        return 0.0
    if isinstance(vals[0], str):
        try:
            dec = sorted({Decimal(v.strip()) for v in vals})
        except InvalidOperation:
            dec = []
        if len(dec) >= 2:
            gaps = [b - a for a, b in zip(dec, dec[1:]) if b > a]
            if gaps:
                return float(min(gaps))
    u = np.unique(np.asarray(vals, dtype="float64"))
    gaps = np.diff(u)
    gaps = gaps[gaps > 0]
    return float(gaps.min()) if gaps.size else 0.0


# ─── 봉당 풋프린트 특징 ──────────────────────────────────────────

def _bar_features(g: pd.DataFrame, tick: float, digits: int) -> dict:
    """
    한 봉의 풋프린트에서 **봉으로 뭉개면 사라지는 것들**을 뽑는다.

    delta 만 남기면 기존 taker_imbalance 와 같아진다. 여기서 재려는 것은
    "델타가 봉 안 어디에 있었나"다.
    """
    lvl = g.groupby("bucket", sort=True).agg(
        buy=("buy_qty", "sum"), sell=("sell_qty", "sum"))
    if lvl.empty:
        return {}
    lvl["vol"] = lvl["buy"] + lvl["sell"]
    lvl["delta"] = lvl["buy"] - lvl["sell"]

    lo, hi = float(lvl.index.min()), float(lvl.index.max())
    span = hi - lo
    total_vol = float(lvl["vol"].sum())
    total_delta = float(lvl["delta"].sum())

    def pos(price: float) -> float:
        """봉 범위 안 위치. 0=저가, 1=고가. 범위가 0이면 가운데."""
        return 0.5 if span <= 0 else (price - lo) / span

    poc = float(lvl["vol"].idxmax())                    # 최다 거래 레벨
    dpoc = float(lvl["delta"].abs().idxmax())           # 최대 |델타| 레벨

    # 상·하 절반의 **불균형 차이.** "위에서는 사고 아래에서는 팔았나"를 본다.
    #
    # 처음엔 (상-하)/|총델타| 로 썼더니 ±7 까지 튀었고, |상|+|하| 로 묶으니
    # 부호가 다를 때마다 ±1 에 붙어 포화됐다(실측 std 0.68, 값이 대부분 ±1).
    # 각 절반을 **그 절반의 거래량**으로 나누면 두 구간의 쏠림 세기를 비교하는
    # 뜻이 되고 -2~+2 안에서 연속적으로 변한다.
    mid = lo + span / 2.0
    up_mask = lvl.index >= mid
    up_vol = float(lvl.loc[up_mask, "vol"].sum())
    dn_vol = float(lvl.loc[~up_mask, "vol"].sum())
    upper = (float(lvl.loc[up_mask, "delta"].sum()) / up_vol) if up_vol else 0.0
    lower = (float(lvl.loc[~up_mask, "delta"].sum()) / dn_vol) if dn_vol else 0.0

    # **미완성 경매**: 봉의 극단 레벨이 한쪽 방향으로만 체결됐나.
    # 고가에서 매도가 전혀 없었으면 위로 밀던 힘이 멈추지 않은 것이다.
    #
    # 처음엔 ±1 을 더했더니 **거의 항상 0 이 나왔다** — 고가와 저가가 둘 다
    # 한쪽인 경우가 흔해서 서로 상쇄됐다. 그런데 실측한 봉에서 고가 물량은
    # 1.856, 저가는 0.006 이었다. 같은 "한쪽"이 아니다.
    # 그래서 **그 레벨이 봉 거래량에서 차지하는 비중으로 가중한다.**
    top, bot = lvl.loc[hi], lvl.loc[lo]
    unfinished = 0.0
    if total_vol > 0:
        if float(top["sell"]) == 0.0 and float(top["buy"]) > 0.0:
            unfinished += float(top["vol"]) / total_vol
        if float(bot["buy"]) == 0.0 and float(bot["sell"]) > 0.0:
            unfinished -= float(bot["vol"]) / total_vol

    # **흡수**: 극단 레벨에 물량이 몰렸나 (평균 레벨 대비).
    avg_lvl_vol = total_vol / len(lvl)
    absorb_hi = float(top["vol"]) / avg_lvl_vol if avg_lvl_vol else 0.0
    absorb_lo = float(bot["vol"]) / avg_lvl_vol if avg_lvl_vol else 0.0

    # **한쪽으로 몰려 체결된 거래량의 비중.** 부호 있음, -1~+1.
    #
    # "가장 강한 레벨의 불균형"을 쓰려 두 번 시도해 두 번 실패했다.
    # 그냥 최대 |비율| 은 항상 ±1 이었고(레벨이 700~1500개고 21% 가 한쪽만
    # 체결되는데 그 레벨들 거래량 중위가 전체의 1/40), 봉 거래량 1% 이상으로
    # 걸러도 여전히 ±1 에 포화됐다(std 0.989).
    # **최댓값은 극단만 보므로 포화된다.** 대신 강하게 기운 레벨들의 거래량이
    # 봉에서 얼마를 차지하는지 세면 포화되지 않고 세기가 남는다.
    ratio = (lvl["delta"] / lvl["vol"].replace(0.0, np.nan)).fillna(0.0)
    strong = ratio.abs() >= IMBALANCE_RATIO
    imb_share = 0.0
    if total_vol > 0 and strong.any():
        side = np.sign(ratio.where(strong, 0.0))
        imb_share = float((lvl["vol"] * side).sum() / total_vol)

    # 그 쏠림이 봉 안 어디에 있었나 (거래량 가중 평균 위치).
    imb_pos = 0.5
    if strong.any():
        w = lvl.loc[strong, "vol"]
        if float(w.sum()) > 0:
            imb_pos = pos(float((lvl.index[strong] * w).sum() / w.sum()))

    return {
        "foot_levels": int(len(lvl)),
        "foot_vol": total_vol,
        "foot_delta": total_delta,
        # 기존 지표와 같은 것 — 비교 기준으로 둔다
        "foot_delta_ratio": total_delta / total_vol if total_vol else 0.0,
        # 봉으로 뭉개면 사라지는 것들
        "foot_poc_pos": pos(poc),
        "foot_dpoc_pos": pos(dpoc),
        "foot_delta_skew": upper - lower,
        "foot_unfinished": unfinished,
        "foot_absorb_hi": absorb_hi,
        "foot_absorb_lo": absorb_lo,
        "foot_imb_share": imb_share,
        # 같은 쏠림이라도 봉 위쪽에 있었는지 아래쪽인지는 뜻이 다르다.
        "foot_imb_pos": imb_pos,
        "foot_tick_span": (span / tick) if tick > 0 else np.nan,
    }


def day_features(symbol: str, day: date, interval: str = "1h", *,
                 session: requests.Session | None = None,
                 refresh: bool = False) -> pd.DataFrame:
    """
    하루치 봉 특징. 파케이로 캐시한다.

    캐시 파일명에 FEATURE_VERSION 이 들어간다 — 특징 정의를 바꾸고 번호를
    안 올리면 옛 값이 조용히 재사용되어, 고친 것이 반영 안 된 채 결과를
    해석하게 된다.
    """
    rule = RULES.get(interval)
    if rule is None:
        raise ArchiveError(f"지원하지 않는 봉: {interval}")

    sym = symbol.upper()
    root = cache_root() / sym
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"feat-v{FEATURE_VERSION}-{interval}-{day:%Y-%m-%d}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    raw = raw_day(sym, day, session=session)
    if raw.empty:
        return pd.DataFrame()

    tick = float(raw.attrs.get("tick") or 0.0) or infer_tick(raw["px"])
    digits = 0
    if tick > 0:
        digits = max(0, min(12, int(round(-np.log10(tick))) + 2))
        raw["bucket"] = (raw["px"] / tick).round().mul(tick).round(digits)
    else:
        raw["bucket"] = raw["px"]

    raw["at"] = pd.to_datetime(raw["ms"], unit="ms", utc=True)
    # **봉 라벨은 봉의 시작 시각이다** — 클라인의 open_time 과 맞춰야 특징이
    # 엉뚱한 봉에 붙지 않는다. 한 칸 밀리면 미래 참조가 된다.
    raw["bar"] = raw["at"].dt.floor(rule)
    raw["buy_qty"] = np.where(raw["buy"], raw["qty"], 0.0)
    raw["sell_qty"] = np.where(raw["buy"], 0.0, raw["qty"])

    rows = {bar: _bar_features(g, tick, digits)
            for bar, g in raw.groupby("bar", sort=True)}
    out = pd.DataFrame.from_dict(
        {k: v for k, v in rows.items() if v}, orient="index")
    if out.empty:
        return out
    out.index.name = "open_time"
    out["foot_tick"] = tick
    out.to_parquet(path)
    return out


def features(symbol: str, interval: str = "1h", days: int = 60, *,
             end: date | None = None, refresh: bool = False,
             progress=None) -> pd.DataFrame:
    """
    최근 days 일의 봉 특징. **하루씩 처리해 봉당 한 줄만 모은다** — 원본을
    한 번에 올리면 수천만 행이 되어 터진다.
    """
    last = end or (datetime.now(timezone.utc).date() - timedelta(days=1))
    sess = requests.Session()
    frames = []
    for i in range(days):
        d = last - timedelta(days=days - 1 - i)
        try:
            part = day_features(symbol, d, interval, session=sess,
                                refresh=refresh)
        except Exception as exc:                     # pragma: no cover
            if progress:
                progress(f"  {d} 실패: {type(exc).__name__}: {exc}")
            continue
        if not part.empty:
            frames.append(part)
        if progress:
            progress(f"  {d} {len(part)}봉")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]
