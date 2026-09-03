"""
데이터 품질 감사.

파이프라인에 단위 테스트는 있지만, **실제로 캐시에 들어있는 2,500만 봉**이
멀쩡한지는 별개 문제다. 봉 하나가 누락되거나 중복되면 지표가 조용히 어긋나고,
그 위에 쌓은 모든 검정이 무의미해진다.

여기서 보는 것
  · 봉 누락      — 인터벌 간격이 어긋난 지점
  · 중복         — 같은 시각이 두 번
  · OHLC 정합성  — high >= max(open,close), low <= min(open,close), high >= low
  · 0 / 음수     — 가격이나 거래량이 0 이하
  · 이상치       — 로버스트 z(MAD 기준)와 절대 수익률을 **둘 다** 넘는 봉.
                   분할·잘못된 덤프 탐지용이다. 어느 하나만 쓰면 안 되는 이유는
                   audit_bars 의 주석에 있다.
  · 정렬         — 인덱스 단조 증가, UTC 시간대

봉 누락은 `ok` 판정에 넣지 않는다. 거래소 점검이나 덤프 결손으로도 생기고,
실제로 SOL·XRP 가 2022-02·04 에 동일한 5일을 빠뜨렸다(바이낸스 측 결손).
구조적 결함(중복·OHLC 위반·0 이하·정렬·시간대)만 `ok=False` 로 본다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.binance import interval_timedelta


@dataclass
class AuditResult:
    symbol: str
    interval: str
    bars: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    missing: int = 0
    missing_spans: list[tuple[str, str, int]] = field(default_factory=list)
    duplicates: int = 0
    ohlc_violations: int = 0
    nonpositive: int = 0
    outliers: int = 0
    outlier_times: list[str] = field(default_factory=list)
    max_outlier_return: float | None = None
    unsorted: bool = False
    naive_tz: bool = False
    coverage: float = 1.0

    @property
    def ok(self) -> bool:
        """구조적 결함이 없는가. 봉 누락은 거래소 점검으로도 생기므로 별도로 본다."""
        return not (self.duplicates or self.ohlc_violations or self.nonpositive
                    or self.unsorted or self.naive_tz)

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol, "interval": self.interval, "bars": self.bars,
            "start": self.start.isoformat() if self.start is not None else None,
            "end": self.end.isoformat() if self.end is not None else None,
            "missing": self.missing,
            # 시각 목록은 개수 열(missing/outliers)과 나란히 표시되므로, 잘리면
            # "31개인데 20건"처럼 어긋나 보인다. 상한을 넉넉히 둔다.
            "missing_spans": self.missing_spans[:100],
            "duplicates": self.duplicates, "ohlc_violations": self.ohlc_violations,
            "nonpositive": self.nonpositive, "outliers": self.outliers,
            "outlier_times": self.outlier_times[:100],
            "max_outlier_return": self.max_outlier_return,
            "unsorted": self.unsorted, "naive_tz": self.naive_tz,
            "coverage": round(self.coverage, 6), "ok": self.ok,
        }


def audit_bars(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    outlier_z: float = 20.0,
    min_abs_return: float = 0.10,
) -> AuditResult:
    """
    캔들 한 덩어리를 감사한다.

    이상치는 **통계적으로 극단이면서 동시에 경제적으로 큰** 봉만 센다.
    z-점수만 쓰면 타임프레임에 따라 의미가 완전히 달라진다 — 1분봉 MAD 가
    0.032% 라서 z>20 이 0.94% 움직임에서 걸리고, 그건 암호화폐에서 정상이다.
    실측에서 1분봉 이상치가 1시간봉의 100배 넘게 나온 것이 그 이유였다.
    min_abs_return 기본 10% 는 "데이터 오류를 의심할 만한 크기"의 하한이다.
    """
    if df.empty:
        return AuditResult(symbol, interval, 0, None, None, coverage=0.0)

    step = interval_timedelta(interval)
    idx = df.index
    res = AuditResult(
        symbol=symbol, interval=interval, bars=len(df),
        start=idx[0], end=idx[-1],
        unsorted=not idx.is_monotonic_increasing,
        naive_tz=idx.tz is None,
        duplicates=int(idx.duplicated().sum()),
    )

    # ─ 봉 누락: 간격이 step 보다 큰 지점
    gaps = idx.to_series().diff()
    big = gaps[gaps > step]
    for ts, gap in big.items():
        n_missing = int(gap / step) - 1
        if n_missing > 0:
            res.missing += n_missing
            res.missing_spans.append(
                ((ts - gap).isoformat(), ts.isoformat(), n_missing)
            )
    expected = int((idx[-1] - idx[0]) / step) + 1
    res.coverage = len(df) / expected if expected > 0 else 1.0

    # ─ OHLC 정합성
    hi, lo = df["high"], df["low"]
    o, c = df["open"], df["close"]
    bad = (hi < lo) | (hi < o) | (hi < c) | (lo > o) | (lo > c)
    res.ohlc_violations = int(bad.sum())

    # ─ 0 이하 값
    price_cols = [x for x in ("open", "high", "low", "close") if x in df.columns]
    res.nonpositive = int((df[price_cols] <= 0).any(axis=1).sum())
    if "volume" in df.columns:
        res.nonpositive += int((df["volume"] < 0).sum())

    # ─ 이상치: 봉 수익률의 **로버스트** z-점수. 분할이나 잘못된 덤프를 잡는다.
    #
    # 표준편차를 쓰면 안 된다. 극단값 하나가 표준편차를 부풀려서 자기 여진을
    # 스스로 가린다 — 종가를 5배로 튀긴 봉을 넣으면 급등(+400%)은 잡지만
    # 곧바로 되돌아온 봉(-80%)은 z가 4까지 떨어져 문턱을 못 넘는다.
    # 중위절대편차(MAD)는 이상치에 둔감해서 둘 다 잡는다.
    # 0.67449 는 정규분포에서 MAD 를 표준편차와 같은 척도로 맞추는 상수다.
    ret_all = c.pct_change()
    ret = ret_all.dropna()
    if len(ret) > 10:
        med = float(ret.median())
        mad = float((ret - med).abs().median())
        if mad > 0:
            z = 0.67449 * (ret_all - med) / mad
            # 통계적 극단 **그리고** 경제적으로 큰 움직임 — 둘 다여야 한다
            hits = ((z.abs() > outlier_z) & (ret_all.abs() > min_abs_return)).fillna(False)
            res.outliers = int(hits.sum())
            res.outlier_times = [t.isoformat() for t in idx[hits]]
            if res.outliers:
                res.max_outlier_return = float(ret_all[hits].abs().max())

    return res


def audit_cache(cfg, symbols: list[str], intervals: list[str]) -> pd.DataFrame:
    """캐시에 들어있는 전체를 감사해 표로 돌려준다."""
    from ..data.cache import get_klines

    kd = cfg.path("paths.klines_dir")
    start, end = cfg.require("backtest.start"), cfg.require("backtest.end")
    rows = []
    for sym in symbols:
        for iv in intervals:
            df = get_klines(sym, iv, start, end,
                            market=cfg.require("market"), klines_dir=kd)
            rows.append(audit_bars(df, sym, iv).as_dict())
    return pd.DataFrame(rows)


def audit_summary(table: pd.DataFrame) -> dict:
    """감사 표를 한 줄 판정으로 접는다."""
    if table.empty:
        return {"ok": False, "reason": "감사 대상이 없습니다"}
    problems = []
    if int(table["duplicates"].sum()):
        problems.append(f"중복 {int(table['duplicates'].sum())}봉")
    if int(table["ohlc_violations"].sum()):
        problems.append(f"OHLC 위반 {int(table['ohlc_violations'].sum())}봉")
    if int(table["nonpositive"].sum()):
        problems.append(f"0 이하 값 {int(table['nonpositive'].sum())}봉")
    if bool(table["unsorted"].any()):
        problems.append("정렬 깨짐")
    if bool(table["naive_tz"].any()):
        problems.append("시간대 없음")
    return {
        "ok": not problems,
        "problems": problems,
        "total_bars": int(table["bars"].sum()),
        "total_missing": int(table["missing"].sum()),
        "min_coverage": float(table["coverage"].min()),
        "total_outliers": int(table["outliers"].sum()),
        "max_outlier_return": (
            float(table["max_outlier_return"].max())
            if "max_outlier_return" in table and table["max_outlier_return"].notna().any()
            else None
        ),
    }
