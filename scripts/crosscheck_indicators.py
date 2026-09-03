"""
지표 구현 대조 — 우리 계산이 트레이딩뷰 기준값과 같은가.

    python scripts/crosscheck_indicators.py
    python scripts/crosscheck_indicators.py --interval 15m --symbols BTCUSDT ETHUSDT

**이것은 참고용 정합성 확인이지 구현 검증이 아니다.**
구현 검증의 권위는 `tests/test_reference_parity.py` 에 있다 — `ta` 라이브러리와
같은 데이터·같은 파라미터로 전 구간을 대조하고, 실측상 오차 0.000% 여야 통과한다.

트레이딩뷰를 기준으로 삼지 않는 이유:
  · 계산 윈도우와 파라미터가 공개돼 있지 않다 (스캐너의 Stoch.K, Mom 등이
    차트 기본값과 같은지 알 수 없다)
  · 자체 집계 피드를 쓴다 — 바이낸스 원본 캔들과 미세하게 다를 수 있다
  · 현재 시점 스냅샷만 준다. 과거 전 구간 대조가 불가능하다
실측에서 RSI·ATR·ADX·MACD·Williams%R·MFI 는 `ta` 와 0.000% 일치하는데도
트레이딩뷰와는 몇 % 씩 어긋난다. 그 차이는 우리 오류가 아니다.

그럼에도 이 스크립트를 남기는 이유: **자릿수가 통째로 다른 오류**는 여기서 먼저
드러난다. 실제로 이 대조가 CCI 버그(5분봉 18% 어긋남)를 찾아내는 계기가 됐다.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                      # noqa: E402
from src.data.cache import get_klines                    # noqa: E402
from src.indicators import compute_all                   # noqa: E402
from src.tradingview import scanner                      # noqa: E402

pd.set_option("display.width", 240)

# 트레이딩뷰 이름 → 우리 특징열 이름. 기본 파라미터가 같은 것만 넣는다.
PAIRS = {
    "close": "close",
    "rsi": "rsi",                    # 둘 다 14
    "adx": "adx",                    # 둘 다 14, Wilder 평활
    "di_plus": "di_plus",
    "di_minus": "di_minus",
    "atr": "atr",                    # 둘 다 14, Wilder
    "macd": "macd",                  # 둘 다 12/26/9
    "macd_signal": "macd_signal",
    "bb_upper": "bb_upper",          # 둘 다 20, 2.0
    "bb_lower": "bb_lower",
    "stoch_k": "stoch_k",            # 둘 다 14/3/3
    "stoch_d": "stoch_d",
    "cci": "cci",                    # 둘 다 20
    "williams_r": "williams_r",      # 둘 다 14
    "mfi": "mfi",                    # 둘 다 14
    "momentum": "momentum",          # 둘 다 10
    "awesome_osc": "awesome_osc",    # 둘 다 5/34
}

# **값 자체로 나누면 안 된다.** CCI·모멘텀·AO 처럼 0을 넘나드는 지표는 분모가 0에
# 가까워져 오차가 수만 %로 튄다 — 실측에서 AO 가 83,000% 로 나왔다. 지표가 틀린 게
# 아니라 척도가 틀린 것이다. 그 지표가 실제로 얼마나 흔들리는지(최근 200봉 표준편차)로
# 나눠서 "몇 시그마 차이인가"를 본다. 지표 종류가 달라도 같은 잣대로 읽힌다.
SD_WINDOW = 200
TOL_PASS = 0.1     # 1시그마의 10% 이내면 사실상 같다
TOL_WARN = 0.5


def gap_in_sd(mine: float, theirs: float, scale: float) -> float:
    if mine is None or theirs is None or not np.isfinite(mine) or not np.isfinite(theirs):
        return float("nan")
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return abs(mine - theirs) / scale


def verdict(err: float) -> str:
    if not np.isfinite(err):
        return "비교불가"
    if err <= TOL_PASS:
        return "일치"
    if err <= TOL_WARN:
        return "근사"
    return "불일치"


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="지표 구현 대조")
    ap.add_argument("--symbols", nargs="*", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--closed-only", action="store_true",
                    help="마감된 봉만으로 계산 (기본은 진행 중인 봉 포함 — "
                         "트레이딩뷰가 실시간 봉으로 계산하기 때문)")
    args = ap.parse_args()

    print(f"지표 구현 대조 · {args.interval} · {', '.join(args.symbols)}")
    print(f"기준: 트레이딩뷰 스캐너 (로그인 불필요) · "
          f"{'마감봉' if args.closed_only else '진행봉 포함'}\n")

    # ── 트레이딩뷰 기준값
    try:
        quotes = {q.symbol.split(":")[-1].replace(".P", ""): q
                  for q in scanner.fetch(args.symbols, list(PAIRS), interval=args.interval)}
    except scanner.TradingViewError as exc:
        print(f"트레이딩뷰 조회 실패: {exc}")
        return 1

    # ── 우리 계산. 지금 시점까지 받아야 대조가 성립한다.
    kd = cfg.path("paths.klines_dir")
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=120)      # 지표 워밍업에 넉넉히

    rows = []
    for sym in args.symbols:
        # refresh 를 쓰지 않는다. 증분 로직이 꼬리만 이어 받는다.
        # (좁은 구간에 refresh 를 걸어 5.9년 캐시를 날린 적이 있다.)
        df = get_klines(sym, args.interval, start, now,
                        market=cfg.require("market"), klines_dir=kd)
        if args.closed_only:
            pass          # get_klines 는 이미 마감봉만 준다
        else:
            # 트레이딩뷰는 진행 중인 봉으로 계산하므로 그 봉을 붙인다
            from src.data.binance import fetch_klines
            live = fetch_klines(sym, args.interval, df.index[-1], now,
                                market=cfg.require("market"), closed_only=False)
            if not live.empty:
                df = pd.concat([df, live]).sort_index()
                df = df[~df.index.duplicated(keep="last")]

        feats = compute_all(df)
        last = feats.iloc[-1]
        q = quotes.get(sym)
        if q is None:
            print(f"  {sym}: 트레이딩뷰 응답 없음 — 건너뜀")
            continue

        for tv_name, our_name in PAIRS.items():
            theirs = q.get(tv_name)
            mine = float(last[our_name]) if our_name in feats.columns else None
            if our_name == "close":
                mine = float(df["close"].iloc[-1])
                scale = float(df["close"].tail(SD_WINDOW).std())
            else:
                scale = (float(feats[our_name].tail(SD_WINDOW).std())
                         if our_name in feats.columns else float("nan"))
            err = gap_in_sd(mine, theirs, scale)
            rows.append({
                "종목": sym, "지표": our_name,
                "우리값": None if mine is None else round(mine, 4),
                "트레이딩뷰": None if theirs is None else round(theirs, 4),
                "차이": None if mine is None or theirs is None else round(abs(mine - theirs), 4),
                "시그마차": None if not np.isfinite(err) else round(err, 3),
                "판정": verdict(err),
            })
        print(f"  {sym} 대조 완료 ({len(df):,}봉, 마지막 {df.index[-1]:%m-%d %H:%M})",
              flush=True)

    tbl = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("대조 결과")
    print("=" * 78)
    print(tbl.to_string(index=False))

    print("\n" + "=" * 78)
    print("지표별 요약 (전 종목)")
    print("=" * 78)
    summary = (tbl[tbl["지표"] != "close"]
               .groupby("지표")
               .agg(평균_시그마차=("시그마차", "mean"),
                    최대_시그마차=("시그마차", "max"),
                    일치=("판정", lambda s: int((s == "일치").sum())),
                    근사=("판정", lambda s: int((s == "근사").sum())),
                    불일치=("판정", lambda s: int((s == "불일치").sum())))
               .round(3)
               .sort_values("최대_시그마차", ascending=False))
    print(summary.to_string())

    print()
    print("=" * 78)
    print("읽는 법")
    print("=" * 78)
    print("  이 표는 **참고용**입니다. 트레이딩뷰는 계산 윈도우와 파라미터가 공개돼")
    print("  있지 않고 자체 집계 피드를 쓰므로 구현 검증의 기준이 될 수 없습니다.")
    print("  구현 검증은 `pytest tests/test_reference_parity.py` 가 담당합니다")
    print("  (ta 라이브러리와 같은 데이터·파라미터로 전 구간 대조, 오차 0.000% 요구).")
    print("  '시그마차' 는 그 지표의 최근 200봉 표준편차 대비 차이입니다.")
    print("  값 자체로 나누면 0을 넘나드는 지표(CCI·모멘텀·AO)에서 오차가 수만 %로")
    print("  튀어 의미가 없습니다. 시그마 기준이라 지표 종류가 달라도 같이 읽힙니다.")
    print()
    huge = summary[summary["최대_시그마차"] > 3]
    if huge.empty:
        print("  3시그마를 넘는 지표는 없습니다 — 큰 구현 오류의 징후는 없습니다.")
    else:
        print(f"  ⚠ 3시그마 초과 {len(huge)}개 — 정의나 파라미터가 다를 수 있습니다:")
        print(huge.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
