"""
복원한 풋프린트에 우위가 있나 — 재는 스크립트.

**비교 기준을 같이 재는 것이 요점이다.** 풋프린트 특징의 IC 만 보면 커 보일
수 있는데, 이 프로젝트에서 이미 확인된 것이 둘 있다.

    직전수익률 z (순수 단기반전)   코인 1시간봉 IC 약 -0.034
    봉 단위 테이커 불균형          코인 1시간봉 IC 약 -0.021

풋프린트가 값이 있다면 **이 둘보다 세거나, 이 둘과 상관이 낮아야** 한다.
둘 다 아니면 봉으로 뭉갠 것을 더 비싸게 다시 만든 것일 뿐이다
(FX 대용에서 이미 겪었다 — flow_extension_fade 와 방향 100% 일치).

사용법
    python scripts/probe_footprint.py --symbols BTCUSDT --days 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import aggtrades as at                                # noqa: E402
from src.indicators import compute_all                              # noqa: E402
from src.indicators.core import zscore                              # noqa: E402

FOOT_COLS = [
    "foot_delta_ratio",     # 봉 전체 델타 비율 — 기존 taker_imbalance 와 같은 것
    "foot_poc_pos",         # 최다거래 레벨이 봉 어디에 있었나
    "foot_dpoc_pos",        # 최대 |델타| 레벨 위치
    "foot_delta_skew",      # 위쪽 쏠림 - 아래쪽 쏠림
    "foot_unfinished",      # 극단 레벨이 한쪽으로만 체결됐나 (거래량 가중)
    "foot_absorb_hi",       # 고가 레벨에 물량이 몰렸나
    "foot_absorb_lo",
    "foot_imb_share",       # 강하게 기운 레벨의 거래량 비중
    "foot_imb_pos",
    "foot_levels",          # 봉이 지난 눈금 수
]


def klines(symbol: str, interval: str) -> pd.DataFrame:
    root = Path(str(at.load_config().get("paths.data_dir", "data")))
    path = root / "klines" / "futures_um" / f"{symbol}_{interval}.parquet"
    if not path.exists():
        raise SystemExit(f"클라인이 없습니다: {path}")
    return pd.read_parquet(path)


def measure(symbol: str, days: int, interval: str,
            end: str | None = None) -> pd.DataFrame | None:
    bars = klines(symbol, interval)
    # 클라인이 끝나는 날까지만 받는다. 그 뒤 날짜는 겹치지 않아 버려진다.
    last = (pd.Timestamp(end, tz="UTC") if end else bars.index[-1]).date()
    print(f"[{symbol}] 아카이브 {days}일 처리 (~{last})", flush=True)
    seen = {"n": 0}

    def tick(msg: str) -> None:
        seen["n"] += 1
        if seen["n"] % 10 == 0:
            print(f"  … {seen['n']}/{days}일", flush=True)

    foot = at.features(symbol, interval, days, end=last, progress=tick)
    if foot.empty:
        print(f"[{symbol}] 특징이 비었습니다")
        return None

    j = bars.join(foot, how="inner")
    if j.empty:
        print(f"[{symbol}] 클라인과 겹치는 구간이 없습니다")
        return None

    base = compute_all(j, ["taker_imbalance", "ofi_z"])
    j = j.join(base)
    j["rev_z"] = zscore(j["close"].pct_change(), 100)
    fwd = j["close"].pct_change().shift(-1)

    rows = {}
    cols = FOOT_COLS + ["taker_imbalance", "ofi_z", "rev_z"]
    for c in cols:
        if c not in j.columns:
            continue
        rows[c] = {
            "IC": j[c].corr(fwd, method="spearman"),
            # 기존 두 기준과의 상관. 높으면 새 정보가 아니다.
            "vs_반전": j[c].corr(j["rev_z"], method="spearman"),
            "vs_테이커": j[c].corr(j["taker_imbalance"], method="spearman"),
        }
    out = pd.DataFrame(rows).T
    out["봉"] = len(j)
    out["종목"] = symbol
    print(f"[{symbol}] 겹치는 봉 {len(j)}개 "
          f"({j.index[0]:%Y-%m-%d} ~ {j.index[-1]:%Y-%m-%d})", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--interval", default="1h")
    # 아카이브는 어제까지 있지만 로컬 클라인은 그보다 앞에서 끝난다.
    # 겹치지 않는 날을 받으면 다운로드만 하고 버리게 되므로 끝날을 지정한다.
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (기본: 클라인 마지막 날)")
    args = ap.parse_args()

    frames = []
    for sym in args.symbols.split(","):
        r = measure(sym.strip().upper(), args.days, args.interval, args.end)
        if r is not None:
            frames.append(r)
    if not frames:
        return 1

    all_ = pd.concat(frames)
    print()
    print("=" * 78)
    print(f"풋프린트 IC · {args.interval} · 비교 기준을 같이 본다")
    print("=" * 78)
    for sym, g in all_.groupby("종목"):
        print(f"\n── {sym} ({int(g['봉'].iloc[0])}봉) "
              "──────────────────────────────")
        show = g[["IC", "vs_반전", "vs_테이커"]].copy()
        show["|IC|"] = show["IC"].abs()
        print(show.sort_values("|IC|", ascending=False)
              .drop(columns=["|IC|"]).round(4).to_string())

    if len(frames) > 1:
        print("\n── 종목 평균 ──────────────────────────────")
        m = all_.groupby(level=0)[["IC", "vs_반전", "vs_테이커"]].mean()
        m["부호일치"] = all_.groupby(level=0)["IC"].apply(
            lambda s: f"{int((s < 0).sum())}-/{int((s > 0).sum())}+")
        print(m.reindex(m["IC"].abs().sort_values(ascending=False).index)
              .round(4).to_string())

    print()
    print("판단 기준: 풋프린트가 값이 있으려면 |IC| 가 반전(-0.034)·테이커"
          "(-0.021)보다 크거나,")
    print("           그 둘과의 상관이 낮아야 한다. 둘 다 아니면 기존 것을"
          " 비싸게 다시 만든 것이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
