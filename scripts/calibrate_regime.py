"""
레짐 분류기 임계값 캘리브레이션.

    python scripts/calibrate_regime.py                # 스윕 결과 표
    python scripts/calibrate_regime.py --quantiles    # 정규화 앵커용 분위수만

임계값을 추측으로 정하지 않기 위한 도구다. 목표는 두 가지를 동시에 만족하는 지점.
  1) 분포가 쓸모 있어야 한다 — 한 국면이 80%를 먹으면 분류기가 일을 안 하는 것이다.
  2) 추세 구간이 충분히 길어야 한다 — 중위 5봉(25분)짜리 추세로는 추세 전략을 못 태운다.

타임프레임이나 시장을 바꾸면 다시 돌려서 RegimeParams 기본값을 갱신해야 한다.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                                    # noqa: E402
from src.data.cache import get_klines                                 # noqa: E402
from src.indicators import compute_all                                # noqa: E402
from src.regime import ALL_REGIMES, KOREAN, RegimeParams, classify    # noqa: E402

ANCHOR_COLS = ["adx", "efficiency_ratio", "choppiness", "bbw_percentile", "atr_ratio"]
NEEDED = ["adx", "efficiency_ratio", "choppiness", "bbw_percentile", "atr_ratio", "ema_spread"]


def load_features(symbols: list[str], interval: str, cfg) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        df = get_klines(
            sym, interval,
            cfg.require("backtest.start"), cfg.require("backtest.end"),
            market=cfg.require("market"), klines_dir=cfg.path("paths.klines_dir"),
        )
        f = compute_all(df, NEEDED)
        f["symbol"] = sym
        frames.append(f)
    return pd.concat(frames)


def describe(reg: pd.DataFrame) -> dict:
    """국면별 비중과 지속 길이. 종목별로 따로 런을 세야 경계에서 섞이지 않는다."""
    r = reg["regime"].dropna()
    if r.empty:
        return {}
    runs = r.ne(r.shift(1)).cumsum()
    run_len = r.groupby(runs).size()
    run_lab = r.groupby(runs).first()

    out = {}
    for g in ALL_REGIMES:
        out[f"{KOREAN[g]}_%"] = round((r == g).mean() * 100, 1)
    trend_runs = run_len[run_lab.isin(["uptrend", "downtrend"])]
    out["추세중위봉"] = int(trend_runs.median()) if len(trend_runs) else 0
    out["추세런수"] = int(len(trend_runs))
    out["전환수"] = int(r.ne(r.shift(1)).sum())
    return out


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--quantiles", action="store_true")
    args = ap.parse_args()

    print(f"{', '.join(args.symbols)} · {args.interval} · "
          f"{cfg.require('backtest.start')} → {cfg.require('backtest.end')}")
    feats = load_features(args.symbols, args.interval, cfg)
    print(f"{len(feats):,}봉 로드\n")

    if args.quantiles:
        q = feats[ANCHOR_COLS].quantile([.1, .25, .5, .75, .9]).round(3)
        print("정규화 앵커용 분위수")
        print(q.to_string())
        print("\nRegimeParams 권장값")
        print(f"  adx_lo={q.loc[.25,'adx']:.1f}  adx_hi={q.loc[.9,'adx']:.1f}")
        print(f"  eff_lo={q.loc[.25,'efficiency_ratio']:.3f}  eff_hi={q.loc[.9,'efficiency_ratio']:.3f}")
        print(f"  chop_lo={q.loc[.1,'choppiness']:.1f}  chop_hi={q.loc[.75,'choppiness']:.1f}")
        return 0

    base = RegimeParams()
    rows = []
    for score_min in (0.30, 0.35, 0.40, 0.45, 0.50):
        for confirm in (2, 3, 5, 8):
            p = replace(base, trend_score_min=score_min, confirm_bars=confirm)
            per_symbol = [
                classify(g.drop(columns="symbol"), p)
                for _, g in feats.groupby("symbol", sort=True)
            ]
            stats = {}
            for reg in per_symbol:
                for k, v in describe(reg).items():
                    stats.setdefault(k, []).append(v)
            row = {"점수임계": score_min, "확정봉": confirm}
            for k, vals in stats.items():
                row[k] = round(sum(vals) / len(vals), 1) if "%" in k else int(sum(vals) / len(vals))
            rows.append(row)

    table = pd.DataFrame(rows)
    table["추세합_%"] = (table["상승추세_%"] + table["하락추세_%"]).round(1)
    cols = ["점수임계", "확정봉", "추세합_%", "상승추세_%", "하락추세_%", "횡보_%",
            "고변동_%", "추세중위봉", "추세런수", "전환수"]
    print("스윕 결과 (3종목 평균)")
    print(table[cols].to_string(index=False))

    print("\n판단 기준: 추세합 30~40%, 횡보 45~55%, 추세중위봉 >= 15")
    good = table[
        table["추세합_%"].between(30, 40)
        & table["횡보_%"].between(45, 55)
        & (table["추세중위봉"] >= 15)
    ]
    if good.empty:
        print("→ 기준을 만족하는 조합이 없습니다. 앵커나 기준 자체를 재검토해야 합니다.")
        print("  가장 가까운 조합:")
        table["거리"] = (
            (table["추세합_%"] - 35).abs() / 5
            + (table["횡보_%"] - 50).abs() / 5
            + (15 - table["추세중위봉"]).clip(lower=0) / 5
        )
        print(table.nsmallest(3, "거리")[cols].to_string(index=False))
    else:
        print(good[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
