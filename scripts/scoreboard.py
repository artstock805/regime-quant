"""
지표 성적표 — 정기 실행용.

    python scripts/scoreboard.py                      전 종목(config universe)
    python scripts/scoreboard.py --symbols BTCUSDT ETHUSDT SOLUSDT
    python scripts/scoreboard.py --universe runs/universe_final.json
    python scripts/scoreboard.py --derivatives        파생 지표까지 포함
    python scripts/scoreboard.py --show               계산 없이 지난 결과만 본다

결과는 `runs/scoreboard/<타임스탬프>.json` 에 쌓이고, 직전 스냅샷과 비교해
**새로 추가된 지표**와 순위 변동을 보여준다.

읽는 순서가 중요하다
    1) 맨 아래 '지속성' 을 먼저 본다. 무작위와 구분되지 않으면 위의 순위는 장식이다.
    2) 'Holm 보정 후 유의' 개수를 본다. 0 이면 상위 지표도 우연일 수 있다.
    3) 그 다음에 순위를 본다.

순위표를 먼저 보면 반드시 1등을 믿게 된다. 이 프로젝트는 그 실수를 이미 했다 —
2년 표본에서 BNB 샤프 1.32 를 보고 통한다고 판단했다가 5.9년에서 -0.33 이 나왔다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                                    # noqa: E402
from src.data.cache import get_klines                                 # noqa: E402
from src.data.derivatives import align_to_bars, get_derivatives       # noqa: E402
from src.indicators import compute_all                                # noqa: E402
from src.validation.scoreboard import (                               # noqa: E402
    ScoreboardConfig, aggregate, build_panel, diff_snapshots,
    make_snapshot, score_periods,
)

pd.set_option("display.width", 220)
OUT_DIR = ROOT / "runs" / "scoreboard"


def latest_snapshot() -> tuple[Path | None, dict | None]:
    if not OUT_DIR.exists():
        return None, None
    files = sorted(OUT_DIR.glob("*.json"))
    if not files:
        return None, None
    return files[-1], json.loads(files[-1].read_text(encoding="utf-8"))


def render(snap: dict, diff: dict, top: int) -> None:
    inds = pd.DataFrame(snap["indicators"])
    pers, stab = snap["persistence"], snap["top_stability"]

    print("=" * 104)
    print(f"지표 성적표 · {len(snap['symbols'])}종목 · 지표 {len(inds)}개 "
          f"· 지평 {snap['config']['horizon']}봉 · {snap['created'][:16]}")
    print("=" * 104)

    # ── 1) 지속성을 먼저 보여준다. 순위를 먼저 보면 1등을 믿게 된다.
    print("\n[1] 이 순위를 믿을 수 있는가")
    if pers.get("pairs"):
        print(f"  기간 쌍 {pers['pairs']}개 · 평균 순위상관 {pers['mean_rho']:+.3f} "
              f"· 순열 p={pers['p']:.3f}")
        print(f"  → {pers['verdict']}")
    else:
        print("  기간이 부족해 지속성을 잴 수 없습니다")
    if stab.get("pairs"):
        print(f"  상위 {stab['k']}개가 다음 기간에도 상위에 남는 비율 "
              f"{stab['hit_rate']*100:.0f}% (무작위 기준선 {stab['baseline']*100:.0f}%)")

    sig = inds[inds["significant"]] if "significant" in inds.columns else inds.iloc[:0]
    print(f"\n  Holm 보정 후 유의한 지표: **{len(sig)}개** / {len(inds)}개"
          + (f" — {list(sig['name'])}" if len(sig) else " (상위 지표도 우연일 수 있다)"))

    # ── 2) 순위
    print(f"\n[2] 상위 {top}개 (기간외 기대R)")
    cols = ["name", "expectancy_r", "t", "p_holm", "periods",
            "positive_periods", "direction_flips", "obs"]
    show = inds[cols].head(top).rename(columns={
        "name": "지표", "expectancy_r": "기대R", "p_holm": "Holm_p",
        "periods": "기간", "positive_periods": "양수기간",
        "direction_flips": "방향뒤집힘", "obs": "관측"})
    print(show.round({"기대R": 4, "t": 2, "Holm_p": 3}).to_string(index=False))

    print(f"\n[3] 하위 5개 (역방향으로 쓸 후보가 아니다 — 그냥 나쁜 것이다)")
    print(inds[cols].tail(5).rename(columns={
        "name": "지표", "expectancy_r": "기대R", "p_holm": "Holm_p",
        "periods": "기간", "positive_periods": "양수기간",
        "direction_flips": "방향뒤집힘", "obs": "관측"})
        .round({"기대R": 4, "t": 2, "Holm_p": 3}).to_string(index=False))

    # ── 3) 지난번과 비교
    print("\n[4] 지난 성적표와 비교")
    if diff.get("is_first"):
        print("  첫 실행입니다 — 다음 실행부터 새 지표와 순위 변동을 보여줍니다")
    else:
        if diff["new"]:
            print(f"  ⚠ 새로 추가된 지표 {len(diff['new'])}개: {diff['new']}")
            print("    지표를 추가하고 성적표를 다시 돌리는 것은 시험 횟수를 늘리는 "
                  "행위다. 아래 누적 시험 횟수를 보라.")
        if diff["gone"]:
            print(f"  사라진 지표: {diff['gone']}")
        big = [m for m in diff["moves"] if abs(m["delta"]) >= 10][:8]
        if big:
            print("  순위가 10계단 이상 움직인 지표:")
            for m in big:
                print(f"    {m['name']:<24} {m['before']:>3}위 → {m['after']:>3}위 "
                      f"({m['delta']:+d})")
            print("    이만큼 움직인다는 것 자체가 순위의 불안정성이다.")
        else:
            print("  10계단 이상 움직인 지표 없음")
        print(f"  누적 시험 횟수: {diff['trials_before']} → {diff['trials_after']}")

    print("\n" + "=" * 104)
    for n in snap.get("notes", []):
        print(f"  · {n}")


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="지표 성적표")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--universe", default=None,
                    help="종목 목록 JSON (예: runs/universe_final.json)")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--period-freq", default="Q", help="Q=분기, M=월, 2Q=반년")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--derivatives", action="store_true",
                    help="미결제약정·펀딩·롱숏비율 지표까지 채점한다")
    ap.add_argument("--show", action="store_true", help="계산 없이 지난 결과만 출력")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    if args.show:
        path, snap = latest_snapshot()
        if snap is None:
            print("저장된 성적표가 없습니다. 먼저 한 번 실행하세요.")
            return 1
        files = sorted(OUT_DIR.glob("*.json"))
        prev = (json.loads(files[-2].read_text(encoding="utf-8"))
                if len(files) >= 2 else None)
        print(f"(저장본: {path.name})")
        render(snap, diff_snapshots(prev, _as_snapshot(snap)), args.top)
        return 0

    if args.universe:
        p = Path(args.universe)
        symbols = json.loads((p if p.is_absolute() else ROOT / p).read_text())
    else:
        symbols = args.symbols or list(cfg.require("universe"))

    sc = ScoreboardConfig(horizon=args.horizon, period_freq=args.period_freq,
                          atr_stop_mult=float(cfg.require("risk.atr_stop_mult")))
    kd, market = cfg.path("paths.klines_dir"), cfg.require("market")
    start, end = cfg.require("backtest.start"), cfg.require("backtest.end")

    print(f"지표 성적표 · {len(symbols)}종목 · {args.interval} · "
          f"기간단위 {args.period_freq}" + (" · 파생 포함" if args.derivatives else ""))
    print("1분봉은 읽지 않는다 — 이 채점에는 봉 내부 판정이 필요 없다\n")

    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for i, sym in enumerate(symbols, 1):
        bars = get_klines(sym, args.interval, start, end,
                          market=market, klines_dir=kd)
        if bars.empty:
            continue
        if args.derivatives:
            d = get_derivatives(sym, start, end, market=market,
                                deriv_dir=cfg.path("paths.derivatives_dir"))
            bars = align_to_bars(bars, d, args.interval)
            if "open_interest" in bars.columns:
                bars = bars[bars["open_interest"].notna()]
        if len(bars) < 2000:
            continue
        frames[sym] = (bars, compute_all(bars))
        if i % 10 == 0 or i == len(symbols):
            print(f"  {i}/{len(symbols)} 로드", flush=True)

    if not frames:
        print("채점할 데이터가 없습니다")
        return 1

    panel = build_panel(frames, sc)
    frames.clear()                     # 봉·특징은 여기서 놓는다 (메모리)
    per_period = score_periods(panel, sc)
    n_ind = per_period["indicator"].nunique() if not per_period.empty else 0
    print(f"\n채점 완료 · 관측 {len(panel):,} · 지표 {n_ind}개 · "
          f"기간 {per_period['period'].nunique() if n_ind else 0}개")

    scores = aggregate(per_period, panel, sc)
    snap = make_snapshot(scores, per_period, list(panel["symbol"].unique()), sc)

    _, prev = latest_snapshot()
    diff = diff_snapshots(prev, snap)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = args.name or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = OUT_DIR / f"{stamp}.json"
    payload = snap.as_dict()
    payload["diff"] = diff
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    render(payload, diff, args.top)
    per_period.to_csv(OUT_DIR / f"{stamp}-periods.csv", index=False,
                      encoding="utf-8-sig")
    print(f"\n저장: {out}")
    print(f"      {OUT_DIR / f'{stamp}-periods.csv'} (기간별 원본)")
    return 0


def _as_snapshot(d: dict):
    """--show 경로에서 diff 를 재계산하기 위한 얇은 어댑터."""
    from src.validation.scoreboard import Snapshot
    return Snapshot(created=d["created"], config=d["config"], symbols=d["symbols"],
                    indicators=d["indicators"], persistence=d["persistence"],
                    top_stability=d["top_stability"], trials=d.get("trials", 0),
                    notes=d.get("notes", []))


if __name__ == "__main__":
    raise SystemExit(main())
