"""
검증 파이프라인 — 한 명령으로 전 검정을 돌리고 리포트를 남긴다.

    python scripts/validate.py                      # 전체 (순열 포함, 느림)
    python scripts/validate.py --quick              # 순열·민감도·효율 생략
    python scripts/validate.py --audit-only         # 데이터 감사만
    python scripts/validate.py --entry-threshold 0.79
    python scripts/validate.py --compare            # 지난 실행들 나란히 비교

결과는 `runs/<타임스탬프>/report.{json,html}` 에 남는다. 설정 스냅샷과 환경
정보(파이썬·pandas 버전·git 커밋)가 함께 들어가므로, 나중에 "그때 무슨 설정으로
그 숫자가 나왔나"에 답할 수 있다.

`robustness.py` 와 `significance.py` 가 하던 일을 전부 포함한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                                    # noqa: E402
from src.validation import audit_summary, compare_reports, rerender  # noqa: E402
from src.validation.audit import audit_cache                          # noqa: E402
from src.validation.suite import SuiteConfig, run_suite               # noqa: E402

pd.set_option("display.width", 240)


def mode_audit(args, cfg) -> int:
    symbols = args.symbols or cfg.require("universe")
    intervals = args.intervals or [args.interval, "1m"]
    print(f"데이터 감사 · {len(symbols)}종목 × {intervals}\n")
    tbl = audit_cache(cfg, symbols, intervals)
    cols = ["symbol", "interval", "bars", "missing", "coverage",
            "duplicates", "ohlc_violations", "nonpositive", "outliers", "ok"]
    print(tbl[cols].to_string(index=False))
    s = audit_summary(tbl)
    print(f"\n총 {s['total_bars']:,}봉 · 누락 {s['total_missing']:,} · "
          f"이상치 {s['total_outliers']} · 최소 커버리지 {s['min_coverage']:.4f}")
    print("구조적 문제 없음" if s["ok"] else f"⚠ {', '.join(s['problems'])}")
    return 0 if s["ok"] else 1


def mode_compare(args, cfg) -> int:
    runs_dir = cfg.path("paths.runs_dir")
    if not runs_dir.exists():
        print(f"실행 기록이 없습니다: {runs_dir}")
        return 1
    dirs = sorted((p for p in runs_dir.iterdir()
                   if p.is_dir() and (p / "report.json").exists()),
                  key=lambda p: p.name)
    if not dirs:
        print("비교할 리포트가 없습니다. 먼저 python scripts/validate.py 를 돌리세요.")
        return 1
    picked = dirs[-args.last:]
    print(f"최근 {len(picked)}개 실행 비교\n")
    print(compare_reports(picked).to_string(index=False))
    return 0


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="레짐 퀀트 검증 파이프라인")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--intervals", nargs="*", default=None, help="--audit-only 에서만")
    ap.add_argument("--entry-threshold", type=float, default=None)
    ap.add_argument("--fee-rate", type=float, default=None,
                    help="수수료 override. 메이커 지정가라면 0.0002")
    ap.add_argument("--quick", action="store_true",
                    help="순열·민감도·효율을 건너뛴다 (몇 분)")
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--permutation-symbols", type=int, default=3)
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--rerender", default=None, metavar="RUN",
                    help="저장된 JSON 에서 HTML 만 다시 만든다 (계산 재실행 없음)")
    ap.add_argument("--last", type=int, default=5, help="--compare 에서 볼 실행 수")
    ap.add_argument("--name", default=None, help="리포트 폴더 이름 (기본: 타임스탬프)")
    ap.add_argument("--strategies", nargs="*", default=None,
                    help="config 의 strategies 를 덮어쓴다")
    ap.add_argument("--calibrate-strength", action="store_true",
                    help="확신도를 자기 최근 분포의 백분위로 바꾼다. 전략이 선언한 "
                         "진입 조건과 앙상블 문턱의 척도를 맞춘다")
    ap.add_argument("--derivatives", action="store_true",
                    help="미결제약정·펀딩·롱숏비율을 붙인다. 아카이브가 2021-01 부터라 "
                         "표본이 짧아지는 대신 파생 지표가 살아난다")
    args = ap.parse_args()

    if args.rerender:
        runs = cfg.path("paths.runs_dir")
        target = Path(args.rerender)
        if not target.exists():
            target = runs / args.rerender
        out = rerender(target)
        print(f"HTML 재생성: {out}")
        return 0
    if args.strategies is not None:
        cfg.raw["strategies"] = list(args.strategies)
    if args.compare:
        return mode_compare(args, cfg)
    if args.audit_only:
        return mode_audit(args, cfg)

    sc = SuiteConfig(
        symbols=args.symbols or cfg.require("universe"),
        interval=args.interval,
        entry_threshold=args.entry_threshold,
        fee_rate=args.fee_rate,
        run_permutation=not args.quick,
        run_sensitivity=not args.quick,
        run_efficiency=not args.quick,
        n_permutations=args.permutations,
        permutation_symbols=args.permutation_symbols,
        use_derivatives=args.derivatives,
        calibrate_strength=args.calibrate_strength,
    )

    print(f"{cfg.require('market')} · {sc.interval} · "
          f"{cfg.require('backtest.start')} → {cfg.require('backtest.end')}")
    print(f"전략 {cfg.get('strategies') or '전부'} · {len(sc.symbols)}종목"
          + (" · 파생 데이터 ON" if sc.use_derivatives else ""))
    if args.quick:
        print("빠른 모드 — 순열·민감도·효율 생략\n")
    else:
        print(f"전체 모드 — 순열 {sc.n_permutations}회 x {sc.permutation_symbols}종목\n")

    report, _ = run_suite(cfg, sc, log=lambda m: print(m, flush=True))
    out = report.save(cfg.path("paths.runs_dir"), name=args.name,
                      trades=getattr(report, "trades", None))
    print(f"\n리포트 저장: {out}")
    print(f"  {out / 'report.html'}")
    return 0 if report.passed == len(report.checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
