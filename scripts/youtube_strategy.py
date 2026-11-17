"""
유튜브 영상의 트레이딩 전략을 FX 엔진의 투표 후보로 저장한다.

    python scripts/youtube_strategy.py "https://www.youtube.com/watch?v=..."
    python scripts/youtube_strategy.py "<URL>" --name my_strategy --overwrite
    python scripts/youtube_strategy.py "<URL>" --enable      저장하고 바로 켠다
    python scripts/youtube_strategy.py --list                저장된 전략
    python scripts/youtube_strategy.py --on  my_strategy     켜기
    python scripts/youtube_strategy.py --off my_strategy     끄기

상황판(FX 상황판 → 유튜브 전략 카드)에서 하는 일과 같다. 저장된 전략은 꺼져
있고, 켜면 엔진의 다음 바퀴부터 투표에 들어간다. 발언권은 최근 학습 구간
성적이 정한다 — 거래가 적거나 기대 R 이 0 이하면 가중치는 0 이다.

GEMINI_API_KEY 또는 OPENAI_API_KEY 환경변수가 필요하다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                        # noqa: E402
from src.youtube import pipeline, store                   # noqa: E402

LINE = "=" * 68


def report(record: dict) -> str:
    spec = record["spec"]
    used = ", ".join(
        f"{i['name']}({', '.join(f'{k}={v}' for k, v in i['params'].items())})"
        for i in spec["indicators_needed"])
    exits = spec["exit_conditions"]
    notes = [f"   - {n}" for n in spec["quant_assumptions"]] or ["   (없음)"]
    return "\n".join([
        LINE,
        f" 전략명      : {spec['strategy_name']}  (엔진 이름 yt_{spec['strategy_name']})",
        f" 대상 시장   : {spec['target_market']}",
        f" 타임프레임  : {spec['timeframe']}",
        f" 사용 지표   : {used or '(없음)'}",
        f" 롱 진입     : {spec['entry_conditions']['long'] or '(없음)'}",
        f" 숏 진입     : {spec['entry_conditions']['short'] or '(없음)'}",
        f" 익절 / 손절 : {exits['take_profit_pct']} / {exits['stop_loss_pct']}  (기록만 — 엔진은 ATR 손절을 쓴다)",
        f" 지표 청산   : {exits['indicator_exit'] or '(없음)'}",
        " 수치화한 가정:", *notes,
        LINE,
        f" 상태        : {'켜짐 — 다음 바퀴부터 투표' if record['enabled'] else '꺼짐'}",
        f" 출처        : {record['source_url']}",
        LINE,
    ])


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="유튜브 전략을 FX 엔진 투표 후보로 저장")
    ap.add_argument("url", nargs="?", help="유튜브 영상 URL")
    ap.add_argument("--name", help="전략 이름을 직접 지정 (영문 소문자 스네이크케이스)")
    ap.add_argument("--overwrite", action="store_true", help="같은 이름이 있으면 덮어쓴다")
    ap.add_argument("--enable", action="store_true", help="저장하고 바로 켠다")
    ap.add_argument("--list", action="store_true", help="저장된 전략 목록")
    ap.add_argument("--on", metavar="NAME", help="전략 켜기")
    ap.add_argument("--off", metavar="NAME", help="전략 끄기")
    args = ap.parse_args()
    cfg = load_config()

    try:
        if args.list:
            records = store.load_all(cfg)
            for r in records:
                s = r["spec"]
                print(f" {'[켜짐]' if r['enabled'] else '[꺼짐]'} {s['strategy_name']:<40} "
                      f"{s['timeframe']:<4} {r['source_url']}")
            if not records:
                print(" 저장된 유튜브 전략이 없습니다.")
            return 0
        if args.on or args.off:
            record = store.set_enabled(args.on or args.off, bool(args.on), cfg)
            print(report(record))
            return 0
        if not args.url:
            ap.error("URL 이 필요합니다 (또는 --list / --on / --off)")
        record = pipeline.run(args.url, name=args.name, overwrite=args.overwrite,
                              cfg=cfg, log=lambda m: print(m, file=sys.stderr))
        if args.enable:
            record = store.set_enabled(record["spec"]["strategy_name"], True, cfg)
        print(report(record))
        return 0
    except (pipeline.PipelineError, store.StoreError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:                 # 네트워크·LLM API 오류
        print(f"오류: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
