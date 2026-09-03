"""
실시간 진입·청산 신호. **주문은 내지 않는다.**

    python scripts/signals.py                        기본 종목 한 번 조회
    python scripts/signals.py --symbols BTCUSDT ETHUSDT SOLUSDT
    python scripts/signals.py --watch 300            5분마다 반복
    python scripts/signals.py --only-actionable      진입·청산만 출력
    python scripts/signals.py --json runs/signals.json

신호는 **마감된 봉** 기준이고 체결은 **다음 봉 시가**를 가정한다. 백테스트와 같은
규칙이라 화면의 신호와 검증 숫자가 같은 것을 가리킨다.

주문 기능이 없는 것은 미완성이 아니다. 5.9년 검증 5항목 중 1개만 통과했고
Deflated Sharpe 는 0.865(기준 0.95)다. 신호를 보는 것과 그 신호로 돈을 거는 것은
다른 결정이고, 후자는 사람이 해야 한다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                        # noqa: E402
from src.live import actionable, scan                     # noqa: E402

pd.set_option("display.width", 220)

BANNER = """
┌──────────────────────────────────────────────────────────────────────────┐
│  실시간 신호 · 조회 전용 — 주문 기능 없음                                │
│  검증 판정 1/5 · Deflated Sharpe 0.865 (기준 0.95)                       │
│  이 신호로 매매할지는 사람이 판단합니다.                                 │
└──────────────────────────────────────────────────────────────────────────┘
""".strip("\n")

ARROW = {"롱진입": "▲", "숏진입": "▼", "청산": "■", "유지": "·", "관망": " "}


def render(signals, only_actionable: bool) -> None:
    if not signals:
        print("  신호를 만들 수 있는 종목이 없습니다 (데이터 부족 또는 네트워크)")
        return
    shown = actionable(signals) if only_actionable else signals
    if not shown:
        print(f"  진입·청산 신호 없음 ({len(signals)}종목 조회, 전부 유지/관망)")
        return

    rows = []
    for s in shown:
        rows.append({
            "": ARROW.get(s.action, "?"),
            "종목": s.symbol, "동작": s.action,
            "점수": round(s.score, 3), "문턱": s.threshold,
            "국면": s.regime_korean, "확신": s.regime_confidence,
            "종가": s.close,
            "손절": s.stop_price, "목표": s.target_price,
            "1R": s.risk_per_unit,
        })
    print(pd.DataFrame(rows).to_string(index=False))

    acts = actionable(shown)
    if acts:
        print(f"\n  ▶ 상태가 바뀐 종목 {len(acts)}개")
        for s in acts:
            voice = [v for v in s.votes if v.has_voice]
            who = ", ".join(f"{v.korean}({v.contribution:+.2f})" for v in voice) or "없음"
            print(f"    {s.symbol:<12} {s.action:<6} 근거: {who}")
            print(f"      마감봉 {s.bar_time[:16]} → 체결가정 {s.next_bar_time[:16]}")


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="실시간 진입·청산 신호 (주문 없음)")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--universe", default=None, help="종목 목록 JSON")
    ap.add_argument("--interval", default=None)
    ap.add_argument("--no-derivatives", action="store_true",
                    help="파생 데이터를 붙이지 않는다 (파생 전략은 신호가 없어진다)")
    ap.add_argument("--only-actionable", action="store_true")
    ap.add_argument("--watch", type=int, default=0, metavar="초",
                    help="이 간격으로 반복 조회한다 (0=한 번만)")
    ap.add_argument("--json", default=None, help="결과를 이 경로에 JSON 으로 쓴다")
    args = ap.parse_args()

    interval = args.interval or cfg.get("live.interval", "1h")
    if args.universe:
        p = Path(args.universe)
        symbols = json.loads((p if p.is_absolute() else ROOT / p).read_text())
    else:
        symbols = args.symbols or list(cfg.get("live.symbols")
                                       or cfg.require("universe"))

    print(BANNER)
    print(f"\n{len(symbols)}종목 · {interval} · 전략 {cfg.get('strategies') or '전부'}"
          + ("" if args.no_derivatives else " · 파생 데이터 ON"))

    while True:
        started = datetime.now(timezone.utc)
        print(f"\n[{started:%Y-%m-%d %H:%M:%S} UTC] 조회 중")
        sigs = scan(cfg, symbols, interval=interval,
                    use_derivatives=not args.no_derivatives,
                    log=(print if len(symbols) <= 20 else None))
        print()
        render(sigs, args.only_actionable)

        if args.json:
            out = Path(args.json)
            if not out.is_absolute():
                out = ROOT / out
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "created": started.isoformat(),
                "interval": interval,
                "order_execution": False,
                "verdict": "1/5 · 실전 투입 근거 아님",
                "signals": [s.as_dict() for s in sigs],
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n  저장: {out}")

        if not args.watch:
            return 0
        print(f"\n  {args.watch}초 후 다시 조회합니다 (Ctrl+C 로 중단)")
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n중단했습니다")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
