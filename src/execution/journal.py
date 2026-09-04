"""
매매 장부 — 무엇을 왜 했는지 남긴다.

자동매매에서 사후에 답해야 하는 질문은 늘 같다. "그때 왜 들어갔지?",
"이 주문 두 번 나간 거 아냐?", "어제 얼마 잃었지?" 장부가 없으면 답할 수 없고,
답할 수 없으면 고칠 수도 없다.

**의도와 결과를 따로 적는다.** 주문을 내려고 했는데 가드가 막은 것도 기록한다.
막힌 기록이 없으면 "왜 진입 안 했지"에 답할 수 없다.

JSONL 로 쌓는다. 줄 단위라 중간에 죽어도 앞부분이 살아 있고, 사람이 그냥 읽을 수 있다.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PATH = ROOT / "runs" / "live" / "journal.jsonl"

# 기록하는 사건. 이 목록에 없는 종류는 쓰지 않는다 — 나중에 집계할 때
# 오타로 만든 종류가 섞여 있으면 숫자가 조용히 틀어진다.
KINDS = {
    "engine",        # 시작·정지·모드 전환
    "signal",        # 신호가 나왔다 (아직 주문 아님)
    "blocked",       # 가드가 막았다
    "order",         # 주문을 냈다
    "reject",        # 거래소가 거부했다
    "position",      # 포지션 확인 결과 (거래소 기준)
    "close",         # 청산됐다 (실현손익 포함)
    "error",         # 예외
    "reconcile",     # 내 기록과 거래소가 어긋났다
}


class Journal:
    """스레드 안전한 추가 전용 장부."""

    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, kind: str, **fields: Any) -> dict:
        if kind not in KINDS:
            raise ValueError(f"모르는 사건 종류: {kind} (가능: {sorted(KINDS)})")
        row = {"at": datetime.now(timezone.utc).isoformat(), "kind": kind, **fields}
        line = json.dumps(row, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return row

    # ─── 읽기 ────────────────────────────────────────────────────

    def tail(self, n: int = 200, kinds: Iterable[str] | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        want = set(kinds) if kinds else None
        rows: list[dict] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue          # 쓰다 만 줄. 버리되 앞은 살린다
                if want and row.get("kind") not in want:
                    continue
                rows.append(row)
        return rows[-n:][::-1]

    def realised_today(self) -> float:
        """오늘 실현손익. 가드가 일일 한도를 볼 때 쓴다."""
        today = datetime.now(timezone.utc).date().isoformat()
        total = 0.0
        for row in self.tail(5000, kinds=["close"]):
            if str(row.get("at", ""))[:10] == today:
                total += float(row.get("realised", 0) or 0)
        return total

    def summary(self) -> dict:
        rows = self.tail(5000)
        closes = [r for r in rows if r.get("kind") == "close"]
        wins = [r for r in closes if float(r.get("realised", 0) or 0) > 0]
        return {
            "events": len(rows),
            "orders": sum(1 for r in rows if r.get("kind") == "order"),
            "blocked": sum(1 for r in rows if r.get("kind") == "blocked"),
            "errors": sum(1 for r in rows if r.get("kind") == "error"),
            "closed": len(closes),
            "wins": len(wins),
            "win_rate": (len(wins) / len(closes)) if closes else None,
            "realised_total": round(
                sum(float(r.get("realised", 0) or 0) for r in closes), 4),
            "realised_today": round(self.realised_today(), 4),
        }
