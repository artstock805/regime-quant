"""
[대체됨] 강건성 검정 → `scripts/validate.py`

이 스크립트가 하던 다섯 검정(통합·부호·다중검정 보정·창별 일관성·Deflated Sharpe)은
`validate.py` 가 전부 포함하고, 순열 검정·파라미터 고원·비용 민감도·워크포워드 효율·
데이터 감사까지 더한다. 결과도 파일로 남는다.

같은 통계 함수를 두 곳에 두면 언젠가 갈라지고, 그러면 어느 쪽 숫자를 믿어야 할지
알 수 없게 된다. 그래서 로직을 `src/validation/` 으로 옮기고 여기는 안내만 남겼다.

    python scripts/validate.py --quick        # 이 스크립트와 같은 범위 (몇 분)
    python scripts/validate.py                # 전체
    python scripts/validate.py --compare      # 지난 실행들 비교
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "scripts" / "validate.py"

MESSAGE = """
robustness.py 는 validate.py 로 대체됐습니다.

  python scripts/validate.py --quick     이 스크립트와 같은 범위 (순열 생략)
  python scripts/validate.py             전체 — 순열·민감도·효율·감사 포함
  python scripts/validate.py --compare   지난 실행들 나란히 비교

--quick 으로 대신 실행합니다. 인자는 그대로 전달됩니다.
"""


def main() -> int:
    print(MESSAGE.strip(), flush=True)
    args = [a for a in sys.argv[1:] if a not in ("--quick",)]
    return subprocess.call([sys.executable, str(TARGET), "--quick", *args])


if __name__ == "__main__":
    raise SystemExit(main())
