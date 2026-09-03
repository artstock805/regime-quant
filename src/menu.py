"""
바탕화면 아이콘이 실행하는 메뉴.

    python -m src.menu

터미널을 모르는 사람이 아이콘을 눌러 들어오는 입구다. 그래서 두 가지를 지킨다.

  · **첫 화면에 판정을 박는다.** 이 도구는 검증 5항목 중 1개만 통과했다.
    메뉴를 먼저 보여 주고 경고를 아래에 숨기면 아무도 읽지 않는다.
  · **주문 메뉴가 없다.** 없는 기능을 회색으로 보여 주면 "언젠가 켜지는 것"으로
    읽힌다. 코드에 아예 없다는 사실을 화면에서도 그대로 둔다.
"""

from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

BANNER = """
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   REGIME QUANT  ·  레짐 기반 암호화폐 백테스트 · 신호 조회 도구          ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   검증 결과 (129종목 · 7,496 거래 · 2021-12 ~ 2026-08)                   ║
║                                                                          ║
║     총우위   +0.0260R   군집 t 3.52  (p=0.0004)   → 우위는 실재한다       ║
║     순우위   +0.0106R   군집 t 1.42  (p=0.154 )   → 0과 구분 안 된다      ║
║     비용이 총우위의 59% 를 먹는다                                        ║
║                                                                          ║
║     판정  1/5 통과      Deflated Sharpe  0.865  (기준 0.95)              ║
║                                                                          ║
║   ▶ 이 도구로 실매매하지 마십시오. 주문 기능은 코드에 없습니다.          ║
║     백테스트와 신호 조회 전용입니다.                                     ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""".strip("\n")

MENU = """
  [1]  실시간 신호 보기            지금 진입·청산 신호가 있는지 (주문 안 함)
  [2]  실시간 신호 계속 감시        5분마다 자동 갱신
  [3]  상황판 열기                 브라우저로 백테스트 재생 화면

  [4]  데이터 받기                 캔들 + 파생 (처음 한 번, 시간이 걸립니다)
  [5]  백테스트 검증 실행           5항목 판정을 직접 재현
  [6]  지표 성적표                 어떤 보조지표가 값을 하는지 + 순위 신뢰도

  [7]  지난 결과 보기              최종 리포트 (HTML)
  [8]  자체 점검                   테스트 전체 실행

  [0]  종료
"""


def run(*args: str, note: str = "") -> None:
    if note:
        print(f"\n  {note}\n")
    print(f"  $ {' '.join(['python', *args])}\n")
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        subprocess.run([PY, *args], cwd=ROOT, env=env, check=False)
    except KeyboardInterrupt:
        print("\n  중단했습니다.")


def data_ready() -> bool:
    from src.config import load_config
    kd = load_config().path("paths.klines_dir")
    return kd.exists() and any(kd.rglob("*.parquet"))


def ask(prompt: str, default: str = "") -> str:
    try:
        got = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return default
    return got or default


def main() -> int:
    while True:
        print("\n" + BANNER)
        if not data_ready():
            print("\n  ⚠ 데이터가 없습니다. 먼저 [4] 데이터 받기를 실행하세요.")
            print("    (신호 조회는 데이터 없이도 동작합니다 — 그때그때 내려받습니다)")
        print(MENU)
        choice = ask("  번호를 고르세요 > ")

        if choice == "0" or choice.lower() in {"q", "exit", "quit"}:
            print("\n  종료합니다.\n")
            return 0

        elif choice == "1":
            run("scripts/signals.py",
                note="마감된 봉 기준으로 신호를 계산합니다. 종목당 몇 초 걸립니다.")

        elif choice == "2":
            sec = ask("  갱신 간격(초, 기본 300) > ", "300")
            run("scripts/signals.py", "--watch", sec, "--only-actionable",
                note="Ctrl+C 로 중단합니다.")

        elif choice == "3":
            if not data_ready():
                print("\n  상황판은 백테스트 결과를 재생하므로 데이터가 필요합니다.")
                print("  [4] 데이터 받기를 먼저 실행하세요.")
                ask("\n  엔터를 누르면 메뉴로 돌아갑니다 ")
                continue
            print("\n  브라우저에서 http://127.0.0.1:8787 을 엽니다.")
            print("  창을 닫으려면 이 터미널에서 Ctrl+C 를 누르세요.\n")
            try:
                webbrowser.open("http://127.0.0.1:8787")
            except Exception:
                pass
            run("-m", "src.server.run", "--prewarm")

        elif choice == "4":
            print("\n  얼마나 받을까요?")
            print("    [a] 8종목  · 1시간봉만        약 30MB · 2분")
            print("    [b] 8종목  · 1시간+1분+파생   약 1GB  · 20분")
            print("    [c] 129종목 · 전체            약 16GB · 3시간  (권장하지 않음)")
            scope = ask("  선택 (a/b/c, 기본 a) > ", "a").lower()
            if scope == "c":
                print("\n  ⚠ 16GB 를 받습니다. 디스크 여유를 먼저 확인하세요.")
                if ask("  계속하려면 'yes' 를 입력 > ").lower() != "yes":
                    continue
                run("scripts/fetch_universe.py", "--workers", "5")
            elif scope == "b":
                run("scripts/fetch_data.py", "--intervals", "1h", "1m")
                run("scripts/fetch_derivatives.py")
            else:
                run("scripts/fetch_data.py", "--intervals", "1h")

        elif choice == "5":
            if not data_ready():
                print("\n  데이터가 필요합니다. [4] 를 먼저 실행하세요.")
                ask("\n  엔터를 누르면 메뉴로 돌아갑니다 ")
                continue
            print("\n  [q] 빠른 검증 (몇 분)   [f] 전체 검증 (순열 포함, 30분+)")
            mode = ask("  선택 (q/f, 기본 q) > ", "q").lower()
            args = ["scripts/validate.py", "--derivatives", "--calibrate-strength"]
            if mode != "f":
                args.append("--quick")
            run(*args, note="설정 파일의 기본값(메이커·상위15%)으로 돌립니다.")

        elif choice == "6":
            if not data_ready():
                print("\n  데이터가 필요합니다. [4] 를 먼저 실행하세요.")
                ask("\n  엔터를 누르면 메뉴로 돌아갑니다 ")
                continue
            run("scripts/scoreboard.py", "--derivatives",
                note="지속성을 먼저 봅니다 — 순위보다 그게 중요합니다.")

        elif choice == "7":
            report = ROOT / "runs" / "final" / "report.html"
            if report.exists():
                print(f"\n  {report}")
                webbrowser.open(report.as_uri())
            else:
                print("\n  저장된 리포트가 없습니다. [5] 검증을 먼저 실행하세요.")
            ask("\n  엔터를 누르면 메뉴로 돌아갑니다 ")

        elif choice == "8":
            run("-m", "pytest", "-q", note="전체 테스트를 실행합니다.")

        else:
            print("\n  그런 번호는 없습니다.")
            continue

        ask("\n  엔터를 누르면 메뉴로 돌아갑니다 ")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
