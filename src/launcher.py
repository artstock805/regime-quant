"""
바탕화면 아이콘이 실제로 실행하는 진입점.

**왜 파이썬이 진입점인가.** 배치파일에 한글을 넣으면 cmd.exe 가 그것을 시스템
ANSI 코드페이지로 읽어 깨뜨리고, 깨진 바이트를 명령어로 파싱해서
`'st'은(는) 명령이 아닙니다` 같은 오류를 낸다. 한국어 Windows 가 아니면 더 심하다.
그래서 `windows\\start.bat` 은 **영문 ASCII 만** 담고, 사람이 읽는 것은 전부
여기서 출력한다. 인코딩을 제어할 수 있는 쪽에서 처리해야 한다.

하는 일
    1) 콘솔을 UTF-8 로 맞춘다
    2) 필요한 패키지가 있는지 본다 — 없으면 그 자리에서 설치를 제안한다
    3) 가상환경(.venv)이 있으면 그쪽 파이썬으로 다시 띄운다
    4) **웹 서버를 켜고 브라우저를 연다**

처음 아이콘을 눌렀을 때 "설치부터 하세요"라고 돌려보내지 않는 것,
그리고 터미널 메뉴가 아니라 **화면**이 뜨는 것이 목적이다.
터미널 메뉴가 필요하면 `--menu` 를 준다.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQUIRED = ["pandas", "numpy", "pyarrow", "requests", "yaml"]
MIN_PYTHON = (3, 11)


def setup_console() -> None:
    """윈도우 콘솔을 UTF-8 로. 이걸 안 하면 한글이 물음표로 나온다."""
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def missing_packages(python: str | None = None) -> list[str]:
    """없는 패키지 목록. python 을 주면 그 인터프리터에서 확인한다."""
    if python:
        code = ("import importlib,sys;"
                f"print(','.join(m for m in {REQUIRED!r} "
                "if importlib.util.find_spec(m) is None))")
        try:
            out = subprocess.run([python, "-c", code], capture_output=True,
                                 text=True, timeout=60).stdout.strip()
        except Exception:
            return list(REQUIRED)
        return [m for m in out.split(",") if m]
    import importlib.util
    return [m for m in REQUIRED if importlib.util.find_spec(m) is None]


def ask(prompt: str, default: str = "") -> str:
    try:
        return input(prompt).strip() or default
    except (EOFError, KeyboardInterrupt):
        return default


def install(python: str) -> bool:
    """가상환경을 만들고 패키지를 넣는다. 성공하면 True."""
    print("\n  [1/2] 가상환경을 만듭니다 (.venv)")
    if not VENV_PY.exists():
        r = subprocess.run([python, "-m", "venv", str(ROOT / ".venv")], cwd=ROOT)
        if r.returncode != 0 or not VENV_PY.exists():
            print("\n  가상환경을 만들지 못했습니다.")
            print("  파이썬이 제대로 설치됐는지 확인하세요 (python -m venv --help).")
            return False
    else:
        print("        이미 있습니다 — 건너뜁니다")

    print("  [2/2] 패키지를 설치합니다. 몇 분 걸립니다.\n")
    subprocess.run([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip", "-q"],
                   cwd=ROOT)
    r = subprocess.run([str(VENV_PY), "-m", "pip", "install",
                        "-r", str(ROOT / "requirements.txt")], cwd=ROOT)
    if r.returncode != 0:
        print("\n  패키지 설치에 실패했습니다. 인터넷 연결을 확인하세요.")
        return False
    print("\n  설치가 끝났습니다.")
    return True


def main(argv: list[str] | None = None) -> int:
    setup_console()
    args = sys.argv[1:] if argv is None else argv
    install_only = "--install" in args
    text_menu = "--menu" in args
    # --serve-only: 브라우저를 열지 않는다 (외부 도구가 띄울 때)
    no_browser = "--serve-only" in args

    if sys.version_info < MIN_PYTHON:
        print(f"\n  Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} 이상이 필요합니다 "
              f"(현재 {sys.version.split()[0]}).")
        print("  https://www.python.org/downloads/ 에서 최신판을 설치하세요.")
        ask("\n  엔터를 누르면 닫힙니다 ")
        return 1

    # ── windows/install.bat 경로: 설치만 하고 끝낸다
    if install_only:
        print("\n" + "=" * 70)
        print("  Regime Quant · 설치")
        print("=" * 70)
        ok = install(sys.executable)
        if ok:
            print("\n  바탕화면 아이콘을 만들려면 windows\\create-shortcut.bat 을,")
            print("  바로 실행하려면 windows\\start.bat 을 실행하세요.")
        return 0 if ok else 1

    # ── 가상환경이 있으면 그쪽으로 넘긴다 (지금이 시스템 파이썬일 때만)
    here = Path(sys.executable).resolve()
    if VENV_PY.exists() and here != VENV_PY.resolve():
        if not missing_packages(str(VENV_PY)):
            os.environ.setdefault("PYTHONUTF8", "1")
            return subprocess.run([str(VENV_PY), "-m", "src.launcher", *args],
                                  cwd=ROOT).returncode

    # ── 패키지 확인
    gone = missing_packages()
    if gone:
        print("\n" + "=" * 70)
        print("  Regime Quant · 첫 실행 준비")
        print("=" * 70)
        print(f"\n  필요한 패키지가 없습니다: {', '.join(gone)}")
        print("  지금 설치할까요? .venv 폴더에만 설치하며 시스템은 건드리지 않습니다.")
        print("  (약 300MB, 몇 분 소요)")
        if ask("\n  설치하려면 y, 취소하려면 엔터 > ").lower() not in {"y", "yes"}:
            print("\n  취소했습니다. 나중에 windows\\install.bat 을 실행해도 됩니다.")
            ask("\n  엔터를 누르면 닫힙니다 ")
            return 1
        if not install(sys.executable):
            ask("\n  엔터를 누르면 닫힙니다 ")
            return 1
        os.environ.setdefault("PYTHONUTF8", "1")
        return subprocess.run([str(VENV_PY), "-m", "src.launcher", *args],
                              cwd=ROOT).returncode

    # ── 준비 완료
    sys.path.insert(0, str(ROOT))
    if text_menu:
        from src.menu import main as menu_main
        return menu_main()
    return serve(open_browser=not no_browser)


def serve(host: str = "127.0.0.1", port: int = 8787,
          open_browser: bool = True) -> int:
    """웹 화면을 켜고 브라우저를 연다."""
    import socket
    import threading
    import webbrowser

    # 포트가 이미 쓰이면 다음 포트를 찾는다. "이미 켜져 있는데 안 열린다"가
    # 가장 흔한 혼란이라, 조용히 실패하지 않고 옮겨서 연다.
    for p in range(port, port + 12):
        with socket.socket() as s:
            if s.connect_ex((host, p)) != 0:
                port = p
                break
    else:
        print(f"\n  {host}:{port}~ 포트가 모두 사용 중입니다.")
        return 1

    url = f"http://{host}:{port}"
    print("\n" + "=" * 70)
    print("  Regime Quant")
    print("=" * 70)
    print(f"\n  화면 주소  {url}")
    print("  이 창을 닫으면 서버도 함께 종료됩니다. (Ctrl+C 로도 종료)")
    print("\n  판정 1/5 · Deflated Sharpe 0.865 (기준 0.95)")
    print("  주문 실행 기능은 없습니다. 백테스트와 신호 조회 전용입니다.\n")

    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    import uvicorn
    from src.server.app import app
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
