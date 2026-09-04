"""
배포 패키징 — 남의 컴퓨터에서 클릭 한 번으로 열리는가.

이 파일이 있는 이유는 실제로 두 번 깨졌기 때문이다. 둘 다 개발 폴더에서는
멀쩡하고 **받는 쪽에서만** 깨지는 종류라, 코드를 읽어서는 알 수 없었다.

  1) `.gitignore` 에 `data/` 라고 써서 `src/data/` 패키지가 통째로 저장소에서
     빠졌다. 클론한 쪽에서 ModuleNotFoundError 로 테스트 12개가 죽었다.

  2) 배치파일에 한글을 넣었더니 cmd.exe 가 시스템 ANSI 코드페이지로 읽어
     깨뜨렸고, 깨진 바이트를 명령어로 파싱해 `'st'은(는) 명령이 아닙니다` 를
     냈다. 바탕화면 아이콘이 아무것도 하지 않는 것처럼 보였다.

그래서 여기서 고정하는 것은 "코드가 맞는가"가 아니라 **"배포물이 온전한가"** 다.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WINDOWS = ROOT / "windows"


# ─── 배치파일 인코딩 ──────────────────────────────────────────────

@pytest.mark.parametrize("name", ["start.bat", "install.bat", "create-shortcut.bat"])
def test_batch_files_are_pure_ascii(name):
    """
    cmd.exe 는 배치파일을 **시스템 ANSI 코드페이지**로 읽는다. UTF-8 로 저장한
    한글은 깨지고, 깨진 바이트가 명령어로 파싱된다. 한국어 Windows 가 아니면
    더 심하다. 사람이 읽는 글은 전부 src/launcher.py 로 옮겨야 한다.
    """
    raw = (WINDOWS / name).read_bytes()
    bad = [(i, b) for i, b in enumerate(raw) if b > 0x7F]
    assert not bad, (
        f"{name}: 비ASCII 바이트 {len(bad)}개 (첫 위치 {bad[0][0]}). "
        "한글은 src/launcher.py 에서 출력하세요.")


@pytest.mark.parametrize("name", ["start.bat", "install.bat", "create-shortcut.bat"])
def test_batch_files_use_crlf(name):
    """LF 만 있는 배치파일은 일부 윈도우에서 마지막 줄을 먹는다."""
    raw = (WINDOWS / name).read_bytes()
    lone_lf = raw.replace(b"\r\n", b"").count(b"\n")
    assert lone_lf == 0, f"{name}: CRLF 가 아닌 줄바꿈이 {lone_lf}개 있습니다"


def test_start_bat_falls_back_when_venv_is_absent():
    """
    처음 받은 사람에게는 .venv 가 없다. 거기서 '설치부터 하세요'라고 끝내면
    아이콘이 아무것도 안 하는 것처럼 보인다. py -3 / python 순으로 물러서야 한다.
    """
    text = (WINDOWS / "start.bat").read_text(encoding="ascii")
    assert ".venv\\Scripts\\python.exe" in text
    assert "py -3" in text and "python --version" in text
    assert "src.launcher" in text


def test_start_bat_sets_utf8_environment():
    text = (WINDOWS / "start.bat").read_text(encoding="ascii")
    assert "PYTHONUTF8=1" in text
    assert "PYTHONIOENCODING=utf-8" in text


# ─── 저장소가 온전한가 ────────────────────────────────────────────

def test_no_source_package_is_excluded_by_gitignore():
    """
    `data/` 처럼 앞에 / 가 없는 규칙은 어느 깊이의 폴더든 매칭한다.
    실제로 `src/data/` 가 통째로 빠져 클론한 쪽에서 import 가 깨졌다.
    """
    if not (ROOT / ".git").exists():
        pytest.skip("git 저장소가 아닙니다")
    pkgs = [p.parent for p in ROOT.glob("src/**/__init__.py")]
    ignored = []
    for pkg in pkgs:
        r = subprocess.run(["git", "check-ignore", "-q", str(pkg / "__init__.py")],
                           cwd=ROOT, capture_output=True)
        if r.returncode == 0:
            ignored.append(pkg.relative_to(ROOT).as_posix())
    assert not ignored, (
        f"소스 패키지가 .gitignore 에 걸립니다: {ignored}. "
        "규칙 앞에 / 를 붙여 루트에 고정하세요.")


def test_every_source_package_is_tracked_by_git():
    """__init__.py 가 있는데 git 이 모르는 폴더는 배포물에서 사라진다."""
    if not (ROOT / ".git").exists():
        pytest.skip("git 저장소가 아닙니다")
    tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.split())
    missing = [p.relative_to(ROOT).as_posix()
               for p in ROOT.glob("src/**/*.py")
               if p.relative_to(ROOT).as_posix() not in tracked]
    assert not missing, f"git 이 추적하지 않는 소스 파일: {missing}"


def test_config_paths_are_relative_in_the_committed_default():
    """
    절대경로를 config.yaml 에 박으면 받는 쪽에서 바로 깨진다.
    각자의 경로는 config/local.yaml 이나 REGIME_QUANT_DATA 로 덮어쓴다.
    """
    import yaml
    raw = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    for key, value in raw.get("paths", {}).items():
        p = Path(str(value))
        assert not p.is_absolute(), (
            f"paths.{key} 가 절대경로입니다: {value}. "
            "config/local.yaml 에 두세요 (git 제외 대상).")


def test_local_config_is_not_committed():
    if not (ROOT / ".git").exists():
        pytest.skip("git 저장소가 아닙니다")
    tracked = subprocess.run(["git", "ls-files", "config/local.yaml"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    assert not tracked, "config/local.yaml 은 각자의 컴퓨터 설정이라 커밋하면 안 됩니다"


# ─── 진입점이 살아 있는가 ─────────────────────────────────────────

def test_launcher_and_menu_are_importable():
    """아이콘 → start.bat → src.launcher → src.menu 경로가 끊기지 않아야 한다."""
    import importlib
    for mod in ("src.launcher", "src.menu"):
        importlib.import_module(mod)


def test_launcher_requires_a_modern_python():
    from src.launcher import MIN_PYTHON
    assert MIN_PYTHON >= (3, 11), "pandas 3.x 가 3.11 이상을 요구한다"


def test_icon_exists_for_the_shortcut():
    ico = ROOT / "assets" / "regime-quant.ico"
    assert ico.exists() and ico.stat().st_size > 1000
