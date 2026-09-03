@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0.."

echo ============================================================
echo   Regime Quant - 설치
echo ============================================================
echo.

REM ── 파이썬 확인. py 런처를 먼저 보고, 없으면 python 을 본다.
set "PYEXE="
for %%P in ("py -3" "python") do (
    %%~P --version >nul 2>&1 && ( set "PYEXE=%%~P" & goto :found )
)
:found
if "%PYEXE%"=="" (
    echo [오류] 파이썬을 찾을 수 없습니다.
    echo.
    echo   https://www.python.org/downloads/ 에서 Python 3.11 이상을 설치하고,
    echo   설치 화면의 "Add Python to PATH" 를 반드시 체크하세요.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('%PYEXE% --version 2^>^&1') do set "PYVER=%%v"
echo 파이썬 %PYVER% 를 찾았습니다.
echo.

REM ── 3.11 미만이면 멈춘다. pandas 3.x 가 요구한다.
%PYEXE% -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)"
if errorlevel 1 (
    echo [오류] Python 3.11 이상이 필요합니다 ^(현재 %PYVER%^).
    pause
    exit /b 1
)

echo [1/3] 가상환경을 만듭니다 (.venv)
if not exist ".venv" ( %PYEXE% -m venv .venv ) else ( echo       이미 있습니다 - 건너뜁니다 )
if errorlevel 1 ( echo [오류] 가상환경 생성 실패 & pause & exit /b 1 )

echo [2/3] 패키지를 설치합니다 (몇 분 걸립니다)
call ".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
call ".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 ( echo [오류] 패키지 설치 실패 & pause & exit /b 1 )

echo [3/3] 설치를 검증합니다
call ".venv\Scripts\python.exe" -m pytest -q --timeout=600
if errorlevel 1 (
    echo.
    echo [경고] 일부 테스트가 실패했습니다. 그래도 실행은 가능하지만
    echo        결과를 신뢰하기 전에 원인을 확인하세요.
)

echo.
echo ============================================================
echo   설치 완료
echo ============================================================
echo.
echo   바탕화면 아이콘을 만들려면:  windows\create-shortcut.bat
echo   바로 실행하려면:             windows\start.bat
echo.
pause
