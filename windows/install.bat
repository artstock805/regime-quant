@echo off
REM Optional explicit installer. Clicking the desktop icon also sets things up.
REM ASCII ONLY - see windows/start.bat for why.
setlocal
cd /d "%~dp0.."
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

py -3 --version >nul 2>&1 && (
    py -3 -m src.launcher --install
    goto :done
)
python --version >nul 2>&1 && (
    python -m src.launcher --install
    goto :done
)

echo.
echo   Python was not found. Install Python 3.11+ from python.org
echo   and TICK "Add Python to PATH", then run this again.
echo.
pause
exit /b 1

:done
pause
