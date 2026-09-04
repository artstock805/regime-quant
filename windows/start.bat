@echo off
REM Regime Quant launcher.
REM ASCII ONLY - cmd.exe mangles non-ASCII batch files on non-UTF8 locales,
REM and the mangled bytes get parsed as commands. All human-readable text
REM lives in src/launcher.py, where encoding can be controlled.
setlocal
cd /d "%~dp0.."
title Regime Quant
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM Prefer the project venv, then the py launcher, then plain python.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m src.launcher %*
    goto :done
)
py -3 --version >nul 2>&1 && (
    py -3 -m src.launcher %*
    goto :done
)
python --version >nul 2>&1 && (
    python -m src.launcher %*
    goto :done
)

echo.
echo   Python was not found on this computer.
echo.
echo   Install Python 3.11 or newer from https://www.python.org/downloads/
echo   and TICK "Add Python to PATH" during setup, then run this again.
echo.
pause
exit /b 1

:done
if errorlevel 1 pause
