@echo off
chcp 65001 >nul
cd /d "%~dp0.."
title Regime Quant

if not exist ".venv\Scripts\python.exe" (
    echo 아직 설치되지 않았습니다. windows\install.bat 을 먼저 실행하세요.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m src.menu %*
if errorlevel 1 pause
