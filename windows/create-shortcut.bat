@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 바탕화면 바로가기를 만듭니다...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0create-shortcut.ps1"
if errorlevel 1 ( echo [오류] 바로가기 생성 실패 & pause & exit /b 1 )
echo 완료되었습니다.
pause
