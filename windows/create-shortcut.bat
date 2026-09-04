@echo off
REM Creates a desktop shortcut. ASCII ONLY - see windows/start.bat.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0create-shortcut.ps1"
if errorlevel 1 echo Failed to create the shortcut.
pause
