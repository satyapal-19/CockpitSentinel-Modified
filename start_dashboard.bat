@echo off
title CockpitSentinel Dashboard
cd /d "%~dp0"
echo ===================================================
echo 🚀 Launching CockpitSentinel Telematics Dashboard...
echo ===================================================
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run_dashboard.py %*
) else (
    python run_dashboard.py %*
)
pause
