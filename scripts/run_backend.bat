@echo off
setlocal enabledelayedexpansion

echo =====================================================
echo   BIOTrace AI Engine - Launcher
echo =====================================================

set SCRIPT_DIR=%~dp0
set ENGINE_DIR=%SCRIPT_DIR%ai-engine
cd /d "%ENGINE_DIR%"

set PYTHON_CMD=

if exist "%ENGINE_DIR%\.venv\Scripts\python.exe" (
    set "PYTHON_CMD=%ENGINE_DIR%\.venv\Scripts\python.exe"
) else if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
) else if exist "%ProgramFiles%\Python311\python.exe" (
    set "PYTHON_CMD=%ProgramFiles%\Python311\python.exe"
) else (
    set "PYTHON_CMD=python"
)

echo Using Python: !PYTHON_CMD!
set "PYTHONPATH=%ENGINE_DIR%\src;%PYTHONPATH%"
if not defined MPLCONFIGDIR set "MPLCONFIGDIR=%TEMP%\biotrace_mpl"

"!PYTHON_CMD!" -m uvicorn biotrace.main:app --app-dir src --host 127.0.0.1 --port 8000
pause
