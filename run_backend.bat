@echo off
setlocal enabledelayedexpansion

echo =====================================================
echo   BIOTrace AI Engine - Backend Launcher
echo =====================================================

set SCRIPT_DIR=%~dp0
if exist "%SCRIPT_DIR%ai-engine" (
    set "ROOT_DIR=%SCRIPT_DIR%"
) else (
    set "ROOT_DIR=%SCRIPT_DIR%..\"
)
set "ENGINE_DIR=%ROOT_DIR%ai-engine"
cd /d "%ENGINE_DIR%"

set PYTHON_CMD=

if exist "%ENGINE_DIR%\.venv\Scripts\python.exe" (
    set "PYTHON_CMD=%ENGINE_DIR%\.venv\Scripts\python.exe"
) else if exist "%ROOT_DIR%.venv\Scripts\python.exe" (
    set "PYTHON_CMD=%ROOT_DIR%.venv\Scripts\python.exe"
) else if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
) else if exist "%ProgramFiles%\Python311\python.exe" (
    set "PYTHON_CMD=%ProgramFiles%\Python311\python.exe"
) else (
    set "PYTHON_CMD=python"
)

echo Using Python: !PYTHON_CMD!
echo Working Dir : %ENGINE_DIR%
set "PYTHONPATH=%ENGINE_DIR%\src;%PYTHONPATH%"
if not defined MPLCONFIGDIR set "MPLCONFIGDIR=%TEMP%\biotrace_mpl"

if not exist "%ENGINE_DIR%\models\mediapipe\face_landmarker.task" (
    echo Downloading required MediaPipe face-landmarker model...
    "!PYTHON_CMD!" "%ROOT_DIR%scripts\download_models.py"
    if errorlevel 1 (
        echo Error: model download failed. Run: python scripts\download_models.py
        pause
        exit /b 1
    )
)

"!PYTHON_CMD!" -m uvicorn biotrace.main:create_app --factory --app-dir src --host 127.0.0.1 --port 8000
pause
