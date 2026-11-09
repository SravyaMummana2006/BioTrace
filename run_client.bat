@echo off
setlocal enabledelayedexpansion

echo =====================================================
echo   BIOTrace Forensic Desktop Client Launcher
echo =====================================================

set SCRIPT_DIR=%~dp0
if exist "%SCRIPT_DIR%java-client" (
    set "ROOT_DIR=%SCRIPT_DIR%"
) else (
    set "ROOT_DIR=%SCRIPT_DIR%..\"
)
set "CLIENT_DIR=%ROOT_DIR%java-client"
cd /d "%CLIENT_DIR%"

if exist "%ROOT_DIR%run_client.ps1" (
    powershell -ExecutionPolicy Bypass -File "%ROOT_DIR%run_client.ps1"
) else (
    powershell -ExecutionPolicy Bypass -File "%ROOT_DIR%scripts\run_client.ps1"
)
pause
