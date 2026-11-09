@echo off
setlocal enabledelayedexpansion

echo =====================================================
echo   BIOTrace Forensic Desktop Client Launcher
echo =====================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%java-client"

powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run_client.ps1"
pause
