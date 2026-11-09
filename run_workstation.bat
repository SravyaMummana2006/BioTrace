@echo off
setlocal
echo ==========================================================
echo   BIOTrace Forensic Platform - Unified Workstation Launcher
echo ==========================================================

set SCRIPT_DIR=%~dp0
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run_workstation.ps1"
pause
