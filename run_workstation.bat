@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_workstation.ps1"
exit /b %errorlevel%
