# BIOTrace Unified Workstation Launcher
# Launches the FastAPI AI Engine, waits for health check, and starts the Desktop Client
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $scriptDir) { $scriptDir = (Get-Location).Path }
if ((Split-Path -Leaf $scriptDir) -eq "scripts") {
    $rootDir = Split-Path -Parent $scriptDir
} else {
    $rootDir = $scriptDir
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   BIOTrace Forensic Platform - Unified One-Click Launcher" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check if AI Engine is already running
$healthUrl = "http://127.0.0.1:8000/api/v1/health"
$engineReady = $false

try {
    $resp = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($resp.status -eq "healthy") {
        $engineReady = $true
        Write-Host "  [OK] AI Engine is already running on http://127.0.0.1:8000" -ForegroundColor Green
    }
} catch {}

if (-not $engineReady) {
    Write-Host "  Starting BIOTrace AI Engine in background..." -ForegroundColor Yellow
    $backendScript = Join-Path $rootDir "run_backend.bat"
    Start-Process cmd.exe -ArgumentList "/c `"$backendScript`"" -WindowStyle Minimized

    Write-Host "  Waiting for AI models to initialize (PyTorch, MediaPipe, OpenCV)..." -ForegroundColor DarkGray
    $maxAttempts = 40
    for ($i = 1; $i -le $maxAttempts; $i++) {
        Start-Sleep -Seconds 2
        try {
            $resp = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($resp.status -eq "healthy") {
                $engineReady = $true
                Write-Host "  [SUCCESS] AI Engine is ONLINE (v$($resp.version))!" -ForegroundColor Green
                break
            }
        } catch {
            Write-Host "  Waiting for server... ($($i * 2)s / $($maxAttempts * 2)s)" -ForegroundColor DarkGray
        }
    }
}

if (-not $engineReady) {
    Write-Host "  [WARNING] Engine took longer than expected to report healthy." -ForegroundColor Yellow
    Write-Host "  Launching Java Forensic Client anyway (will reconnect automatically)..." -ForegroundColor Yellow
}

Write-Host "  Launching BIOTrace Java Forensic Desktop Workstation..." -ForegroundColor Cyan
$clientScript = Join-Path $rootDir "run_client.ps1"
& powershell -ExecutionPolicy Bypass -File $clientScript
