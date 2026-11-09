# BIOTrace AI Engine Launcher Script for Windows PowerShell
[CmdletBinding()]
param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $scriptDir) { $scriptDir = Get-Location }
$rootDir = $scriptDir
$engineDir = Join-Path $rootDir "ai-engine"
$srcDir = Join-Path $engineDir "src"

# Locate suitable Python executable
$candidatePythons = @(
    (Join-Path $engineDir ".venv\Scripts\python.exe"),
    (Join-Path $rootDir ".venv\Scripts\python.exe"),
    "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python311\python.exe",
    "C:\Program Files\Python311\python.exe",
    "C:\Program Files\Python312\python.exe"
)

$pythonExe = $null

foreach ($cand in $candidatePythons) {
    if (Test-Path $cand) {
        # Check if uvicorn is available in this python environment
        $check = & $cand -c "import uvicorn; print('OK')" 2>$null
        if ($check -match "OK") {
            $pythonExe = $cand
            break
        }
    }
}

if (-not $pythonExe) {
    # Check default python in PATH
    $pyCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pyCmd) {
        $check = & $pyCmd.Source -c "import uvicorn; print('OK')" 2>$null
        if ($check -match "OK") {
            $pythonExe = $pyCmd.Source
        }
    }
}

if (-not $pythonExe) {
    # Try py -3.11 or py launcher
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $check = & py -3.11 -c "import uvicorn; print('OK')" 2>$null
        if ($check -match "OK") {
            $pythonExe = "py -3.11"
        }
    }
}

if (-not $pythonExe) {
    Write-Host "Error: Could not find a Python interpreter with 'uvicorn' installed." -ForegroundColor Red
    Write-Host "Please install dependencies with: pip install -r ai-engine/requirements.txt" -ForegroundColor Yellow
    exit 1
}

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  BIOTrace AI Engine - Backend Server Launcher       " -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  Python    : $pythonExe" -ForegroundColor Green
Write-Host "  Directory : $engineDir" -ForegroundColor Green
Write-Host "  Source    : $srcDir" -ForegroundColor Green
Write-Host "  Endpoint  : http://$HostAddress`:$Port" -ForegroundColor Green
Write-Host "  Swagger UI: http://$HostAddress`:$Port/docs" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Cyan

Set-Location $engineDir
$env:PYTHONPATH = $srcDir
if (-not $env:MPLCONFIGDIR) {
    $env:MPLCONFIGDIR = Join-Path ([System.IO.Path]::GetTempPath()) "biotrace_mpl"
    if (-not (Test-Path $env:MPLCONFIGDIR)) {
        New-Item -ItemType Directory -Path $env:MPLCONFIGDIR -Force | Out-Null
    }
}

$uvicornArgs = @("-m", "uvicorn", "biotrace.main:app", "--app-dir", "src", "--host", $HostAddress, "--port", "$Port")
if ($Reload) {
    $uvicornArgs += "--reload"
}

if ($pythonExe -eq "py -3.11") {
    & py -3.11 @uvicornArgs
} else {
    & $pythonExe @uvicornArgs
}
