# BIOTrace Java Forensic Client Launcher Script for Windows PowerShell
[CmdletBinding()]
param(
    [switch]$SkipCompile
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $scriptDir) { $scriptDir = Get-Location }
$rootDir = $scriptDir
$clientDir = Join-Path $rootDir "java-client"
$pomFile = Join-Path $clientDir "pom.xml"

# Dynamic Java Resolution
$javaExe = $null
if ($env:JAVA_HOME -and (Test-Path (Join-Path $env:JAVA_HOME "bin\java.exe"))) {
    $javaExe = Join-Path $env:JAVA_HOME "bin\java.exe"
} elseif (Test-Path "C:\Program Files\Java\jdk-26.0.2\bin\java.exe") {
    $javaExe = "C:\Program Files\Java\jdk-26.0.2\bin\java.exe"
} else {
    $cmd = Get-Command java -ErrorAction SilentlyContinue
    if ($cmd) {
        $javaExe = $cmd.Source
    } else {
        $jdkFallback = Get-ChildItem "C:\Program Files\Java" -Filter "jdk-*" -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
        if ($jdkFallback -and (Test-Path (Join-Path $jdkFallback.FullName "bin\java.exe"))) {
            $javaExe = Join-Path $jdkFallback.FullName "bin\java.exe"
        }
    }
}

if (-not $javaExe) {
    Write-Host "Error: Java runtime not found. Please install JDK 21+ or set JAVA_HOME." -ForegroundColor Red
    exit 1
}

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  BIOTrace Forensic Desktop Client Launcher          " -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  Java      : $javaExe" -ForegroundColor Green
Write-Host "  Directory : $clientDir" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Cyan

# Check if global mvn is available
$mvnCmd = Get-Command mvn -ErrorAction SilentlyContinue
$runMvnScript = Join-Path $rootDir "scripts\run_mvn.ps1"

Set-Location $clientDir

$phases = @()
if (-not $SkipCompile) {
    $phases += "compile"
}
$phases += "exec:java"

if ($mvnCmd) {
    Write-Host "Executing via system Maven..." -ForegroundColor Yellow
    & mvn @phases
} elseif (Test-Path $runMvnScript) {
    Write-Host "Executing via bundled Maven wrapper..." -ForegroundColor Yellow
    & powershell -ExecutionPolicy Bypass -File $runMvnScript -f $pomFile @phases
} else {
    Write-Host "Error: Neither Maven nor scripts/run_mvn.ps1 was found." -ForegroundColor Red
    exit 1
}
