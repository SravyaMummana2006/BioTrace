# Start the local FastAPI backend and Java desktop client in one session.
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = @(
    (Join-Path $root ".venv\Scripts\python.exe")
    (Join-Path $root "ai-engine\.venv\Scripts\python.exe")
) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $python) {
    throw "BioTrace needs a .venv with the AI engine dependencies. See the Windows setup in README.md."
}

$maven = Get-Command mvn.cmd -CommandType Application -ErrorAction SilentlyContinue
if (-not $maven) {
    throw "BioTrace needs Maven on PATH and a JDK configured via JAVA_HOME or PATH."
}

# Use a separate loopback port so an existing backend can stay running.
$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
try {
    $listener.Start()
    $port = $listener.LocalEndpoint.Port
} finally {
    $listener.Stop()
}

$apiUrl = "http://127.0.0.1:$port"
$backend = Start-Process -FilePath $python -WorkingDirectory $root -NoNewWindow -PassThru -ArgumentList @(
    "-m", "uvicorn", "biotrace.main:create_app", "--factory", "--app-dir", "ai-engine/src",
    "--host", "127.0.0.1", "--port", "$port"
)
$previousApiUrl = $env:BIOTRACE_API_URL
try {
    Write-Host "Waiting for BioTrace backend at $apiUrl"
    $deadline = [System.Diagnostics.Stopwatch]::StartNew()
    $ready = $false
    while ($deadline.Elapsed.TotalSeconds -lt 90) {
        if ($backend.HasExited) {
            throw "BioTrace backend exited before it became ready (exit code $($backend.ExitCode))."
        }
        try {
            $health = Invoke-RestMethod -Uri "$apiUrl/api/v1/health" -TimeoutSec 2
            $ready = $health.status -eq "healthy"
        } catch {
            # Uvicorn may still be importing the application.
        }
        if ($ready) { break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) {
        throw "Timed out waiting for the BioTrace backend."
    }

    $env:BIOTRACE_API_URL = $apiUrl
    Push-Location (Join-Path $root "java-client")
    try {
        & $maven.Source -q compile exec:java
        $clientExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
} finally {
    $env:BIOTRACE_API_URL = $previousApiUrl
    if (-not $backend.HasExited) {
        # The Windows virtual environment executable can spawn a child Python process.
        & taskkill.exe /PID $backend.Id /T /F | Out-Null
    }
    $backend.Dispose()
}
exit $clientExitCode
