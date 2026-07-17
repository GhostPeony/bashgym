<#
.SYNOPSIS
    Start the Bash Gym development environment (backend + frontend).

.DESCRIPTION
    Launches the FastAPI backend (port 8003, hot reload) and Vite frontend
    dev server (port 5173) in parallel. Press Ctrl+C to stop both.

.PARAMETER BackendOnly
    Only start the backend API server.

.PARAMETER FrontendOnly
    Only start the frontend dev server.

.PARAMETER Electron
    Start frontend as full Electron app instead of browser-only Vite server.

.PARAMETER Port
    Backend port (default: 8003, matching frontend/.env.local).

.EXAMPLE
    .\dev.ps1                  # Start both
    .\dev.ps1 -BackendOnly     # Backend only
    .\dev.ps1 -FrontendOnly    # Frontend only
    .\dev.ps1 -Electron        # Backend + Electron app
#>

param(
    [switch]$BackendOnly,
    [switch]$FrontendOnly,
    [switch]$Electron,
    [int]$Port = 8003
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

Write-Host ""
Write-Host "  BASH GYM DEV" -ForegroundColor White
Write-Host "  ============" -ForegroundColor DarkGray
Write-Host ""

$jobs = @()

try {
    # --- Backend ---
    # Electron main owns its authenticated FastAPI child so the per-launch
    # bootstrap secret never crosses through the renderer or a shell argument.
    $electronOwnsBackend = $Electron -and -not $BackendOnly
    if (-not $FrontendOnly -and -not $electronOwnsBackend) {
        Write-Host "  [API]      http://localhost:$Port/api" -ForegroundColor Cyan
        Write-Host "  [Docs]     http://localhost:$Port/docs" -ForegroundColor DarkCyan
        Write-Host "  [WS]       ws://localhost:$Port/ws" -ForegroundColor DarkCyan

            $backendJob = Start-Job -Name "backend" -ScriptBlock {
                param($root, $port)
                Set-Location $root
                $env:PYTHONUTF8 = "1"  # UTF-8 console for emoji/rich output (NeMo Data Designer)
                & python run_backend.py --port $port 2>&1
                if ($LASTEXITCODE -ne 0) { throw "Backend exited with code $LASTEXITCODE" }
            } -ArgumentList $ProjectRoot, $Port

        $jobs += $backendJob
    }

    # --- Frontend ---
    if (-not $BackendOnly) {
        $frontendDir = Join-Path $ProjectRoot "frontend"

        if ($Electron) {
            Write-Host "  [Electron] starting..." -ForegroundColor Magenta
            $frontendJob = Start-Job -Name "frontend" -ScriptBlock {
                param($dir, $port)
                Set-Location $dir
                $env:BASHGYM_API_BASE = "http://127.0.0.1:$port/api"
                & npm run electron:dev 2>&1
                if ($LASTEXITCODE -ne 0) { throw "Electron host exited with code $LASTEXITCODE" }
            } -ArgumentList $frontendDir, $Port
        } else {
            Write-Host "  [Frontend] http://localhost:5173" -ForegroundColor Green
            $frontendJob = Start-Job -Name "frontend" -ScriptBlock {
                param($dir)
                Set-Location $dir
                & npm run dev 2>&1
                if ($LASTEXITCODE -ne 0) { throw "Frontend exited with code $LASTEXITCODE" }
            } -ArgumentList $frontendDir
        }

        $jobs += $frontendJob
    }

    Write-Host ""
    Write-Host "  Press Ctrl+C to stop all services." -ForegroundColor DarkGray
    Write-Host ""

    # Stream output from both jobs
    while ($true) {
        foreach ($job in $jobs) {
            $output = Receive-Job -Job $job -ErrorAction SilentlyContinue
            if ($output) {
                $prefix = if ($job.Name -eq "backend") { "[API]" } else { "[FE] " }
                $color = if ($job.Name -eq "backend") { "Cyan" } else { "Green" }
                foreach ($line in $output) {
                    Write-Host "  $prefix $line" -ForegroundColor $color
                }
            }

            if ($job.State -eq "Failed") {
                Write-Host "  $($job.Name) process failed!" -ForegroundColor Red
                Receive-Job -Job $job -ErrorAction SilentlyContinue | ForEach-Object {
                    Write-Host "  ERROR: $_" -ForegroundColor Red
                }
            }
        }
        $activeJobs = @($jobs | Where-Object { $_.State -in @('NotStarted', 'Running', 'Blocked') })
        if ($jobs.Count -gt 0 -and $activeJobs.Count -eq 0) {
            $failedJobs = @($jobs | Where-Object { $_.State -eq 'Failed' })
            if ($failedJobs.Count -gt 0) {
                throw "One or more BashGym services failed."
            }
            break
        }
        Start-Sleep -Milliseconds 500
    }
}
finally {
    Write-Host ""
    Write-Host "  Shutting down..." -ForegroundColor Yellow

    foreach ($job in $jobs) {
        Stop-Job -Job $job -ErrorAction SilentlyContinue
        Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
    }

    # Kill any leftover uvicorn on our port
    $portProcessIds = (Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue).OwningProcess | Select-Object -Unique
    if ($portProcessIds) {
        foreach ($processId in $portProcessIds) {
            $portProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
            if ($portProcess.CommandLine -match '(?i)(run_backend\.py|uvicorn|bashgym\.api\.routes:create_app)') {
                Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
            }
        }
    }

    Write-Host "  Done." -ForegroundColor DarkGray
}
