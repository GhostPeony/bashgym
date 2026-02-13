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
    if (-not $FrontendOnly) {
        Write-Host "  [API]      http://localhost:$Port/api" -ForegroundColor Cyan
        Write-Host "  [Docs]     http://localhost:$Port/docs" -ForegroundColor DarkCyan
        Write-Host "  [WS]       ws://localhost:$Port/ws" -ForegroundColor DarkCyan

        $backendJob = Start-Job -Name "backend" -ScriptBlock {
            param($root, $port)
            Set-Location $root
            & python run_backend.py --port $port 2>&1
        } -ArgumentList $ProjectRoot, $Port

        $jobs += $backendJob
    }

    # --- Frontend ---
    if (-not $BackendOnly) {
        $frontendDir = Join-Path $ProjectRoot "frontend"

        if ($Electron) {
            Write-Host "  [Electron] starting..." -ForegroundColor Magenta
            $frontendJob = Start-Job -Name "frontend" -ScriptBlock {
                param($dir)
                Set-Location $dir
                & npm run electron:dev 2>&1
            } -ArgumentList $frontendDir
        } else {
            Write-Host "  [Frontend] http://localhost:5173" -ForegroundColor Green
            $frontendJob = Start-Job -Name "frontend" -ScriptBlock {
                param($dir)
                Set-Location $dir
                & npm run dev 2>&1
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
    $portPid = (Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue).OwningProcess | Select-Object -Unique
    if ($portPid) {
        foreach ($pid in $portPid) {
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        }
    }

    Write-Host "  Done." -ForegroundColor DarkGray
}
