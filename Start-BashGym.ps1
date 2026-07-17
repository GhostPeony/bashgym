<#
.SYNOPSIS
    Start, focus, restart, stop, or inspect the local BashGym desktop stack.

.DESCRIPTION
    This is the one-click Windows entry point for BashGym. It owns the Electron
    development stack, selects a usable Python interpreter, records startup logs,
    and cleans up stale BashGym processes without touching unrelated Node/Python apps.

.PARAMETER Restart
    Stop the current BashGym desktop stack before starting a fresh one.

.PARAMETER Stop
    Stop the current BashGym desktop stack and exit.

.PARAMETER Status
    Print renderer, Electron, and backend status and exit.

.PARAMETER InstallShortcut
    Create or refresh the BashGym shortcut on the current user's desktop.

.PARAMETER Foreground
    Run the underlying development host in the current console.
#>

[CmdletBinding()]
param(
    [switch]$Restart,
    [switch]$Stop,
    [switch]$Status,
    [switch]$InstallShortcut,
    [switch]$Foreground,
    [ValidateRange(10, 300)]
    [int]$StartupTimeoutSeconds = 150
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$FrontendRoot = Join-Path $ProjectRoot 'frontend'
$DevScript = Join-Path $ProjectRoot 'dev.ps1'
$ElectronExe = Join-Path $FrontendRoot 'node_modules\electron\dist\electron.exe'
$StateRoot = Join-Path $env:LOCALAPPDATA 'BashGym'
$StateFile = Join-Path $StateRoot 'desktop-stack.json'
$StdoutLog = Join-Path $StateRoot 'desktop-startup.stdout.log'
$StderrLog = Join-Path $StateRoot 'desktop-startup.stderr.log'
$LauncherLog = Join-Path $StateRoot 'desktop-launcher.log'

function Write-LauncherEvent {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('INFO', 'ERROR')][string]$Level,
        [Parameter(Mandatory = $true)][string]$Message
    )
    New-Item -ItemType Directory -Path $StateRoot -Force | Out-Null
    $line = '{0} [{1}] {2}' -f [DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss.fff'), $Level, $Message
    Add-Content -LiteralPath $LauncherLog -Value $line
}

function Test-HttpEndpoint {
    param([Parameter(Mandatory = $true)][string]$Uri)
    try {
        $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 500
    }
    catch {
        return $false
    }
}

function Get-MainElectronProcess {
    if (-not (Test-Path -LiteralPath $ElectronExe)) { return $null }
    $escapedElectron = [regex]::Escape($ElectronExe)
    return Get-CimInstance Win32_Process -Filter "Name = 'electron.exe'" -ErrorAction SilentlyContinue |
        Where-Object {
            $_.ExecutablePath -eq $ElectronExe -and
            $_.CommandLine -match "^`"?$escapedElectron`"?\s+\.(?:\s|$)"
        } |
        Select-Object -First 1
}

function Get-BashGymProcessIds {
    $processes = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    $byId = @{}
    foreach ($process in $processes) { $byId[[int]$process.ProcessId] = $process }

    $owned = [System.Collections.Generic.HashSet[int]]::new()
    $escapedProject = [regex]::Escape($ProjectRoot)
    $escapedFrontend = [regex]::Escape($FrontendRoot)

    foreach ($process in $processes) {
        $commandLine = [string]$process.CommandLine
        $isElectron = $process.Name -eq 'electron.exe' -and $process.ExecutablePath -eq $ElectronExe
        $isDevHost = $commandLine -match "$escapedProject[\\/]dev\.ps1.*-Electron"
        $isFrontendChild = $process.Name -in @('node.exe', 'cmd.exe') -and
            $commandLine -match $escapedFrontend -and
            $commandLine -match '(?i)(electron|vite|concurrently|esbuild)'
        $isManagedBackend = $process.Name -in @('python.exe', 'pythonw.exe') -and
            $commandLine -match 'bashgym\.api\.routes:create_app' -and
            $commandLine -match '(?:--port\s+|--port=)8003(?:\s|$)'

        if ($isElectron -or $isDevHost -or $isFrontendChild -or $isManagedBackend) {
            [void]$owned.Add([int]$process.ProcessId)
        }
    }

    # Pull in children of known BashGym processes, including renderer helpers.
    do {
        $changed = $false
        foreach ($process in $processes) {
            if ($owned.Contains([int]$process.ParentProcessId) -and
                $owned.Add([int]$process.ProcessId)) {
                $changed = $true
            }
        }
    } while ($changed)

    # Pull in only the recognizable npm/cmd ancestors of an owned frontend child.
    $ownedSnapshot = @($owned)
    foreach ($processId in $ownedSnapshot) {
        $current = $byId[$processId]
        while ($current -and $byId.ContainsKey([int]$current.ParentProcessId)) {
            $parent = $byId[[int]$current.ParentProcessId]
            $parentCommand = [string]$parent.CommandLine
            if ($parentCommand -match '(?i)(npm-cli\.js.*electron:dev|npm\.cmd.*electron:dev|concurrently.*vite)') {
                [void]$owned.Add([int]$parent.ProcessId)
                $current = $parent
            }
            else {
                break
            }
        }
    }

    return @($owned | Where-Object { $_ -ne $PID })
}

function Stop-BashGymStack {
    $processIds = @(Get-BashGymProcessIds)
    if ($processIds.Count -eq 0) { return }

    $processes = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    $depth = @{}
    foreach ($processId in $processIds) { $depth[$processId] = 0 }
    for ($iteration = 0; $iteration -lt $processIds.Count; $iteration++) {
        foreach ($process in $processes) {
            $processId = [int]$process.ProcessId
            $parentId = [int]$process.ParentProcessId
            if ($depth.ContainsKey($processId) -and $depth.ContainsKey($parentId)) {
                $depth[$processId] = [Math]::Max($depth[$processId], $depth[$parentId] + 1)
            }
        }
    }

    foreach ($processId in ($processIds | Sort-Object { $depth[$_] } -Descending)) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }

    # Electron/ConPTY helpers can linger briefly after their parent is killed.
    # Re-enumerate and re-issue termination while they drain so a transient
    # helper cannot block the next desktop launch.
    $deadline = [DateTime]::UtcNow.AddSeconds(20)
    do {
        Start-Sleep -Milliseconds 150
        $remaining = @(Get-BashGymProcessIds)
        foreach ($remainingId in $remaining) {
            Stop-Process -Id $remainingId -Force -ErrorAction SilentlyContinue
        }
    } while ($remaining.Count -gt 0 -and [DateTime]::UtcNow -lt $deadline)

    if ($remaining.Count -gt 0) {
        throw "BashGym processes did not stop: $($remaining -join ', ')"
    }
    Remove-Item -LiteralPath $StateFile -Force -ErrorAction SilentlyContinue
}

function Get-UsablePython {
    $candidates = @()
    if ($env:BASHGYM_PYTHON) { $candidates += $env:BASHGYM_PYTHON }
    $candidates += (Join-Path $ProjectRoot '.venv\Scripts\python.exe')
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) { $candidates += $pythonCommand.Source }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }

        # Do not invoke the candidate directly under ErrorActionPreference=Stop.
        # Windows PowerShell promotes native stderr (for example, a missing
        # module traceback) to a terminating NativeCommandError before we can
        # inspect LASTEXITCODE and continue to the next interpreter.
        $startInfo = New-Object System.Diagnostics.ProcessStartInfo
        $startInfo.FileName = $candidate
        $startInfo.Arguments = '-c "import httpx, uvicorn"'
        $startInfo.UseShellExecute = $false
        $startInfo.CreateNoWindow = $true
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true

        $probe = New-Object System.Diagnostics.Process
        $probe.StartInfo = $startInfo
        try {
            if (-not $probe.Start()) { continue }
            $stdout = $probe.StandardOutput.ReadToEnd()
            $stderr = $probe.StandardError.ReadToEnd()
            if (-not $probe.WaitForExit(10000)) {
                try { $probe.Kill() } catch { }
                Write-LauncherEvent -Level 'ERROR' -Message "Python probe timed out: $candidate"
                continue
            }
            if ($probe.ExitCode -eq 0) { return $candidate }
            $reason = if (-not [string]::IsNullOrWhiteSpace($stderr)) {
                $stderr.Trim()
            }
            elseif (-not [string]::IsNullOrWhiteSpace($stdout)) {
                $stdout.Trim()
            }
            else {
                'no diagnostic output'
            }
            Write-LauncherEvent -Level 'INFO' -Message "Skipping Python candidate '$candidate' (exit $($probe.ExitCode)): $reason"
        }
        catch {
            Write-LauncherEvent -Level 'INFO' -Message "Skipping Python candidate '$candidate': $($_.Exception.Message)"
        }
        finally {
            $probe.Dispose()
        }
    }
    throw 'No usable Python installation found. BashGym needs Python with httpx and uvicorn installed.'
}

function Install-BashGymShortcut {
    $desktop = [Environment]::GetFolderPath('Desktop')
    if (-not $desktop) { throw 'Windows did not return a Desktop folder.' }
    $shortcutPath = Join-Path $desktop 'BashGym.lnk'
    $powershell = (Get-Command powershell.exe -ErrorAction Stop).Source
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $powershell
    $shortcut.Arguments = "-NoLogo -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$PSCommandPath`""
    $shortcut.WorkingDirectory = $ProjectRoot
    $shortcut.Description = 'Start or focus the BashGym desktop environment'
    if (Test-Path -LiteralPath (Join-Path $FrontendRoot 'build\icon.ico')) {
        $shortcut.IconLocation = (Join-Path $FrontendRoot 'build\icon.ico')
    }
    elseif (Test-Path -LiteralPath $ElectronExe) {
        $shortcut.IconLocation = "$ElectronExe,0"
    }
    $shortcut.Save()
    return $shortcutPath
}

function Write-StackStatus {
    $electron = Get-MainElectronProcess
    [pscustomobject]@{
        Electron = if ($electron) { "running (PID $($electron.ProcessId))" } else { 'stopped' }
        Renderer = if (Test-HttpEndpoint 'http://127.0.0.1:5173') { 'ready' } else { 'stopped' }
        Backend = if (Test-HttpEndpoint 'http://127.0.0.1:8003/api/health') { 'ready' } else { 'stopped' }
        Logs = $StateRoot
    }
}

function Show-LaunchError {
    param([Parameter(Mandatory = $true)][string]$Message)
    try {
        Add-Type -AssemblyName System.Windows.Forms
        [void][System.Windows.Forms.MessageBox]::Show(
            "$Message`r`n`r`nLogs: $StateRoot",
            'BashGym could not start',
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        )
    }
    catch {
        # The caller still receives the terminating error when UI is unavailable.
    }
}

try {
    if ($InstallShortcut) {
        $shortcutPath = Install-BashGymShortcut
        Write-Output "Desktop shortcut ready: $shortcutPath"
        if (-not ($Restart -or $Stop -or $Status)) { return }
    }

    if ($Status) {
        Write-StackStatus
        return
    }

    if ($Stop) {
        Stop-BashGymStack
        Write-Output 'BashGym stopped.'
        return
    }

    $existingElectron = Get-MainElectronProcess
    if ($existingElectron -and -not $Restart) {
        try {
            Add-Type -AssemblyName Microsoft.VisualBasic
            [void][Microsoft.VisualBasic.Interaction]::AppActivate([int]$existingElectron.ProcessId)
        }
        catch { }
        Write-Output "BashGym is already running (PID $($existingElectron.ProcessId))."
        return
    }

    if ($Restart -or (Get-BashGymProcessIds).Count -gt 0) {
        Stop-BashGymStack
    }

    if (-not (Test-Path -LiteralPath $DevScript)) { throw "Missing launcher: $DevScript" }
    if (-not (Test-Path -LiteralPath $ElectronExe)) {
        throw "Frontend dependencies are missing. Run 'npm install' in $FrontendRoot once."
    }

    $env:BASHGYM_PROJECT_ROOT = $ProjectRoot
    $env:BASHGYM_PYTHON = Get-UsablePython
    Write-LauncherEvent -Level 'INFO' -Message "Selected Python: $($env:BASHGYM_PYTHON)"
    $env:BASHGYM_API_BASE = 'http://127.0.0.1:8003/api'
    $env:BASHGYM_DEV_SERVER_URL = 'http://127.0.0.1:5173'
    $env:PYTHONUTF8 = '1'

    New-Item -ItemType Directory -Path $StateRoot -Force | Out-Null
    Set-Content -LiteralPath $StdoutLog -Value ''
    Set-Content -LiteralPath $StderrLog -Value ''

    $powershell = (Get-Command powershell.exe -ErrorAction Stop).Source
    $arguments = @(
        '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', "`"$DevScript`"", '-Electron'
    )

    if ($Foreground) {
        & $powershell @arguments
        exit $LASTEXITCODE
    }

    $hostProcess = Start-Process -FilePath $powershell -ArgumentList $arguments `
        -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog

    [pscustomobject]@{
        processId = $hostProcess.Id
        startedAt = [DateTime]::UtcNow.ToString('o')
        projectRoot = $ProjectRoot
        python = $env:BASHGYM_PYTHON
    } | ConvertTo-Json | Set-Content -LiteralPath $StateFile

    $deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
    do {
        Start-Sleep -Milliseconds 250
        $hostProcess.Refresh()
        if ($hostProcess.HasExited) {
            $stderrTail = if (Test-Path $StderrLog) { (Get-Content $StderrLog -Tail 20) -join "`n" } else { '' }
            throw "The BashGym desktop host exited during startup. $stderrTail"
        }
        $electronReady = $null -ne (Get-MainElectronProcess)
        $rendererReady = Test-HttpEndpoint 'http://127.0.0.1:5173'
        $backendReady = Test-HttpEndpoint 'http://127.0.0.1:8003/api/health'
    } while ((-not ($electronReady -and $rendererReady -and $backendReady)) -and
        [DateTime]::UtcNow -lt $deadline)

    if (-not ($electronReady -and $rendererReady -and $backendReady)) {
        throw "BashGym startup timed out after $StartupTimeoutSeconds seconds."
    }

    Write-LauncherEvent -Level 'INFO' -Message 'BashGym desktop stack is ready.'
    Write-StackStatus
}
catch {
    $errorMessage = $_.Exception.Message
    try { Write-LauncherEvent -Level 'ERROR' -Message $errorMessage } catch { }
    Show-LaunchError -Message $errorMessage
    throw
}
