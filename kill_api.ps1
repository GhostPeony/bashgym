# Find and kill uvicorn process
$procs = Get-WmiObject Win32_Process -Filter "name='python.exe'"
foreach ($p in $procs) {
    if ($p.CommandLine -like "*uvicorn*" -or $p.CommandLine -like "*8003*") {
        Write-Host "Killing process $($p.ProcessId): $($p.CommandLine)"
        Stop-Process -Id $p.ProcessId -Force
    }
}
