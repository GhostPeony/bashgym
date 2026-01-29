# Find process on port 8003
$connections = Get-NetTCPConnection -LocalPort 8003 -State Listen -ErrorAction SilentlyContinue
foreach ($conn in $connections) {
    $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "Port 8003 owned by PID $($conn.OwningProcess): $($proc.ProcessName)"
        $wmi = Get-WmiObject Win32_Process -Filter "ProcessId=$($conn.OwningProcess)"
        Write-Host "Command: $($wmi.CommandLine)"
        Write-Host "Killing..."
        Stop-Process -Id $conn.OwningProcess -Force
    }
}
