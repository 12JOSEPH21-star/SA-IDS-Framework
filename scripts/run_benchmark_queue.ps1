param(
    [Parameter(Mandatory = $true)]
    [string[]]$Configs,
    [string]$PythonPath = ".\.venv311\Scripts\python.exe",
    [string]$QueueName = "benchmark_queue_main",
    [int]$WaitForPid = 0,
    [switch]$SkipCompleted
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-QueueStatus {
    param(
        [string]$StatusPath,
        [hashtable]$Payload
    )
    $Payload["updated_at"] = (Get-Date).ToString("s")
    ($Payload | ConvertTo-Json -Depth 8) | Out-File -FilePath $StatusPath -Encoding utf8
}

$resolvedPython = (Resolve-Path $PythonPath).Path
$queueDir = Join-Path (Get-Location) ("outputs\" + $QueueName)
New-Item -ItemType Directory -Force -Path $queueDir | Out-Null
$statusPath = Join-Path $queueDir "queue_status.json"
$logPath = Join-Path $queueDir "queue.log"

$resolvedConfigs = @()
foreach ($config in $Configs) {
    $resolvedConfigs += (Resolve-Path $config).Path
}

$status = [ordered]@{
    queue_name = $QueueName
    python_path = $resolvedPython
    configs = $resolvedConfigs
    wait_for_pid = $WaitForPid
    status = "starting"
    current_config = ""
    completed = @()
    failed = @()
}
Write-QueueStatus -StatusPath $statusPath -Payload $status

if ($WaitForPid -gt 0) {
    $status.status = "waiting_for_pid"
    Write-QueueStatus -StatusPath $statusPath -Payload $status
    while (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 15
        Write-QueueStatus -StatusPath $statusPath -Payload $status
    }
}

$status.status = "running"
Write-QueueStatus -StatusPath $statusPath -Payload $status

foreach ($configPath in $resolvedConfigs) {
    $configPayload = Get-Content $configPath -Raw | ConvertFrom-Json
    $outputDir = [string]$configPayload.output_dir
    $summaryPath = Join-Path $outputDir "summary.json"
    $status.current_config = $configPath
    $status.current_output_dir = $outputDir
    Write-QueueStatus -StatusPath $statusPath -Payload $status

    if ($SkipCompleted -and (Test-Path $summaryPath)) {
        $status.completed += [ordered]@{
            config = $configPath
            output_dir = $outputDir
            skipped = $true
            exit_code = 0
        }
        Write-QueueStatus -StatusPath $statusPath -Payload $status
        continue
    }

    Add-Content -Path $logPath -Value ("[{0}] START {1}" -f (Get-Date).ToString("s"), $configPath)
    & $resolvedPython -m task_cli benchmark-run --config $configPath *>> $logPath
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0) {
        $status.completed += [ordered]@{
            config = $configPath
            output_dir = $outputDir
            skipped = $false
            exit_code = $exitCode
        }
    }
    else {
        $status.failed += [ordered]@{
            config = $configPath
            output_dir = $outputDir
            exit_code = $exitCode
        }
        $status.status = "failed"
        Write-QueueStatus -StatusPath $statusPath -Payload $status
        exit $exitCode
    }
    Add-Content -Path $logPath -Value ("[{0}] DONE {1} exit={2}" -f (Get-Date).ToString("s"), $configPath, $exitCode)
    Write-QueueStatus -StatusPath $statusPath -Payload $status
}

$status.status = "completed"
$status.current_config = ""
Write-QueueStatus -StatusPath $statusPath -Payload $status
