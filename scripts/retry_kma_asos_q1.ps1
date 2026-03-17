param(
    [Parameter(Mandatory = $true)]
    [string]$StartDate,

    [Parameter(Mandatory = $true)]
    [string]$EndDate,

    [Parameter(Mandatory = $true)]
    [string]$AuthKey,

    [string]$OutputDir = "data\asos_korea_q1_2025",

    [string[]]$StationIds = @("0"),

    [int]$RetryWaitSeconds = 900,

    [string]$PythonExe = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-PythonExe {
    param([string]$Requested)
    if ($Requested) {
        return $Requested
    }
    $venvPython = Join-Path $PSScriptRoot "..\.venv311\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return (Resolve-Path $venvPython).Path
    }
    return "python"
}

function Get-QuotaResetSleepSeconds {
    $now = Get-Date
    $next = $now.Date.AddDays(1).AddMinutes(5)
    return [Math]::Max(60, [int][Math]::Ceiling(($next - $now).TotalSeconds))
}

$python = Resolve-PythonExe -Requested $PythonExe
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedOutputDir = Join-Path $repoRoot $OutputDir
New-Item -ItemType Directory -Force -Path $resolvedOutputDir | Out-Null
$logPath = Join-Path $resolvedOutputDir "asos_download_retry.log"

while ($true) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] Attempting ASOS API Hub Q1 download."

    $commandArgs = @(
        "-m", "task_cli", "kma-download",
        "--source", "asos_hourly_apihub",
        "--output-dir", $resolvedOutputDir,
        "--station-ids"
    ) + $StationIds + @(
        "--start-date", $StartDate,
        "--end-date", $EndDate,
        "--service-key", $AuthKey,
        "--standardized-csv", (Join-Path $resolvedOutputDir "asos_framework.csv"),
        "--framework-config", (Join-Path $resolvedOutputDir "framework_asos_q1.json"),
        "--overwrite"
    )

    & $python @commandArgs 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -eq 0) {
        $done = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Add-Content -Path $logPath -Value "[$done] ASOS download completed successfully."
        break
    }

    $sleepSeconds = [Math]::Max($RetryWaitSeconds, (Get-QuotaResetSleepSeconds))
    $retryAt = (Get-Date).AddSeconds($sleepSeconds).ToString("yyyy-MM-dd HH:mm:ss")
    Add-Content -Path $logPath -Value "[$timestamp] Download failed. Retrying at $retryAt."
    Start-Sleep -Seconds $sleepSeconds
}
