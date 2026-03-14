param(
    [Parameter(Mandatory = $true)]
    [string]$StartTimestamp,

    [Parameter(Mandatory = $true)]
    [string]$EndTimestamp,

    [Parameter(Mandatory = $true)]
    [string]$AuthKey,

    [string]$OutputDir = "data\\aws_korea_q1_2025",
    [string]$PythonExe = "python",
    [string]$StationId = "0",
    [int]$IntervalMinutes = 60,
    [int]$ChunkSize = 24,
    [int]$RetryWaitSeconds = 900,
    [int]$MaxAttempts = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-LastAwsTimestamp {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RawCsvPath
    )

    if (-not (Test-Path $RawCsvPath)) {
        return $null
    }

    $lastLine = Get-Content $RawCsvPath -Tail 20 | Where-Object { $_.Trim() } | Select-Object -Last 1
    if (-not $lastLine) {
        return $null
    }

    $parts = $lastLine.Split(",")
    if ($parts.Count -eq 0 -or $parts[0] -eq "tm") {
        return $null
    }

    return [datetime]::ParseExact($parts[0], "yyyyMMddHHmm", $null)
}

function Write-Status {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line
    Add-Content -Path $script:LogPath -Value $line -Encoding UTF8
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ResolvedOutputDir = Join-Path $RepoRoot $OutputDir
$RawCsvPath = Join-Path $ResolvedOutputDir "aws_raw.csv"
$DownloaderPath = Join-Path $PSScriptRoot "download_aws_apihub_range.py"
$script:LogPath = Join-Path $ResolvedOutputDir "aws_download_retry.log"

if (-not (Test-Path $ResolvedOutputDir)) {
    New-Item -ItemType Directory -Path $ResolvedOutputDir | Out-Null
}

$targetEnd = [datetime]::ParseExact($EndTimestamp, "yyyy-MM-ddTHH:mm", $null)
$attempt = 0

while ($true) {
    $currentEnd = Get-LastAwsTimestamp -RawCsvPath $RawCsvPath
    if ($null -ne $currentEnd -and $currentEnd -ge $targetEnd) {
        Write-Status "Download already complete through $($currentEnd.ToString('yyyy-MM-ddTHH:mm'))."
        break
    }

    if ($MaxAttempts -gt 0 -and $attempt -ge $MaxAttempts) {
        Write-Status "Reached MaxAttempts=$MaxAttempts before completion."
        exit 1
    }

    $attempt += 1
    $currentLabel = if ($null -eq $currentEnd) { "none" } else { $currentEnd.ToString("yyyy-MM-ddTHH:mm") }
    Write-Status "Attempt $attempt starting from last timestamp $currentLabel."

    $arguments = @(
        $DownloaderPath,
        "--start", $StartTimestamp,
        "--end", $EndTimestamp,
        "--interval-minutes", "$IntervalMinutes",
        "--chunk-size", "$ChunkSize",
        "--auth-key", $AuthKey,
        "--output-dir", $ResolvedOutputDir,
        "--station-id", $StationId,
        "--resume"
    )

    & $PythonExe @arguments
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        $completedEnd = Get-LastAwsTimestamp -RawCsvPath $RawCsvPath
        if ($null -ne $completedEnd -and $completedEnd -ge $targetEnd) {
            Write-Status "Download completed through $($completedEnd.ToString('yyyy-MM-ddTHH:mm'))."
            break
        }

        Write-Status "Downloader exited cleanly but target end was not reached yet. Sleeping $RetryWaitSeconds seconds."
        Start-Sleep -Seconds $RetryWaitSeconds
        continue
    }

    Write-Status "Downloader exited with code $exitCode. Sleeping $RetryWaitSeconds seconds before retry."
    Start-Sleep -Seconds $RetryWaitSeconds
}
