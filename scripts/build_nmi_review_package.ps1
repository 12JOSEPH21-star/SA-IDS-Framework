param(
    [string]$PackageName = "saids-nmi-paper-materials-slim-20260317.zip"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$distDir = Join-Path $root "dist"
$stageName = [System.IO.Path]::GetFileNameWithoutExtension($PackageName)
$stageDir = Join-Path $distDir $stageName
$zipPath = Join-Path $distDir $PackageName
$reviewSummaryPath = Join-Path $root "review\NMI_PACKAGE_SUMMARY_20260317.md"

function Copy-RelativePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $sourcePath = Join-Path $root $RelativePath
    if (-not (Test-Path $sourcePath)) {
        return
    }

    $destinationPath = Join-Path $stageDir $RelativePath
    $destinationParent = Split-Path -Parent $destinationPath
    if (-not (Test-Path $destinationParent)) {
        New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
    }

    Copy-Item -Path $sourcePath -Destination $destinationPath -Recurse -Force
}

function Copy-BenchmarkArtifacts {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BenchmarkName
    )

    $relativeBase = "outputs\$BenchmarkName"
    $artifactFiles = @(
        "summary.json",
        "paper_tables.json",
        "report.md",
        "predictive_mnar.csv",
        "fault_diagnosis.csv",
        "reliability_shift.csv",
        "coverage_over_time.csv",
        "ablation.csv",
        "policy_runtime.csv",
        "runtime.csv",
        "region_holdout.csv",
        "significance_summary.json"
    )

    foreach ($fileName in $artifactFiles) {
        Copy-RelativePath "$relativeBase\$fileName"
    }

    Copy-RelativePath "$relativeBase\figures"
}

if (Test-Path $stageDir) {
    Remove-Item $stageDir -Recurse -Force
}
if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

New-Item -ItemType Directory -Path $stageDir -Force | Out-Null

$topLevelFiles = @(
    "models.py",
    "policy.py",
    "reliability.py",
    "pipeline.py",
    "experiment.py",
    "benchmark_suite.py",
    "README.md",
    "requirements-research.txt",
    "pytest.ini",
    "benchmark_joint_q1.json",
    "benchmark_joint_q1_medium.json",
    "benchmark_joint_q1_large.json",
    "benchmark_korea_noaa_q1_medium.json",
    "benchmark_japan_q1_medium.json",
    "benchmark_china_q1_medium.json",
    "benchmark_us_q1_medium.json"
)

$directories = @(
    "task_cli",
    "scripts",
    "tests"
)

$reviewFiles = @(
    "review\REVIEW_SUMMARY.md",
    "review\NMI_UPDATE_GAP_ANALYSIS_20260317.md",
    "review\NMI_BENCHMARK_COMPARISON_20260317.csv",
    "review\NMI_BENCHMARK_COMPARISON_20260317.json",
    "review\NMI_BENCHMARK_COMPARISON_20260317.md",
    "review\nmi_section_revision_instructions_extracted.txt",
    "review\pre_submission_feedback_analysis_extracted.txt",
    "review\saids_nmi_rewritten_from_feedback_extracted.txt"
)

$dataFiles = @(
    "data\joint_weather_network_q1_2025\framework_joint_q1.json",
    "data\joint_weather_network_q1_2025\joint_build_manifest.json",
    "data\joint_weather_network_q1_2025\station_registry_q1.csv",
    "data\joint_weather_network_q1_2025\station_registry_summary.json",
    "data\external_region_pilots_q1_2025_summary.json",
    "data\korea_data_extensions_20260317_summary.json",
    "data\noaa_isd_korea_q1_2025\framework_isd_q1.json",
    "data\noaa_isd_korea_q1_2025\noaa_download_manifest.json",
    "data\noaa_isd_japan_q1_2025\framework_isd_q1.json",
    "data\noaa_isd_japan_q1_2025\noaa_download_manifest.json",
    "data\noaa_isd_china_q1_2025\framework_isd_q1.json",
    "data\noaa_isd_china_q1_2025\noaa_download_manifest.json",
    "data\noaa_isd_us_q1_2025\framework_isd_q1.json",
    "data\noaa_isd_us_q1_2025\noaa_download_manifest.json",
    "data\event_truth_q1_2025\kma_event_manifest.json",
    "data\event_truth_q1_2025_station\station_event_truth_q1.csv",
    "data\qc_maintenance_metadata_q1_2025\q1_operational_qc_summary.csv"
)

$benchmarkNames = @(
    "benchmark_joint_q1",
    "benchmark_joint_q1_medium",
    "benchmark_joint_q1_large",
    "benchmark_korea_noaa_q1_medium",
    "benchmark_japan_q1_medium",
    "benchmark_china_q1_medium",
    "benchmark_us_q1_medium"
)

$summaryContent = @"
# NMI Package Summary

This bundle contains the current manuscript-facing code, benchmark configs,
review notes, and completed benchmark outputs without raw observation archives.

## Included benchmark result sets
- benchmark_joint_q1
- benchmark_joint_q1_medium
- benchmark_joint_q1_large
- benchmark_korea_noaa_q1_medium
- benchmark_japan_q1_medium
- benchmark_china_q1_medium
- benchmark_us_q1_medium

## Included comparison artifact
- review/NMI_BENCHMARK_COMPARISON_20260317.md
- review/NMI_BENCHMARK_COMPARISON_20260317.csv
- review/NMI_BENCHMARK_COMPARISON_20260317.json

## Excluded from the package
- Raw AWS/ASOS/NOAA/ERA5 CSV and NetCDF files
- Virtual environments and caches
- Detached run heartbeats, cache blobs, and temporary folders
"@

$summaryParent = Split-Path -Parent $reviewSummaryPath
if (-not (Test-Path $summaryParent)) {
    New-Item -ItemType Directory -Path $summaryParent -Force | Out-Null
}
Set-Content -Path $reviewSummaryPath -Value $summaryContent -Encoding UTF8

foreach ($relativePath in $topLevelFiles) {
    Copy-RelativePath $relativePath
}

foreach ($relativePath in $directories) {
    Copy-RelativePath $relativePath
}

foreach ($relativePath in $reviewFiles) {
    Copy-RelativePath $relativePath
}

Copy-RelativePath "review\NMI_PACKAGE_SUMMARY_20260317.md"

foreach ($relativePath in $dataFiles) {
    Copy-RelativePath $relativePath
}

foreach ($benchmarkName in $benchmarkNames) {
    Copy-BenchmarkArtifacts $benchmarkName
}

Get-ChildItem -Path $stageDir -Directory -Recurse -Force |
    Where-Object { $_.Name -in @("__pycache__", ".pytest_cache") } |
    Remove-Item -Recurse -Force

Get-ChildItem -Path $stageDir -File -Recurse -Force |
    Where-Object { $_.Extension -in @(".pyc", ".pyo") } |
    Remove-Item -Force

Compress-Archive -Path (Join-Path $stageDir "*") -DestinationPath $zipPath -Force

Get-Item $zipPath | Select-Object FullName, Length, LastWriteTime
