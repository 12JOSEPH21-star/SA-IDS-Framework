param(
    [string]$BundleName = "silence-aware-ids-review-bundle-clean-20260315c.zip"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$distDir = Join-Path $repoRoot "dist"
$archiveDir = Join-Path $distDir "archive"
$stageDir = Join-Path $distDir "review_bundle_stage"
$zipPath = Join-Path $distDir $BundleName

New-Item -ItemType Directory -Force -Path $archiveDir | Out-Null

Get-ChildItem -Path $repoRoot -Filter "*.zip" -File | ForEach-Object {
    Move-Item -Force $_.FullName (Join-Path $archiveDir $_.Name)
}

if (Test-Path $stageDir) {
    Remove-Item -Recurse -Force $stageDir
}
if (Test-Path $zipPath) {
    Remove-Item -Force $zipPath
}
New-Item -ItemType Directory -Force -Path $stageDir | Out-Null

$topFiles = @(
    "README.md",
    "requirements-research.txt",
    "pytest.ini",
    "models.py",
    "policy.py",
    "reliability.py",
    "pipeline.py",
    "experiment.py",
    "benchmark_suite.py",
    "framework_config.json",
    "framework_isd.json",
    "research_config.json",
    "benchmark_joint_q1.json",
    "benchmark_joint_q1_medium.json"
)

foreach ($relativePath in $topFiles) {
    $sourcePath = Join-Path $repoRoot $relativePath
    if (Test-Path $sourcePath) {
        Copy-Item $sourcePath -Destination $stageDir
    }
}

$dirTargets = @("task_cli", "tests", "scripts", "review")
foreach ($relativePath in $dirTargets) {
    $sourcePath = Join-Path $repoRoot $relativePath
    if (Test-Path $sourcePath) {
        Copy-Item $sourcePath -Destination $stageDir -Recurse
    }
}

$fileTargets = @(
    "data\era5_korea_q1_2025\era5_download_manifest.json",
    "data\era5_korea_q1_2025\framework_era5_q1.json",
    "data\noaa_isd_korea_q1_2025\noaa_download_manifest.json",
    "data\noaa_isd_korea_q1_2025\framework_isd_q1.json",
    "data\noaa_isd_korea_q1_2025\isd_station_metadata.csv",
    "data\joint_weather_network_q1_2025\joint_build_manifest.json",
    "data\joint_weather_network_q1_2025\framework_joint_q1.json",
    "data\joint_weather_network_q1_2025\framework_joint_q1_smoke.json",
    "data\asos_korea_q1_2025\kma_download_manifest.json",
    "data\asos_korea_q1_2025\asos_download_retry.log",
    "data\event_truth_q1_2025\kma_event_manifest.json",
    "data\event_truth_q1_2025_info\kma_event_manifest.json",
    "data\nwp_anchor_recent_ldaps\kma_nwp_manifest.json",
    "data\nwp_anchor_recent_rdaps\kma_nwp_manifest.json"
)

foreach ($relativePath in $fileTargets) {
    $sourcePath = Join-Path $repoRoot $relativePath
    if (Test-Path $sourcePath) {
        $destinationPath = Join-Path $stageDir $relativePath
        New-Item -ItemType Directory -Force -Path (Split-Path $destinationPath) | Out-Null
        Copy-Item $sourcePath -Destination $destinationPath
    }
}

$outputDirs = @(
    "outputs\benchmark_joint_q1",
    "outputs\framework_run"
)

foreach ($relativePath in $outputDirs) {
    $sourcePath = Join-Path $repoRoot $relativePath
    if (Test-Path $sourcePath) {
        $destinationParent = Join-Path $stageDir (Split-Path $relativePath)
        New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
        Copy-Item $sourcePath -Destination $destinationParent -Recurse
    }
}

$cacheDir = Join-Path $stageDir "outputs\benchmark_joint_q1\cache"
if (Test-Path $cacheDir) {
    Remove-Item -Recurse -Force $cacheDir
}

Get-ChildItem -Path $stageDir -Recurse -Directory -Force |
    Where-Object { $_.Name -in @("__pycache__", ".pytest_cache", ".venv311", ".venv311a", ".venv311b") } |
    Remove-Item -Recurse -Force

Get-ChildItem -Path $stageDir -Recurse -Include *.pyc -File | Remove-Item -Force

Compress-Archive -Path (Join-Path $stageDir "*") -DestinationPath $zipPath -CompressionLevel Optimal

$zipItem = Get-Item $zipPath
$fileCount = (Get-ChildItem -Path $stageDir -Recurse -File | Measure-Object).Count

[PSCustomObject]@{
    Zip = $zipItem.FullName
    Bytes = $zipItem.Length
    Files = $fileCount
} | Format-List
