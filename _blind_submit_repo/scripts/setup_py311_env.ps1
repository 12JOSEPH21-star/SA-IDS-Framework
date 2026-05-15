param(
    [string]$EnvDir = ".venv311",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

function Resolve-Python311 {
    try {
        $pythonExe = py -3.11 -c "import sys; print(sys.executable)" 2>$null
    } catch {
        $pythonExe = $null
    }

    $pythonExe = ($pythonExe | Select-Object -First 1).Trim()
    if (-not $pythonExe -or $pythonExe -match "Anaconda") {
        winget install --id Python.Python.3.11 -e --scope user --silent --accept-package-agreements --accept-source-agreements | Out-Host
        $pythonExe = (py -3.11 -c "import sys; print(sys.executable)" | Select-Object -First 1).Trim()
    }

    if (-not $pythonExe) {
        throw "Unable to resolve a usable Python 3.11 interpreter."
    }
    return $pythonExe
}

$python311 = Resolve-Python311

if ($Recreate -and (Test-Path $EnvDir)) {
    Remove-Item -Recurse -Force $EnvDir
}

if (-not (Test-Path $EnvDir)) {
    & $python311 -m venv $EnvDir
}

$venvPython = Join-Path (Resolve-Path $EnvDir) "Scripts\\python.exe"
& $venvPython -m pip install -r requirements-research.txt
& $venvPython -c "import sys, torch, gpytorch; print(sys.executable); print(sys.version); print('torch', torch.__version__); print('gpytorch', gpytorch.__version__)"
