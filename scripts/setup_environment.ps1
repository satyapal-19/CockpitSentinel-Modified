param(
    [string]$PythonVersion = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    $pythonVersionFile = Join-Path $repoRoot ".python-version"
    $desiredVersion = $PythonVersion
    if (-not $desiredVersion -and (Test-Path $pythonVersionFile)) {
        $desiredVersion = (Get-Content $pythonVersionFile).Trim()
    }

    $created = $false
    $py = Get-Command py -ErrorAction SilentlyContinue

    if ($py) {
        # Try specified or preferred Python versions (e.g., 3.11, 3.12) to avoid incompatible Python 3.13
        $versionsToTry = @()
        if ($desiredVersion) { $versionsToTry += $desiredVersion }
        $versionsToTry += @("3.11", "3.12", "3")

        foreach ($ver in $versionsToTry) {
            $test = & py "-$ver" -c "import sys; print(sys.version_info.major, sys.version_info.minor)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $test) {
                Write-Host "Creating virtual environment using py -$ver..."
                & py "-$ver" -m venv (Join-Path $repoRoot ".venv")
                $created = $true
                break
            }
        }
    }

    if (-not $created) {
        Write-Host "Creating virtual environment using system python..."
        python -m venv (Join-Path $repoRoot ".venv")
    }
}

& $venvPython -m pip install --quiet --disable-pip-version-check python-dotenv
& $venvPython (Join-Path $repoRoot "scripts\setup_environment.py")
