Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    # Use the latest available Python (>=3.11)
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        & py -3 -m venv (Join-Path $repoRoot ".venv")
    } else {
        python -m venv (Join-Path $repoRoot ".venv")
    }
}

& $venvPython -m pip install --quiet --disable-pip-version-check python-dotenv
& $venvPython (Join-Path $repoRoot "scripts\setup_environment.py")
