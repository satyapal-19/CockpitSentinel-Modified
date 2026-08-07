Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\\Scripts\\python.exe"

if (-not (Test-Path $venvPython)) {
    py -3.11 -m venv (Join-Path $repoRoot ".venv")
}

& $venvPython (Join-Path $repoRoot "scripts\\setup_environment.py")
