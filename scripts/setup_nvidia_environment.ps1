param(
    [Parameter(Mandatory = $true)]
    [string]$TorchIndexUrl
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    throw "Run scripts\\setup_environment.ps1 before configuring NVIDIA PyTorch."
}

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "NVIDIA driver tools were not found. Install or update the NVIDIA driver first."
}

& nvidia-smi
& $venvPython -m pip uninstall -y torch torchvision torchaudio
& $venvPython -m pip install --index-url $TorchIndexUrl torch torchvision torchaudio
& $venvPython -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print(torch.cuda.get_device_name(0))"
