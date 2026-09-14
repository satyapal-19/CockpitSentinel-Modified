param(
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu121"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    throw "Run scripts\setup_environment.ps1 before configuring NVIDIA PyTorch."
}

# Resolve nvidia-smi either from PATH or well-known Windows locations
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
    $fallbackPaths = @(
        "$env:SystemRoot\System32\nvidia-smi.exe",
        "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        "${env:ProgramFiles(x86)}\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    )
    foreach ($path in $fallbackPaths) {
        if (Test-Path $path) {
            $nvidiaSmi = $path
            break
        }
    }
}

if (-not $nvidiaSmi) {
    throw "NVIDIA driver tools (nvidia-smi) were not found. Please install or update your NVIDIA GPU drivers."
}

Write-Host "Checking NVIDIA GPU status..."
& $nvidiaSmi

Write-Host "Installing CUDA-enabled PyTorch from: $TorchIndexUrl"
& $venvPython -m pip uninstall -y torch torchvision torchaudio
& $venvPython -m pip install --index-url $TorchIndexUrl torch torchvision torchaudio
& $venvPython -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('CUDA active:', torch.cuda.get_device_name(0))"
