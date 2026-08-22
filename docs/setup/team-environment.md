# Team Environment Guide

Each member clones the same repository and creates a local environment. Do not copy a `.venv`
folder between machines.

## Shared Baseline

All machines need Python 3.11 or newer.

### Windows

```powershell
git clone <repository-url>
cd CockpitSentinel
.\scripts\setup_environment.ps1
.\. venv\Scripts\Activate.ps1
python scripts\verify_environment.py
pytest -q
```

### macOS / Linux

```bash
git clone <repository-url>
cd CockpitSentinel
bash scripts/setup.sh
source .venv/bin/activate
python scripts/verify_environment.py
pytest -q
```

The setup script creates `.env` from `.env.example` if it doesn't exist.
Edit `.env` to customise storage paths. By default everything stays inside the repo directory.

## CPU Machine

Use the standard setup for data organisation, OpenCV and MediaPipe development, integration,
automated tests, and demonstration runs.

## NVIDIA GPU Machine

Use the standard setup first, then install the CUDA-enabled PyTorch build:

```powershell
.\scripts\setup_nvidia_environment.ps1 -TorchIndexUrl https://download.pytorch.org/whl/cu126
```

Validate GPU access:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Git Workflow

Create a branch for each change: `feature/drowsiness-signals`, `feature/distraction-detection`, or
`feature/integration-alerts`. Open a pull request into `main`, require one teammate review, and run
the environment check and tests before merging.

Keep `main` runnable.
