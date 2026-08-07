# Team Environment Guide

Each member clones the same repository and creates a local environment. Do not copy a `.venv`
folder between laptops.

## Shared Baseline

All three laptops use Python 3.11 and the same branch before work begins.

```powershell
git clone <repository-url>
cd CockpitSentinel
Copy-Item .env.example .env
.\scripts\setup_environment.ps1
.\.venv\Scripts\python .\scripts\verify_environment.py
.\.venv\Scripts\python -m pytest -q
```

Set local locations in `.env`. Dataset archives, trained weights, logs, and credentials stay local.

## CPU Laptop

Use the standard setup for data organization, OpenCV and MediaPipe development, integration,
automated tests, and demonstration runs.

## NVIDIA Laptops

Use the standard setup first, then install the CUDA-enabled PyTorch build with the index URL that
matches the installed NVIDIA driver. Check the driver with `nvidia-smi`, then run:

```powershell
.\scripts\setup_nvidia_environment.ps1 -TorchIndexUrl https://download.pytorch.org/whl/cu126
```

Validate GPU access:

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Use these laptops for YOLO fine-tuning and training experiments. Store completed model weights in
the agreed shared storage location and record the dataset version, configuration, and metrics.

## Git Workflow

Create a branch for each change: `feature/drowsiness-signals`, `feature/distraction-detection`, or
`feature/integration-alerts`. Open a pull request into `main`, require one teammate review, and run
the environment check and tests before merging.

Keep `main` runnable. The technical manager owns releases, repository settings, and merge decisions.
