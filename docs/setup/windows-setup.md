# Windows Setup

## Prerequisites

- Python 3.11 or newer (install from [python.org](https://www.python.org/downloads/))
- Git

Make sure `python --version` or `py --version` works in PowerShell.

## First-Time Setup

```powershell
git clone <repository-url>
cd CockpitSentinel
.\scripts\setup_environment.ps1
```

This will:
- Create a `.venv` virtual environment
- Copy `.env.example` to `.env` (if `.env` doesn't exist yet)
- Install all dependencies from `requirements.txt`
- Download pretrained models (YOLOv8, YOLOWorld, MediaPipe Face Landmarker)
- Set up the data directory structure

By default, all data and models are stored inside the repository under `data/`, `models/`,
and `experiments/`. These directories are git-ignored.

To use a custom storage path, edit `.env`:

```ini
COCKPIT_DATA_ROOT=E:\MyData\CockpitSentinel
COCKPIT_MODELS_ROOT=E:\MyData\CockpitSentinel\models
```

## Running the Monitor

```powershell
.\.venv\Scripts\Activate.ps1
cockpit-sentinel
```

Or:

```powershell
python scripts\run_drowsiness_monitor.py
```

> **Important:** Do NOT run `.py` files directly (e.g. `.\scripts\run_drowsiness_monitor.py`).
> On Windows this may open your code editor instead of executing the script.

## Dataset Downloads

Several datasets are distributed through Kaggle and require credentials.
Add these to `.env` before re-running the setup:

```ini
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

If credentials are missing, the setup still creates the folder structure with manifest files
so archives can be added manually.

## Verification

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\verify_environment.py
pytest
```
