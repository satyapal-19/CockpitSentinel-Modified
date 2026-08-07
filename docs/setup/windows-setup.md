# Windows Setup

This repository is currently a setup scaffold for the driver monitoring system described
in the synopsis PDF. It prepares the environment, storage, baseline dependencies, and starter assets
without claiming the monitoring application itself is complete.

## What The Setup Does

- creates or reuses the local `.venv`
- installs Python dependencies from the project requirement files
- creates the expected project storage under `D:\CockpitSentinel`
- prepares dataset folders for NTHU, YawDD, State Farm, and extra Kaggle experiments
- downloads `yolov8n.pt` once into `D:\CockpitSentinel\models\pretrained`
- preserves anything that already exists

## First-Time Use

```powershell
.\scripts\setup_environment.ps1
```

## Dataset Downloads

Several datasets referenced in the synopsis are distributed through Kaggle and require credentials.
Add these to `.env` before re-running the setup if you want automatic downloads:

```env
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

If credentials are missing, the setup still prepares the correct folder structure and leaves
manifest files behind so the team can add the archives manually later.

## Verification

```powershell
.\.venv\Scripts\Activate.ps1
python .\scripts\verify_environment.py
pytest
```
