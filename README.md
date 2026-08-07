# CockpitSentinel

Driver monitoring system using computer vision for real-time drowsiness and distraction detection.

## Stack

- Python 3.11
- OpenCV, MediaPipe, Ultralytics (YOLOv8)
- PyTorch for local development and model training
- Scikit-Learn, NumPy, Pandas

## Quick Start

```powershell
# 1. Bootstrap the environment and storage
.\scripts\setup_environment.ps1

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Validate the environment
python .\scripts\verify_environment.py

# 4. Run tests
pytest
```

## Data And Models

Large files live on `D:\CockpitSentinel\` and are intentionally kept out of git.

The setup prepares:

- `D:\CockpitSentinel\data\external\nthu`
- `D:\CockpitSentinel\data\external\yawdd`
- `D:\CockpitSentinel\data\external\state_farm`
- `D:\CockpitSentinel\data\external\kaggle_driver_monitoring`
- `D:\CockpitSentinel\data\external\custom`
- `D:\CockpitSentinel\models\pretrained\yolov8n.pt`

Kaggle downloads are attempted only when `KAGGLE_USERNAME` and `KAGGLE_KEY` are configured.
Otherwise the correct folder structure and manifest files are still created.

## Notes

- CPU and NVIDIA setup guidance is in `docs/setup/team-environment.md`.
- This repo is currently a setup scaffold, not the completed monitoring application.
- Detailed Windows setup notes live in `docs/setup/windows-setup.md`.

## Team Workflow

Use one shared repository and create a separate environment on each laptop. Do not commit
datasets, model weights, virtual environments, logs, or `.env` files. See
`docs/setup/team-environment.md` for the CPU/GPU split, branch workflow, and setup commands.

## Team

- Satyapal Gaikwad
- Darshan Patil
- Yashodhan Jadhav

Walchand College of Engineering, Sangli - Mini Project 2026-27
