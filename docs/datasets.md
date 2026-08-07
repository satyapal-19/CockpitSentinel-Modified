# Dataset Registry

All datasets are stored under `{COCKPIT_DATA_ROOT}/data/`. Raw data is immutable.

The synopsis specifically names the NTHU Driver Drowsiness Detection dataset, YawDD, State Farm
Distracted Driver Detection, and additional Kaggle driver monitoring datasets. The setup flow
prepares those locations in advance.

| Dataset | Location | Source | Purpose | Status |
|---------|----------|--------|---------|--------|
| NTHU DDD | `external/nthu/` | Kaggle or public mirror | Drowsiness | Prepared by setup |
| YawDD | `external/yawdd/` | Dataset mirror / manual import | Yawning | Prepared by setup |
| State Farm | `external/state_farm/` | Kaggle | Distraction | Prepared by setup |
| Kaggle driver monitoring | `external/kaggle_driver_monitoring/` | Kaggle | Extra experiments | Prepared by setup |
| Custom webcam | `external/custom/` | Local capture | Integration testing | Empty placeholder |

## Setup Behavior

- `scripts/setup_environment.py` creates dataset folders only if missing.
- Existing downloads are preserved and never overwritten.
- Kaggle downloads run only when credentials are configured.
- A `README.txt` manifest is written into empty dataset folders to show what belongs there.

## Directory Layout

```text
data/
|-- raw/          # Original downloads (never modify)
|-- processed/    # Resized, normalized, train/val/test splits
`-- external/     # Per-dataset subfolders
```
