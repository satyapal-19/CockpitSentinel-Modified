"""Bootstrap the CockpitSentinel development environment."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORAGE_ROOT = Path("D:/CockpitSentinel")
ENV_FILE = REPO_ROOT / ".env"
ENV_EXAMPLE = REPO_ROOT / ".env.example"
VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

REPO_DIRS = [
    REPO_ROOT / "logs" / "app",
    REPO_ROOT / "logs" / "env",
    REPO_ROOT / "logs" / "training",
    REPO_ROOT / "notebooks" / "01_eda",
    REPO_ROOT / "notebooks" / "02_preprocessing",
    REPO_ROOT / "notebooks" / "03_model_experiments",
]

DATASET_LAYOUT = {
    "nthu": "NTHU Driver Drowsiness Detection dataset for fatigue experiments.",
    "yawdd": "YawDD dataset for yawning and drowsiness analysis.",
    "state_farm": "State Farm distracted driver detection dataset.",
    "kaggle_driver_monitoring": "Additional Kaggle driver monitoring datasets.",
    "custom": "Custom webcam captures for integration testing and demos.",
}

KAGGLE_DATASETS = {
    "nthu": ("datasets", "taweilo/driver-drowsiness-dataset-ddd"),
    "state_farm": ("competitions", "state-farm-distracted-driver-detection"),
}

STORAGE_RELATIVE_DIRS = [
    Path("backups"),
    Path("cache"),
    Path("cache/kaggle"),
    Path("cache/ultralytics"),
    Path("data"),
    Path("data/raw"),
    Path("data/processed"),
    Path("data/processed/features"),
    Path("data/processed/splits"),
    Path("data/external"),
    Path("experiments"),
    Path("experiments/mlruns"),
    Path("models"),
    Path("models/pretrained"),
    Path("models/trained"),
    Path("models/trained/drowsiness"),
    Path("models/trained/distraction"),
    Path("models/trained/fusion"),
]


def run(
    command: list[str], *, env: dict[str, str] | None = None, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(command)}")
    return subprocess.run(
        command,
        check=True,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=capture,
    )


def check_python() -> None:
    version = sys.version_info
    if (version.major, version.minor) != (3, 11):
        raise SystemExit("Python 3.11 is required for this project setup.")


def ensure_env_file() -> None:
    if ENV_FILE.exists():
        print(f"[skip] {ENV_FILE.name} already exists")
        return
    shutil.copyfile(ENV_EXAMPLE, ENV_FILE)
    print(f"[ok] created {ENV_FILE.name} from {ENV_EXAMPLE.name}")


def ensure_repo_dirs() -> None:
    for path in REPO_DIRS:
        path.mkdir(parents=True, exist_ok=True)
        print(f"[ok] ensured repo directory: {path}")


def resolve_storage_root() -> Path:
    values = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}
    configured_root = values.get("COCKPIT_DATA_ROOT")
    if configured_root:
        return Path(configured_root)
    return DEFAULT_STORAGE_ROOT


def ensure_storage_dirs(storage_root: Path) -> None:
    for relative_path in STORAGE_RELATIVE_DIRS:
        target = storage_root / relative_path
        target.mkdir(parents=True, exist_ok=True)
        print(f"[ok] ensured storage directory: {target}")

    external_root = storage_root / "data" / "external"
    for dataset_name, description in DATASET_LAYOUT.items():
        dataset_dir = external_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        manifest = dataset_dir / "README.txt"
        if manifest.exists():
            print(f"[skip] dataset manifest already exists: {manifest}")
            continue
        manifest.write_text(
            f"{dataset_name}\n{description}\nPlace downloaded archives or extracted files here.\n",
            encoding="utf-8",
        )
        print(f"[ok] wrote dataset manifest: {manifest}")


def install_requirements() -> None:
    # Keep pip output compact so setup remains reliable in constrained terminals.
    run(
        [
            str(VENV_PYTHON),
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            "--progress-bar",
            "off",
            "-r",
            "requirements.txt",
            "-r",
            "requirements-dev.txt",
        ]
    )


def copy_yolo_weights(storage_root: Path) -> None:
    target = storage_root / "models" / "pretrained" / "yolov8n.pt"
    if target.exists():
        print(f"[skip] pretrained weights already present: {target}")
        return

    python_code = (
        "from pathlib import Path; "
        "from ultralytics.utils.downloads import attempt_download_asset; "
        "print(Path(attempt_download_asset('yolov8n.pt')).resolve())"
    )
    completed = run([str(VENV_PYTHON), "-c", python_code], capture=True)
    source = Path(completed.stdout.strip().splitlines()[-1])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"[ok] copied pretrained weights to {target}")


def kaggle_credentials() -> tuple[str | None, str | None]:
    env_values = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}
    username = os.environ.get("KAGGLE_USERNAME") or env_values.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY") or env_values.get("KAGGLE_KEY")
    return username, key


def maybe_download_kaggle_datasets(storage_root: Path) -> None:
    username, key = kaggle_credentials()
    if not username or not key:
        print("[skip] Kaggle credentials are not configured; dataset download skipped")
        return

    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = username
    env["KAGGLE_KEY"] = key

    external_root = storage_root / "data" / "external"
    for folder_name, (kind, identifier) in KAGGLE_DATASETS.items():
        target_dir = external_root / folder_name
        has_content = any(child.name != "README.txt" for child in target_dir.iterdir())
        if has_content:
            print(f"[skip] dataset already present: {target_dir}")
            continue

        if kind == "competitions":
            command = [
                str(VENV_PYTHON),
                "-m",
                "kaggle",
                "competitions",
                "download",
                "-c",
                identifier,
                "-p",
                str(target_dir),
            ]
        else:
            command = [
                str(VENV_PYTHON),
                "-m",
                "kaggle",
                "datasets",
                "download",
                "-d",
                identifier,
                "-p",
                str(target_dir),
                "--unzip",
            ]

        run(command, env=env)


def main() -> int:
    check_python()
    ensure_env_file()
    ensure_repo_dirs()
    storage_root = resolve_storage_root()
    ensure_storage_dirs(storage_root)
    install_requirements()
    copy_yolo_weights(storage_root)
    maybe_download_kaggle_datasets(storage_root)
    print("[done] environment bootstrap complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
