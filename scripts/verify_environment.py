"""Environment validation script — run after setup to confirm v0.1.0 readiness."""

from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path


def check_python_version() -> tuple[bool, str]:
    v = sys.version_info
    ok = v.major == 3 and v.minor == 11
    return ok, f"{v.major}.{v.minor}.{v.micro}"


def check_imports() -> list[tuple[str, bool, str]]:
    packages = [
        "cv2",
        "mediapipe",
        "ultralytics",
        "numpy",
        "pandas",
        "sklearn",
        "yaml",
        "dotenv",
        "psutil",
    ]
    results = []
    for pkg in packages:
        try:
            importlib.import_module(pkg)
            results.append((pkg, True, "ok"))
        except ImportError as exc:
            results.append((pkg, False, str(exc)))
    return results


def check_env_vars() -> list[tuple[str, bool, str]]:
    vars_to_check = ["COCKPIT_DATA_ROOT", "COCKPIT_MODELS_ROOT", "MLFLOW_TRACKING_URI"]
    results = []
    for var in vars_to_check:
        val = os.environ.get(var)
        if val:
            exists = Path(val).exists() if not val.startswith("file:") else True
            results.append((var, exists, val))
        else:
            results.append((var, False, "not set"))
    return results


def check_gpu() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"CUDA available: {torch.cuda.get_device_name(0)}"
        return "CUDA not available (CPU mode)"
    except ImportError:
        return "PyTorch not installed yet"


def main() -> int:
    print("=" * 60)
    print("CockpitSentinel Environment Check")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")

    ok, ver = check_python_version()
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] Python version: {ver} (required: 3.11.x)")

    print("\n--- Package imports ---")
    all_ok = ok
    for pkg, passed, msg in check_imports():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {pkg}: {msg}")
        all_ok = all_ok and passed

    print("\n--- Environment variables ---")
    for var, passed, msg in check_env_vars():
        status = "PASS" if passed else "WARN"
        print(f"[{status}] {var}: {msg}")

    print(f"\n--- GPU ---\n{check_gpu()}")

    print("\n" + "=" * 60)
    if all_ok:
        print("Environment ready.")
        return 0
    print("Environment incomplete — fix FAIL items above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
