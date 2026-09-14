"""Environment validation script — run after setup to confirm readiness."""

from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path

# Ensure src is on pythonpath
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

MIN_PYTHON = (3, 11)


def check_python_version() -> tuple[bool, str]:
    v = sys.version_info
    ok = (v.major, v.minor) >= MIN_PYTHON
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
            results.append((var, False, "not set (using defaults)"))
    return results


def check_gpu() -> str:
    try:
        from cockpit_sentinel.utils.device import get_device_diagnostic_info

        diag = get_device_diagnostic_info()
        lines = []
        if diag.get("cuda_available"):
            lines.append(
                f"CUDA available: {diag.get('cuda_device_name')} (CUDA {diag.get('cuda_version')})"
            )
            mem = diag.get("cuda_memory")
            if mem:
                lines.append(
                    f"  VRAM: {mem.get('total_gb')} GB total "
                    f"({mem.get('allocated_gb')} GB allocated)"
                )
        elif diag.get("mps_available"):
            lines.append("Apple Silicon MPS (Metal Performance Shaders) available")
        else:
            lines.append("No GPU acceleration available (CPU mode)")

        lines.append(f"Default inference device: {diag.get('resolved_default')}")
        return "\n".join(lines)
    except Exception:
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
    min_ver = f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
    print(f"[{status}] Python version: {ver} (required: >= {min_ver})")
    v = sys.version_info
    if (v.major, v.minor) >= (3, 13):
        print(
            "  [NOTE] Python 3.13 detected. Pinned dependencies (numpy 1.x, mediapipe 0.10.x) "
            "are officially tested on Python 3.11-3.12."
        )

    print("\n--- Package imports ---")
    all_ok = ok
    for pkg, passed, msg in check_imports():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {pkg}: {msg}")
        all_ok = all_ok and passed

    print("\n--- Environment variables ---")
    for var, passed, msg in check_env_vars():
        status = "PASS" if passed else "INFO"
        print(f"[{status}] {var}: {msg}")

    print(f"\n--- Hardware & Acceleration ---\n{check_gpu()}")

    print("\n" + "=" * 60)
    if all_ok:
        print("Environment ready.")
        return 0
    print("Environment incomplete — fix FAIL items above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
