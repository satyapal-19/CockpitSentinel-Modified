"""Hardware acceleration and compute device detection utilities."""

from __future__ import annotations

import logging
import platform
from typing import Any

logger = logging.getLogger(__name__)


def is_cuda_available() -> bool:
    """Return True if PyTorch can access an NVIDIA CUDA GPU."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def is_mps_available() -> bool:
    """Return True if PyTorch can access Apple Silicon Metal Performance Shaders (MPS)."""
    try:
        import torch

        return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except ImportError:
        return False


def get_cuda_device_name(index: int = 0) -> str | None:
    """Return the name of the CUDA device, or None if unavailable."""
    try:
        import torch

        if torch.cuda.is_available() and index < torch.cuda.device_count():
            return str(torch.cuda.get_device_name(index))
    except Exception:
        pass
    return None


def get_cuda_memory_info(index: int = 0) -> dict[str, float] | None:
    """Return total and reserved CUDA memory in gigabytes, or None if unavailable."""
    try:
        import torch

        if torch.cuda.is_available() and index < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(index)
            total_gb = props.total_memory / (1024**3)
            allocated_gb = torch.cuda.memory_allocated(index) / (1024**3)
            return {
                "total_gb": round(total_gb, 2),
                "allocated_gb": round(allocated_gb, 2),
            }
    except Exception:
        pass
    return None


def resolve_device(requested: str | None = None) -> str:
    """Resolve a user device string or auto-detect the best available hardware accelerator.

    Args:
        requested: Device choice ('auto', 'cpu', 'cuda', 'cuda:0', 'mps', '0', etc.)
                   If None or 'auto', automatically selects CUDA -> MPS -> CPU.

    Returns:
        A normalized device string suitable for PyTorch and Ultralytics
        (e.g. 'cuda:0', 'mps', 'cpu').

    Raises:
        ValueError: If a specific hardware accelerator is requested but not available.
    """
    if requested is None or requested.strip().lower() in ("", "auto"):
        if is_cuda_available():
            return "cuda:0"
        if is_mps_available():
            return "mps"
        return "cpu"

    device_str = requested.strip().lower()

    if device_str == "cpu":
        return "cpu"

    if device_str == "mps":
        if not is_mps_available():
            raise ValueError(
                "Apple Silicon MPS device requested, but MPS is not available on this system."
            )
        return "mps"

    if device_str.startswith("cuda") or device_str.isdigit():
        if not is_cuda_available():
            raise ValueError(
                f"CUDA device requested ('{requested}'), but CUDA is not available. "
                "Ensure an NVIDIA GPU with proper drivers and CUDA-enabled PyTorch are installed."
            )
        # Normalize numeric GPU ID like '0' to 'cuda:0'
        if device_str.isdigit():
            return f"cuda:{device_str}"
        if device_str == "cuda":
            return "cuda:0"
        return device_str

    # Pass through other explicit device specifications (e.g. directml / openvino if supported)
    return device_str


def get_mediapipe_delegate(device: str | None = None) -> Any:
    """Return the appropriate MediaPipe BaseOptions.Delegate enum value.

    Gracefully falls back to CPU delegate if GPU delegate is requested but not supported.
    """
    try:
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except ImportError:
        return None

    resolved = resolve_device(device)
    if resolved.startswith("cuda") or resolved == "gpu":
        try:
            return BaseOptions.Delegate.GPU
        except (AttributeError, Exception) as exc:
            logger.warning("MediaPipe GPU delegate requested but unavailable: %s. Using CPU.", exc)
            return BaseOptions.Delegate.CPU

    return BaseOptions.Delegate.CPU


def get_device_diagnostic_info() -> dict[str, Any]:
    """Gather complete hardware acceleration details for diagnostics and environment reports."""
    cuda_avail = is_cuda_available()
    mps_avail = is_mps_available()

    info: dict[str, Any] = {
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cuda_available": cuda_avail,
        "mps_available": mps_avail,
        "resolved_default": resolve_device("auto"),
    }

    if cuda_avail:
        try:
            import torch

            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = get_cuda_device_name(0)
            info["cuda_memory"] = get_cuda_memory_info(0)
            info["cuda_version"] = torch.version.cuda
        except Exception as exc:
            info["cuda_error"] = str(exc)

    return info
