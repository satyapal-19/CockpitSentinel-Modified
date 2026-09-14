"""Unit tests for modular device selection and hardware acceleration utilities."""

from unittest.mock import patch

import pytest

from cockpit_sentinel.utils.device import (
    get_cuda_device_name,
    get_cuda_memory_info,
    get_device_diagnostic_info,
    get_mediapipe_delegate,
    resolve_device,
)


def test_resolve_device_explicit_cpu():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("CPU") == "cpu"


def test_resolve_device_auto_fallback_to_cpu_when_no_gpu():
    with patch("cockpit_sentinel.utils.device.is_cuda_available", return_value=False), patch(
        "cockpit_sentinel.utils.device.is_mps_available", return_value=False
    ):
        assert resolve_device("auto") == "cpu"
        assert resolve_device(None) == "cpu"


def test_resolve_device_auto_picks_cuda_when_available():
    with patch("cockpit_sentinel.utils.device.is_cuda_available", return_value=True):
        assert resolve_device("auto") == "cuda:0"


def test_resolve_device_auto_picks_mps_when_available():
    with patch("cockpit_sentinel.utils.device.is_cuda_available", return_value=False), patch(
        "cockpit_sentinel.utils.device.is_mps_available", return_value=True
    ):
        assert resolve_device("auto") == "mps"


def test_resolve_device_raises_when_cuda_requested_but_unavailable():
    with (
        patch("cockpit_sentinel.utils.device.is_cuda_available", return_value=False),
        pytest.raises(ValueError, match="CUDA device requested"),
    ):
        resolve_device("cuda")


def test_resolve_device_normalizes_numeric_device_id():
    with patch("cockpit_sentinel.utils.device.is_cuda_available", return_value=True):
        assert resolve_device("0") == "cuda:0"
        assert resolve_device("1") == "cuda:1"


def test_get_mediapipe_delegate_cpu():
    from mediapipe.tasks.python.core.base_options import BaseOptions

    delegate = get_mediapipe_delegate("cpu")
    assert delegate == BaseOptions.Delegate.CPU


def test_get_device_diagnostic_info_structure():
    info = get_device_diagnostic_info()
    assert "platform" in info
    assert "cuda_available" in info
    assert "mps_available" in info
    assert "resolved_default" in info


def test_cuda_device_name_when_no_cuda():
    with patch("cockpit_sentinel.utils.device.is_cuda_available", return_value=False):
        assert get_cuda_device_name(0) is None
        assert get_cuda_memory_info(0) is None
