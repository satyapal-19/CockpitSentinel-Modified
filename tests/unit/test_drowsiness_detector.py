"""Tests for drowsiness geometry and signal stabilization."""

import pytest

from cockpit_sentinel.domain import DriverSignals
from cockpit_sentinel.drowsiness.detector import (
    DrowsinessConfig,
    SignalStabilizer,
    eye_aspect_ratio,
    mouth_aspect_ratio,
)


def test_eye_aspect_ratio_matches_known_geometry():
    ratio = eye_aspect_ratio([(0, 0), (0, 1), (1, 1), (4, 0), (1, -1), (0, -1)])

    assert ratio == pytest.approx(0.5)


def test_mouth_aspect_ratio_matches_known_geometry():
    ratio = mouth_aspect_ratio([(0, 0), (4, 4), (4, -4), (8, 0)])

    assert ratio == pytest.approx(1.0)


def test_stabilizer_requires_consecutive_active_frames():
    stabilizer = SignalStabilizer(minimum_consecutive_frames=3)
    raw = DriverSignals(eyes_closed=True)

    assert not stabilizer.update(raw).eyes_closed
    assert not stabilizer.update(raw).eyes_closed
    assert stabilizer.update(raw).eyes_closed
    assert not stabilizer.update(DriverSignals()).eyes_closed


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        DrowsinessConfig(minimum_consecutive_frames=0)
