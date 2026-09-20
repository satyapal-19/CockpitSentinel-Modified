"""Unit tests for 3D Head Pose and HeadNoddingTracker (fatigue micro-droop detection)."""

from dataclasses import dataclass

import pytest

from cockpit_sentinel.drowsiness.detector import (
    HeadNoddingTracker,
    _estimate_head_pose,
    _estimate_head_yaw,
)


@dataclass
class DummyLandmark:
    x: float
    y: float


def test_head_nodding_tracker_validates_parameters():
    with pytest.raises(ValueError, match="pitch_threshold_degrees"):
        HeadNoddingTracker(pitch_threshold_degrees=0)
    with pytest.raises(ValueError, match="droop_duration_seconds"):
        HeadNoddingTracker(droop_duration_seconds=-1)
    with pytest.raises(ValueError, match="window_seconds"):
        HeadNoddingTracker(window_seconds=0)


def test_neutral_pitch_does_not_trigger_nodding():
    tracker = HeadNoddingTracker(pitch_threshold_degrees=15.0, droop_duration_seconds=1.0)

    for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
        is_nodding, duration = tracker.update(pitch=2.0, timestamp=t)
        assert not is_nodding
        assert duration == 0.0


def test_sustained_forward_droop_triggers_nodding():
    tracker = HeadNoddingTracker(pitch_threshold_degrees=15.0, droop_duration_seconds=1.0)

    # Normal driving posture
    tracker.update(pitch=5.0, timestamp=0.0)

    # Driver chin starts drooping forward (pitch >= 18 deg)
    is_nodding, dur = tracker.update(pitch=18.0, timestamp=1.0)
    assert not is_nodding
    assert dur == pytest.approx(0.0)

    # Droop maintained for 0.6 seconds (total 0.6s, not yet 1.0s)
    is_nodding, dur = tracker.update(pitch=19.5, timestamp=1.6)
    assert not is_nodding
    assert dur == pytest.approx(0.6)

    # Droop maintained past 1.0s (total 1.2s)
    is_nodding, dur = tracker.update(pitch=21.0, timestamp=2.2)
    assert is_nodding
    assert dur >= 1.0


def test_cyclic_nodding_triggers_nodding():
    tracker = HeadNoddingTracker(pitch_threshold_degrees=15.0, droop_duration_seconds=2.0)

    # Nod 1: dips down then jerks back up
    tracker.update(pitch=18.0, timestamp=0.0)
    tracker.update(pitch=5.0, timestamp=0.5)

    # Nod 2: second dip occurs within the observation window
    is_nodding, _ = tracker.update(pitch=19.0, timestamp=1.2)
    assert is_nodding


def test_head_nodding_tracker_reset():
    tracker = HeadNoddingTracker(droop_duration_seconds=1.0)
    tracker.update(pitch=20.0, timestamp=0.0)
    tracker.update(pitch=20.0, timestamp=1.5)

    is_nodding, _ = tracker.update(pitch=20.0, timestamp=1.6)
    assert is_nodding

    tracker.reset()
    is_nodding, dur = tracker.update(pitch=20.0, timestamp=2.0)
    assert not is_nodding
    assert dur == 0.0


def test_estimate_head_pose_returns_pitch_yaw_roll():
    import cv2
    import numpy as np

    from cockpit_sentinel.drowsiness.detector import HEAD_MODEL_POINTS

    width, height = 640, 480
    focal = float(width)
    camera_matrix = np.array(
        [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.array([[0.0], [0.0], [1000.0]], dtype=np.float64)
    projected, _ = cv2.projectPoints(HEAD_MODEL_POINTS, rvec, tvec, camera_matrix, np.zeros((4, 1)))

    landmarks = [DummyLandmark(x=0.0, y=0.0) for _ in range(468)]
    for pose_idx, pt in zip((1, 152, 33, 263, 61, 291), projected.reshape(-1, 2), strict=True):
        landmarks[pose_idx] = DummyLandmark(x=float(pt[0] / width), y=float(pt[1] / height))

    pitch, yaw, roll = _estimate_head_pose(landmarks, width=width, height=height)
    yaw_convenience = _estimate_head_yaw(landmarks, width=width, height=height)

    assert isinstance(pitch, float)
    assert isinstance(yaw, float)
    assert isinstance(roll, float)
    assert yaw == pytest.approx(yaw_convenience, abs=1e-3)
    assert pitch == pytest.approx(0.0, abs=1.0)
