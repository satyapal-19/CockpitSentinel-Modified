"""Unit tests for 3D Head Pose and yaw estimation."""

from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from cockpit_sentinel.drowsiness.detector import (
    HEAD_MODEL_POINTS,
    _estimate_head_pose,
    _estimate_head_yaw,
)


@dataclass
class DummyLandmark:
    x: float
    y: float


def test_estimate_head_pose_returns_pitch_yaw_roll():
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
