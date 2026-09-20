"""Unit tests for EyeOcclusionDetector (sunglasses and eye occlusion detection)."""

import numpy as np
import pytest

from cockpit_sentinel.drowsiness.occlusion import EyeOcclusionDetector


def test_eye_occlusion_detector_validates_parameters():
    with pytest.raises(ValueError, match="contrast_threshold"):
        EyeOcclusionDetector(contrast_threshold=0)
    with pytest.raises(ValueError, match="darkness_threshold"):
        EyeOcclusionDetector(darkness_threshold=-1)
    with pytest.raises(ValueError, match="hysteresis_frames"):
        EyeOcclusionDetector(hysteresis_frames=0)


def test_natural_eye_contrast_is_not_occluded():
    detector = EyeOcclusionDetector(contrast_threshold=12.0, hysteresis_frames=2)

    # Synthetic natural eye patch: high contrast between sclera (~230) and iris (~30)
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 150
    # Add high contrast pattern in eye regions
    frame[30:50, 20:40] = 30
    frame[30:50, 60:80] = 230

    left_points = [(20, 30), (40, 30), (40, 50), (20, 50)]
    right_points = [(60, 30), (80, 30), (80, 50), (60, 50)]

    for _ in range(3):
        is_occluded, left_std, right_std = detector.update(frame, left_points, right_points)

    assert not is_occluded
    assert left_std >= 0.0 or right_std >= 0.0


def test_uniform_dark_lenses_detected_as_occluded():
    detector = EyeOcclusionDetector(
        contrast_threshold=12.0, darkness_threshold=28.0, hysteresis_frames=3
    )

    # Synthetic sunglasses: dark uniform lens region (~15 intensity, near-zero std)
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 15

    left_points = [(20, 30), (40, 30), (40, 50), (20, 50)]
    right_points = [(60, 30), (80, 30), (80, 50), (60, 50)]

    # Frame 1: not yet past hysteresis
    is_occ, _, _ = detector.update(frame, left_points, right_points)
    assert not is_occ

    # Frame 2
    detector.update(frame, left_points, right_points)

    # Frame 3: reaches hysteresis threshold
    is_occ, left_std, right_std = detector.update(frame, left_points, right_points)
    assert is_occ
    assert left_std < 12.0
    assert right_std < 12.0


def test_detector_reset_clears_hysteresis():
    detector = EyeOcclusionDetector(hysteresis_frames=2)
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 15
    pts = [(20, 30), (40, 30), (40, 50), (20, 50)]

    detector.update(frame, pts, pts)
    detector.update(frame, pts, pts)
    is_occ, _, _ = detector.update(frame, pts, pts)
    assert is_occ

    detector.reset()
    # Immediate next frame should not be occluded until hysteresis re-accumulates
    is_occ, _, _ = detector.update(frame, pts, pts)
    assert not is_occ
