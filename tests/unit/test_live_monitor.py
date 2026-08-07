"""Tests for the live monitoring frame composition layer."""

import numpy as np

from cockpit_sentinel.alerts.policy import DEFAULT_POLICY
from cockpit_sentinel.detection import DistractionAnalysis
from cockpit_sentinel.domain import DriverSignals, RiskLevel
from cockpit_sentinel.drowsiness import DrowsinessAnalysis
from cockpit_sentinel.pipeline.live_monitor import LiveMonitor, parse_source


class StubDetector:
    """Return fixed detector output without requiring a camera or model asset."""

    def analyze(self, frame: np.ndarray) -> DrowsinessAnalysis:
        del frame
        return DrowsinessAnalysis(
            signals=DriverSignals(yawning=True),
            face_detected=True,
            eye_aspect_ratio=0.31,
            mouth_aspect_ratio=0.72,
            head_yaw_degrees=4.0,
        )


class StubDistractionDetector:
    def analyze(self, frame: np.ndarray) -> DistractionAnalysis:
        del frame
        return DistractionAnalysis(
            signals=DriverSignals(phone_detected=True), phone_confidence=0.91
        )


def test_live_monitor_composes_detector_and_risk_policy():
    frame = np.zeros((180, 320, 3), dtype=np.uint8)

    processed = LiveMonitor(StubDetector(), DEFAULT_POLICY).process(frame)

    assert processed.assessment.level is RiskLevel.CAUTION
    assert processed.analysis.signals.yawning
    assert processed.image.shape == frame.shape
    assert np.any(processed.image != frame)


def test_parse_source_handles_webcams_and_video_paths():
    assert parse_source("0") == 0
    assert parse_source("sample-video.mp4") == "sample-video.mp4"


def test_live_monitor_merges_drowsiness_and_distraction_signals():
    frame = np.zeros((180, 320, 3), dtype=np.uint8)

    processed = LiveMonitor(
        StubDetector(), DEFAULT_POLICY, StubDistractionDetector()
    ).process(frame)

    assert processed.distraction is not None
    assert processed.assessment.level is RiskLevel.CRITICAL
    assert "phone detected" in processed.assessment.reasons
