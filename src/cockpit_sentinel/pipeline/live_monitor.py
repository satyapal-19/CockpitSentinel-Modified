"""Interactive webcam and video runner for the fatigue monitoring pipeline."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from dotenv import dotenv_values

from cockpit_sentinel.alerts.policy import AlertPolicy, load_alert_policy
from cockpit_sentinel.detection import (
    DistractionAnalysis,
    DistractionDetector,
    load_distraction_config,
)
from cockpit_sentinel.domain import DriverSignals, RiskAssessment, RiskLevel
from cockpit_sentinel.drowsiness import (
    DrowsinessAnalysis,
    DrowsinessDetector,
    load_drowsiness_config,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
WINDOW_TITLE = "CockpitSentinel - Drowsiness Monitor"
RISK_COLORS = {
    RiskLevel.SAFE: (61, 153, 112),
    RiskLevel.CAUTION: (33, 180, 241),
    RiskLevel.WARNING: (22, 137, 243),
    RiskLevel.CRITICAL: (49, 69, 220),
}


class FrameAnalyzer(Protocol):
    """Detector interface used by the live monitoring loop."""

    def analyze(self, frame: np.ndarray) -> DrowsinessAnalysis: ...


class DistractionAnalyzer(Protocol):
    """Object detector interface used by the live monitoring loop."""

    def analyze(self, frame: np.ndarray) -> DistractionAnalysis: ...


@dataclass(frozen=True, slots=True)
class MonitorFrame:
    """One processed frame with detector output, risk result, and display image."""

    analysis: DrowsinessAnalysis
    distraction: DistractionAnalysis | None
    assessment: RiskAssessment
    image: np.ndarray


class LiveMonitor:
    """Combine face analysis and risk scoring for displayable video frames."""

    def __init__(
        self,
        detector: FrameAnalyzer,
        policy: AlertPolicy,
        distraction_detector: DistractionAnalyzer | None = None,
    ) -> None:
        self._detector = detector
        self._policy = policy
        self._distraction_detector = distraction_detector

    def process(self, frame: np.ndarray) -> MonitorFrame:
        analysis = self._detector.analyze(frame)
        distraction = None
        if self._distraction_detector is not None:
            distraction = self._distraction_detector.analyze(frame)
        distraction_signals = distraction.signals if distraction else DriverSignals()
        signals = merge_signals(analysis.signals, distraction_signals)
        assessment = self._policy.assess(signals)
        image = draw_monitor_overlay(frame, analysis, assessment)
        return MonitorFrame(
            analysis=analysis,
            distraction=distraction,
            assessment=assessment,
            image=image,
        )


def merge_signals(drowsiness: DriverSignals, distraction: DriverSignals) -> DriverSignals:
    """Preserve fatigue signals while adding object-detection distraction signals."""

    return DriverSignals(
        eyes_closed=drowsiness.eyes_closed,
        yawning=drowsiness.yawning,
        looking_away=drowsiness.looking_away,
        phone_detected=distraction.phone_detected,
        smoking_detected=distraction.smoking_detected,
    )


def draw_monitor_overlay(
    frame: np.ndarray, analysis: DrowsinessAnalysis, assessment: RiskAssessment
) -> np.ndarray:
    """Draw readable status, active reasons, and measurements onto a frame copy."""

    image = frame.copy()
    color = RISK_COLORS[assessment.level]
    cv2.rectangle(image, (0, 0), (image.shape[1], 112), (25, 25, 25), thickness=-1)
    cv2.putText(
        image,
        f"Risk: {assessment.level.upper()} ({assessment.score})",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )
    status = ", ".join(assessment.reasons) if assessment.reasons else "attentive"
    cv2.putText(
        image,
        f"Status: {status}",
        (16, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    if analysis.face_detected:
        metrics = [
            f"EAR: {analysis.eye_aspect_ratio:.2f}",
            f"MAR: {analysis.mouth_aspect_ratio:.2f}",
            f"Yaw: {analysis.head_yaw_degrees:.1f} deg",
        ]
        cv2.putText(
            image,
            " | ".join(metrics),
            (16, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            image,
            "Face not detected",
            (16, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
    return image


def run_live_monitor(
    source: int | str,
    detector: FrameAnalyzer,
    policy: AlertPolicy,
    distraction_detector: DistractionAnalyzer | None = None,
) -> None:
    """Open a webcam or video file and show processed frames until Q or Esc is pressed."""

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    monitor = LiveMonitor(detector, policy, distraction_detector)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            processed = monitor.process(frame)
            cv2.imshow(WINDOW_TITLE, processed.image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def parse_source(value: str) -> int | str:
    """Treat numeric inputs as webcam indexes and all other inputs as file paths."""

    return int(value) if value.isdigit() else value


def resolve_model_path(filename: str) -> Path:
    """Resolve a shared model path from the environment or local .env file."""

    env_values = dotenv_values(REPO_ROOT / ".env")
    models_root = (
        os.environ.get("COCKPIT_MODELS_ROOT")
        or env_values.get("COCKPIT_MODELS_ROOT")
        or "D:/CockpitSentinel/models"
    )
    return Path(models_root) / "pretrained" / filename


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the drowsiness webcam/video monitor.")
    parser.add_argument(
        "--source", default="0", help="Webcam index or video file path (default: 0)."
    )
    parser.add_argument(
        "--model-path", type=Path, default=resolve_model_path("face_landmarker.task")
    )
    parser.add_argument("--phone-model-path", type=Path, default=resolve_model_path("yolov8n.pt"))
    parser.add_argument(
        "--smoking-model-path", type=Path, default=resolve_model_path("yolov8s-worldv2.pt")
    )
    args = parser.parse_args()

    drowsiness_config = load_drowsiness_config(REPO_ROOT / "configs" / "drowsiness.yaml")
    distraction_config = load_distraction_config(REPO_ROOT / "configs" / "distraction.yaml")
    alert_policy = load_alert_policy(REPO_ROOT / "configs" / "alerts.yaml")
    print("Press Q or Esc to stop the monitor.")
    with DrowsinessDetector(args.model_path, drowsiness_config) as detector:
        distraction_detector = DistractionDetector(
            args.phone_model_path,
            args.smoking_model_path,
            distraction_config,
        )
        run_live_monitor(parse_source(args.source), detector, alert_policy, distraction_detector)


if __name__ == "__main__":
    main()
