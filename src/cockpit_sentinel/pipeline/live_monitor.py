"""Interactive webcam and video runner for the fatigue monitoring pipeline."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from cockpit_sentinel.alerts.audio import AudioManager
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
from cockpit_sentinel.utils.device import resolve_device
from cockpit_sentinel.utils.paths import (
    find_project_root,
    resolve_config_path,
)
from cockpit_sentinel.utils.paths import (
    resolve_model_path as _resolve_model_path,
)

REPO_ROOT = find_project_root()
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
        audio_manager: AudioManager | None = None,
    ) -> None:
        self._detector = detector
        self._policy = policy
        self._distraction_detector = distraction_detector
        self._audio_manager = audio_manager

    def process(self, frame: np.ndarray) -> MonitorFrame:
        analysis = self._detector.analyze(frame)
        distraction = None
        if self._distraction_detector is not None:
            distraction = self._distraction_detector.analyze(frame)
        distraction_signals = distraction.signals if distraction else DriverSignals()
        signals = merge_signals(analysis.signals, distraction_signals)
        assessment = self._policy.assess(signals)

        if self._audio_manager is not None:
            first_reason = assessment.reasons[0] if assessment.reasons else None
            self._audio_manager.trigger(assessment.level, reason=first_reason)

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
        perclos_fatigue=drowsiness.perclos_fatigue,
        microsleep_detected=drowsiness.microsleep_detected,
        talking=drowsiness.talking,
        head_nodding=drowsiness.head_nodding,
        eye_occluded=drowsiness.eye_occluded,
    )


def draw_monitor_overlay(
    frame: np.ndarray, analysis: DrowsinessAnalysis, assessment: RiskAssessment
) -> np.ndarray:
    """Draw readable status, active reasons, and measurements onto a frame copy."""

    image = frame.copy()
    color = RISK_COLORS[assessment.level]
    cv2.rectangle(image, (0, 0), (image.shape[1], 118), (25, 25, 25), thickness=-1)
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

    # Occlusion / Sunglasses mode badge
    if analysis.eye_occluded:
        cv2.putText(
            image,
            "[SUNGLASSES MODE]",
            (image.shape[1] - 220, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

    status = ", ".join(assessment.reasons) if assessment.reasons else "attentive"
    if analysis.talking and not assessment.reasons:
        status = "attentive (talking)"
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
        if analysis.eye_occluded:
            ear_str = "EAR: occluded"
            perclos_str = "PERCLOS: --"
        else:
            ear_str = (
                f"EAR: {analysis.eye_aspect_ratio:.2f}"
                if analysis.eye_aspect_ratio is not None
                else "EAR: --"
            )
            perclos_str = (
                f"PERCLOS: {analysis.perclos * 100:.1f}%"
                if analysis.perclos is not None
                else "PERCLOS: --"
            )

        mar_str = (
            f"MAR: {analysis.mouth_aspect_ratio:.2f}"
            if analysis.mouth_aspect_ratio is not None
            else "MAR: --"
        )
        yaw_str = (
            f"Yaw: {analysis.head_yaw_degrees:.1f}"
            if analysis.head_yaw_degrees is not None
            else "Yaw: --"
        )
        pitch_str = (
            f"Pitch: {analysis.head_pitch_degrees:.1f}"
            if analysis.head_pitch_degrees is not None
            else "Pitch: --"
        )

        metrics = [ear_str, mar_str, yaw_str, pitch_str, perclos_str]
        cv2.putText(
            image,
            " | ".join(metrics),
            (16, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            image,
            "Face not detected",
            (16, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

    # Urgent emergency banners at bottom of display
    h, w = image.shape[:2]
    if analysis.microsleep_detected:
        cv2.rectangle(image, (0, h - 45), (w, h), (0, 0, 220), thickness=-1)
        cv2.putText(
            image,
            "!!! MICROSLEEP DETECTED - WAKE UP !!!",
            (max(16, w // 2 - 230), h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    elif analysis.head_nodding:
        banner_text = (
            "!!! HEAD NODDING (SUNGLASSES FALLBACK) - WAKE UP !!!"
            if analysis.eye_occluded
            else "!!! HEAD DROOP / NODDING DETECTED - STAY ALERT !!!"
        )
        cv2.rectangle(image, (0, h - 45), (w, h), (0, 69, 255), thickness=-1)
        cv2.putText(
            image,
            banner_text,
            (max(16, w // 2 - 280), h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return image


def run_live_monitor(
    source: int | str,
    detector: FrameAnalyzer,
    policy: AlertPolicy,
    distraction_detector: DistractionAnalyzer | None = None,
    audio_manager: AudioManager | None = None,
) -> None:
    """Open a webcam or video file and show processed frames until Q or Esc is pressed."""

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    monitor = LiveMonitor(detector, policy, distraction_detector, audio_manager=audio_manager)
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


def resolve_model_path(filename: str, models_root: Path | None = None) -> Path:
    """Resolve a shared model path from the environment or local .env file."""
    return _resolve_model_path(filename, models_root=models_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the drowsiness webcam/video monitor.")
    parser.add_argument(
        "--source", default="0", help="Webcam index or video file path (default: 0)."
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("COCKPIT_DEVICE", "auto"),
        help="Inference compute device: 'auto', 'cpu', 'cuda', 'cuda:0', 'mps' (default: auto).",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable 3-tier acoustic alert cues (default: audio enabled).",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=None,
        help="Optional root directory containing pretrained models.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to MediaPipe face landmarker model (.task).",
    )
    parser.add_argument(
        "--phone-model-path",
        type=Path,
        default=None,
        help="Path to YOLO phone detection model (.pt).",
    )
    parser.add_argument(
        "--smoking-model-path",
        type=Path,
        default=None,
        help="Path to YOLOWorld smoking detection model (.pt).",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Directory containing YAML configuration files.",
    )
    parser.add_argument(
        "--drowsiness-config",
        type=Path,
        default=None,
        help="Path to drowsiness YAML configuration file.",
    )
    parser.add_argument(
        "--distraction-config",
        type=Path,
        default=None,
        help="Path to distraction YAML configuration file.",
    )
    parser.add_argument(
        "--alert-policy",
        type=Path,
        default=None,
        help="Path to alert policy YAML configuration file.",
    )
    args = parser.parse_args()

    model_path = args.model_path or resolve_model_path("face_landmarker.task", args.models_root)
    phone_model_path = args.phone_model_path or resolve_model_path("yolov8n.pt", args.models_root)
    smoking_model_path = args.smoking_model_path or resolve_model_path(
        "yolov8s-worldv2.pt", args.models_root
    )

    # Check for missing model files before loading to provide helpful instructions
    missing_models: list[str] = []
    for name, path in [
        ("Face Landmarker", model_path),
        ("Phone Detector", phone_model_path),
        ("Smoking Detector", smoking_model_path),
    ]:
        if not path.exists():
            missing_models.append(f"  - {name}: {path}")

    if missing_models:
        print("\n[ERROR] Required model file(s) not found:")
        for line in missing_models:
            print(line)
        print("\nPlease run the setup script to download models:")
        print("  python scripts/setup_environment.py")
        print(
            "Or specify custom model paths via CLI flags "
            "(--model-path, --phone-model-path, etc.).\n"
        )
        raise SystemExit(1)

    drowsiness_cfg_path = resolve_config_path(
        "drowsiness.yaml", custom_dir=args.config_dir, explicit_file=args.drowsiness_config
    )
    distraction_cfg_path = resolve_config_path(
        "distraction.yaml", custom_dir=args.config_dir, explicit_file=args.distraction_config
    )
    alerts_cfg_path = resolve_config_path(
        "alerts.yaml", custom_dir=args.config_dir, explicit_file=args.alert_policy
    )

    drowsiness_config = load_drowsiness_config(drowsiness_cfg_path)
    distraction_config = load_distraction_config(distraction_cfg_path)
    alert_policy = load_alert_policy(alerts_cfg_path)

    target_device = resolve_device(args.device)
    print(f"CockpitSentinel starting on device: {target_device}")
    audio_status = "disabled (--no-audio)" if args.no_audio else "active (3-tier escalation)"
    print(f"Audio alert system: {audio_status}")
    print("Press Q or Esc in the video window to stop.")

    try:
        with (
            AudioManager(enabled=not args.no_audio) as audio_mgr,
            DrowsinessDetector(model_path, drowsiness_config, delegate=target_device) as detector,
            DistractionDetector(
                phone_model_path,
                smoking_model_path,
                distraction_config,
                device=target_device,
            ) as distraction_detector,
        ):
            run_live_monitor(
                parse_source(args.source),
                detector,
                alert_policy,
                distraction_detector,
                audio_manager=audio_mgr,
            )
    except KeyboardInterrupt:
        print("\nMonitor interrupted by user. Exiting cleanly.")


if __name__ == "__main__":
    main()
