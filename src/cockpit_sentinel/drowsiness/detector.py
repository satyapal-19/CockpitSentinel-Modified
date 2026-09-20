"""MediaPipe-based fatigue and attention signal extraction."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import dist
from pathlib import Path
from typing import Any, Protocol

import cv2
import mediapipe as mp
import numpy as np
import yaml
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from cockpit_sentinel.domain import DriverSignals
from cockpit_sentinel.drowsiness.occlusion import EyeOcclusionDetector
from cockpit_sentinel.drowsiness.yawn_speech import YawnSpeechDiscriminator

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MOUTH = (61, 13, 14, 291)
HEAD_POSE = (1, 152, 33, 263, 61, 291)
HEAD_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)


class Landmark(Protocol):
    """Minimal landmark shape returned by MediaPipe."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class DrowsinessConfig:
    """Thresholds for converting face geometry into driver signals."""

    eye_aspect_ratio_threshold: float = 0.22
    mouth_aspect_ratio_threshold: float = 0.60
    head_yaw_threshold_degrees: float = 30.0
    minimum_consecutive_frames: int = 3
    perclos_window_seconds: float = 30.0
    perclos_fatigue_threshold: float = 0.20
    microsleep_threshold_seconds: float = 1.5
    yawn_min_duration_seconds: float = 1.5
    speech_mar_threshold: float = 0.38
    speech_oscillation_hz: float = 1.5
    head_pitch_threshold_degrees: float = 15.0
    eye_contrast_threshold: float = 12.0

    def __post_init__(self) -> None:
        if not 0 < self.eye_aspect_ratio_threshold < 1:
            raise ValueError("eye_aspect_ratio_threshold must be between 0 and 1.")
        if self.mouth_aspect_ratio_threshold <= 0:
            raise ValueError("mouth_aspect_ratio_threshold must be positive.")
        if not 0 < self.head_yaw_threshold_degrees <= 90:
            raise ValueError("head_yaw_threshold_degrees must be between 0 and 90.")
        if self.minimum_consecutive_frames < 1:
            raise ValueError("minimum_consecutive_frames must be at least 1.")
        if self.perclos_window_seconds <= 0:
            raise ValueError("perclos_window_seconds must be positive.")
        if not 0 < self.perclos_fatigue_threshold <= 1:
            raise ValueError("perclos_fatigue_threshold must be between 0 and 1.")
        if self.microsleep_threshold_seconds <= 0:
            raise ValueError("microsleep_threshold_seconds must be positive.")
        if self.yawn_min_duration_seconds <= 0:
            raise ValueError("yawn_min_duration_seconds must be positive.")
        if self.speech_mar_threshold <= 0:
            raise ValueError("speech_mar_threshold must be positive.")
        if self.speech_oscillation_hz <= 0:
            raise ValueError("speech_oscillation_hz must be positive.")
        if not 0 < self.head_pitch_threshold_degrees <= 90:
            raise ValueError("head_pitch_threshold_degrees must be between 0 and 90.")
        if self.eye_contrast_threshold <= 0:
            raise ValueError("eye_contrast_threshold must be positive.")


@dataclass(frozen=True, slots=True)
class DrowsinessAnalysis:
    """Raw geometry and stabilized signals for one video frame."""

    signals: DriverSignals
    face_detected: bool
    eye_aspect_ratio: float | None = None
    mouth_aspect_ratio: float | None = None
    head_yaw_degrees: float | None = None
    head_pitch_degrees: float | None = None
    head_roll_degrees: float | None = None
    perclos: float | None = None
    closure_duration_seconds: float = 0.0
    microsleep_detected: bool = False
    eye_occluded: bool = False
    talking: bool = False
    head_nodding: bool = False


def load_drowsiness_config(path: Path) -> DrowsinessConfig:
    """Load detector thresholds from a YAML configuration file."""

    with path.open(encoding="utf-8") as config_file:
        raw_config: Any = yaml.safe_load(config_file)
    if not isinstance(raw_config, dict):
        raise ValueError("Drowsiness configuration must be a YAML mapping.")

    expected_keys = set(DrowsinessConfig.__dataclass_fields__)
    if set(raw_config) != expected_keys:
        raise ValueError("Drowsiness configuration must define every threshold.")

    return DrowsinessConfig(**raw_config)


def eye_aspect_ratio(points: Sequence[tuple[float, float]]) -> float:
    """Return the standard six-point eye aspect ratio."""

    if len(points) != 6:
        raise ValueError("Eye aspect ratio requires exactly six points.")
    vertical = dist(points[1], points[5]) + dist(points[2], points[4])
    horizontal = 2 * dist(points[0], points[3])
    return vertical / horizontal if horizontal else 0.0


def mouth_aspect_ratio(points: Sequence[tuple[float, float]]) -> float:
    """Return vertical mouth opening relative to mouth width."""

    if len(points) != 4:
        raise ValueError("Mouth aspect ratio requires exactly four points.")
    horizontal = dist(points[0], points[3])
    vertical = dist(points[1], points[2])
    return vertical / horizontal if horizontal else 0.0


class SignalStabilizer:
    """Require a signal to persist for consecutive frames before alerting."""

    def __init__(self, minimum_consecutive_frames: int) -> None:
        if minimum_consecutive_frames < 1:
            raise ValueError("minimum_consecutive_frames must be at least 1.")
        self._minimum_consecutive_frames = minimum_consecutive_frames
        self._counts = {name: 0 for name in DriverSignals.__dataclass_fields__}

    def update(self, raw_signals: DriverSignals) -> DriverSignals:
        stabilized: dict[str, bool] = {}
        for name in self._counts:
            active = getattr(raw_signals, name)
            self._counts[name] = self._counts[name] + 1 if active else 0
            stabilized[name] = self._counts[name] >= self._minimum_consecutive_frames
        return DriverSignals(**stabilized)

    def reset(self) -> None:
        for name in self._counts:
            self._counts[name] = 0


class PERCLOSTracker:
    """Calculate percentage of eye closure (PERCLOS) and detect micro-sleep events."""

    def __init__(
        self,
        window_seconds: float = 30.0,
        fatigue_threshold: float = 0.20,
        microsleep_seconds: float = 1.5,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")
        if not 0 < fatigue_threshold <= 1:
            raise ValueError("fatigue_threshold must be between 0 and 1.")
        if microsleep_seconds <= 0:
            raise ValueError("microsleep_seconds must be positive.")

        self.window_seconds = window_seconds
        self.fatigue_threshold = fatigue_threshold
        self.microsleep_seconds = microsleep_seconds

        self._history: deque[tuple[float, bool]] = deque()
        self._closure_start_time: float | None = None

    def update(
        self, eyes_closed: bool, timestamp: float | None = None
    ) -> tuple[float, float, bool, bool]:
        """Record eye state and return (perclos, closure_duration, is_fatigued, is_microsleep)."""
        now = time.monotonic() if timestamp is None else timestamp

        if eyes_closed:
            if self._closure_start_time is None:
                self._closure_start_time = now
            closure_duration = now - self._closure_start_time
        else:
            self._closure_start_time = None
            closure_duration = 0.0

        is_microsleep = closure_duration >= self.microsleep_seconds

        self._history.append((now, eyes_closed))

        cutoff = now - self.window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        closed_count = sum(1 for _, closed in self._history if closed)
        perclos = (closed_count / len(self._history)) if self._history else 0.0
        is_fatigued = perclos >= self.fatigue_threshold

        return perclos, closure_duration, is_fatigued, is_microsleep

    def reset(self) -> None:
        """Clear temporal window and closure timing state."""
        self._history.clear()
        self._closure_start_time = None


class HeadNoddingTracker:
    """Detects downward head droop and nodding behavior indicative of fatigue."""

    def __init__(
        self,
        *,
        pitch_threshold_degrees: float = 15.0,
        droop_duration_seconds: float = 1.0,
        window_seconds: float = 4.0,
    ) -> None:
        if pitch_threshold_degrees <= 0:
            raise ValueError("pitch_threshold_degrees must be positive.")
        if droop_duration_seconds <= 0:
            raise ValueError("droop_duration_seconds must be positive.")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")

        self.pitch_threshold_degrees = pitch_threshold_degrees
        self.droop_duration_seconds = droop_duration_seconds
        self.window_seconds = window_seconds
        self._droop_start_time: float | None = None
        self._history: deque[tuple[float, float]] = deque()

    def update(self, pitch: float, timestamp: float | None = None) -> tuple[bool, float]:
        """Record head pitch and return (is_nodding, sustained_droop_duration)."""
        now = time.monotonic() if timestamp is None else timestamp
        self._history.append((now, pitch))

        cutoff = now - self.window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        # Continuous downward droop
        if pitch >= self.pitch_threshold_degrees:
            if self._droop_start_time is None:
                self._droop_start_time = now
            droop_duration = now - self._droop_start_time
        else:
            self._droop_start_time = None
            droop_duration = 0.0

        is_sustained = droop_duration >= self.droop_duration_seconds

        # Cyclic nod detection: repeated downward pitch excursions with recovery
        nod_count = 0
        in_nod = False
        for _, p in self._history:
            if p >= self.pitch_threshold_degrees:
                if not in_nod:
                    nod_count += 1
                    in_nod = True
            elif p < (self.pitch_threshold_degrees - 5.0):
                in_nod = False

        is_nodding = is_sustained or (nod_count >= 2)
        return is_nodding, droop_duration

    def reset(self) -> None:
        """Clear droop timing and history state."""
        self._droop_start_time = None
        self._history.clear()


class DrowsinessDetector:
    """Extract eye, mouth, and head-direction signals from BGR OpenCV frames."""

    def __init__(
        self,
        model_path: Path,
        config: DrowsinessConfig | None = None,
        delegate: str | None = None,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Face landmark model was not found: {model_path}")

        self.config = config or DrowsinessConfig()
        self.delegate_name = delegate or "cpu"
        self._landmarker = self._create_landmarker(model_path, self.delegate_name)
        self._stabilizer = SignalStabilizer(self.config.minimum_consecutive_frames)
        self._perclos_tracker = PERCLOSTracker(
            window_seconds=self.config.perclos_window_seconds,
            fatigue_threshold=self.config.perclos_fatigue_threshold,
            microsleep_seconds=self.config.microsleep_threshold_seconds,
        )
        self._yawn_speech = YawnSpeechDiscriminator(
            yawn_mar_threshold=self.config.mouth_aspect_ratio_threshold,
            yawn_min_duration_seconds=self.config.yawn_min_duration_seconds,
            speech_mar_threshold=self.config.speech_mar_threshold,
            speech_oscillation_hz=self.config.speech_oscillation_hz,
        )
        self._occlusion_detector = EyeOcclusionDetector(
            contrast_threshold=self.config.eye_contrast_threshold,
        )
        self._nodding_tracker = HeadNoddingTracker(
            pitch_threshold_degrees=self.config.head_pitch_threshold_degrees,
        )

    @staticmethod
    def _create_landmarker(model_path: Path, delegate: str) -> vision.FaceLandmarker:
        delegate_choice = delegate.strip().lower()
        if delegate_choice in ("gpu", "cuda"):
            try:
                options = vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(model_path),
                        delegate=BaseOptions.Delegate.GPU,
                    ),
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                )
                return vision.FaceLandmarker.create_from_options(options)
            except Exception:
                pass

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path),
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> DrowsinessDetector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def analyze(self, frame: np.ndarray, timestamp: float | None = None) -> DrowsinessAnalysis:
        """Analyze one BGR frame and return stabilized fatigue/attention signals."""

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a three-channel BGR frame.")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(image)
        if not result.face_landmarks:
            self._stabilizer.reset()
            self._perclos_tracker.reset()
            self._yawn_speech.reset()
            self._occlusion_detector.reset()
            self._nodding_tracker.reset()
            return DrowsinessAnalysis(signals=DriverSignals(), face_detected=False)

        landmarks = result.face_landmarks[0]
        height, width = frame.shape[:2]
        left_eye_pts = _pixel_points(landmarks, LEFT_EYE, width, height)
        right_eye_pts = _pixel_points(landmarks, RIGHT_EYE, width, height)
        left_ear = eye_aspect_ratio(left_eye_pts)
        right_ear = eye_aspect_ratio(right_eye_pts)
        ear = (left_ear + right_ear) / 2
        mar = mouth_aspect_ratio(_pixel_points(landmarks, MOUTH, width, height))
        pitch, yaw, roll = _estimate_head_pose(landmarks, width, height)

        # 1. Sunglasses / Eye Occlusion check
        is_occluded, _, _ = self._occlusion_detector.update(frame, left_eye_pts, right_eye_pts)

        # 2. Speech vs. Yawn Discrimination
        is_yawn, is_talking, _ = self._yawn_speech.update(mar, ear=ear, timestamp=timestamp)

        # 3. Head Nodding / Micro-droop check
        is_nodding, _ = self._nodding_tracker.update(pitch, timestamp=timestamp)

        # 4. Eye closure & PERCLOS
        if is_occluded:
            raw_eyes_closed = False
            perclos, closure_duration, is_fatigued, is_microsleep = 0.0, 0.0, False, False
            self._perclos_tracker.reset()
        else:
            raw_eyes_closed = ear < self.config.eye_aspect_ratio_threshold
            perclos, closure_duration, is_fatigued, is_microsleep = self._perclos_tracker.update(
                raw_eyes_closed, timestamp=timestamp
            )

        raw_signals = DriverSignals(
            eyes_closed=raw_eyes_closed,
            yawning=is_yawn,
            looking_away=abs(yaw) > self.config.head_yaw_threshold_degrees,
            perclos_fatigue=is_fatigued,
            microsleep_detected=is_microsleep,
            talking=is_talking,
            head_nodding=is_nodding,
            eye_occluded=is_occluded,
        )

        stabilized = self._stabilizer.update(raw_signals)
        if is_microsleep:
            stabilized = replace(stabilized, microsleep_detected=True)
        if is_occluded:
            stabilized = replace(stabilized, eye_occluded=True)
        if is_nodding:
            stabilized = replace(stabilized, head_nodding=True)

        return DrowsinessAnalysis(
            signals=stabilized,
            face_detected=True,
            eye_aspect_ratio=ear if not is_occluded else None,
            mouth_aspect_ratio=mar,
            head_yaw_degrees=yaw,
            head_pitch_degrees=pitch,
            head_roll_degrees=roll,
            perclos=perclos if not is_occluded else None,
            closure_duration_seconds=closure_duration,
            microsleep_detected=is_microsleep,
            eye_occluded=is_occluded,
            talking=is_talking,
            head_nodding=is_nodding,
        )


def _pixel_points(
    landmarks: Sequence[Landmark], indices: Sequence[int], width: int, height: int
) -> list[tuple[float, float]]:
    return [(landmarks[index].x * width, landmarks[index].y * height) for index in indices]


def _estimate_head_pose(
    landmarks: Sequence[Landmark], width: int, height: int
) -> tuple[float, float, float]:
    """Estimate 3D head orientation returning (pitch, yaw, roll) in degrees."""
    image_points = np.array(_pixel_points(landmarks, HEAD_POSE, width, height), dtype=np.float64)
    focal_length = float(width)
    camera_matrix = np.array(
        [[focal_length, 0.0, width / 2], [0.0, focal_length, height / 2], [0.0, 0.0, 1.0]]
    )
    try:
        success, rotation_vector, _ = cv2.solvePnP(
            HEAD_MODEL_POINTS,
            image_points,
            camera_matrix,
            np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return 0.0, 0.0, 0.0

    if not success:
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    projection_matrix = np.hstack((rotation_matrix, np.zeros((3, 1))))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
    pitch = float(euler_angles[0, 0])
    yaw = float(euler_angles[1, 0])
    roll = float(euler_angles[2, 0])
    return pitch, yaw, roll


def _estimate_head_yaw(landmarks: Sequence[Landmark], width: int, height: int) -> float:
    """Convenience helper returning just yaw in degrees."""
    _, yaw, _ = _estimate_head_pose(landmarks, width, height)
    return yaw
