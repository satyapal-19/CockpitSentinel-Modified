"""Driver profile management and facial biometric recognition subsystem."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from math import dist
from pathlib import Path
from typing import Any

from cockpit_sentinel.drowsiness.detector import Landmark

logger = logging.getLogger(__name__)

# Key structural facial bone landmark indices in MediaPipe FaceMesh (478 landmarks)
# These points are anatomically anchored and remain stable across facial expressions.
GLABELLA = 10
NOSE_BRIDGE = 168
NOSE_TIP = 1
SUBNASALE = 2
CHIN_TIP = 152
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
LEFT_TRAGUS = 234
RIGHT_TRAGUS = 454
LEFT_GONION = 172
RIGHT_GONION = 397
LEFT_CHEEK = 116
RIGHT_CHEEK = 345


@dataclass
class DriverProfile:
    """Individual driver calibration settings and biometric identity."""

    driver_id: str
    name: str
    resting_ear: float = 0.30
    ear_threshold: float = 0.22
    resting_mar: float = 0.18
    mar_threshold: float = 0.60
    head_pitch_threshold: float = 15.0
    perclos_window_seconds: float = 30.0
    biometric_signature: list[float] | None = None
    calibrated: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DriverProfile:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def extract_biometric_signature(
    landmarks: Sequence[Landmark], width: int, height: int
) -> list[float] | None:
    """Extract scale-invariant geometric bone distance signature vector (20 features).

    Returns None if landmarks are incomplete or scale is invalid.
    """
    if len(landmarks) < 468:
        return None

    def pt(idx: int) -> tuple[float, float]:
        return (landmarks[idx].x * width, landmarks[idx].y * height)

    # Reference scale: Inter-ocular distance between outer eye corners
    scale = dist(pt(LEFT_EYE_OUTER), pt(RIGHT_EYE_OUTER))
    if scale < 10.0:
        return None

    # Calculate pairwise anatomical bone distances normalized by inter-ocular scale
    features = [
        dist(pt(GLABELLA), pt(CHIN_TIP)) / scale,  # Total face height
        dist(pt(LEFT_TRAGUS), pt(RIGHT_TRAGUS)) / scale,  # Face width
        dist(pt(LEFT_GONION), pt(RIGHT_GONION)) / scale,  # Jaw width
        dist(pt(LEFT_EYE_INNER), pt(RIGHT_EYE_INNER)) / scale,  # Inner eye separation
        dist(pt(NOSE_BRIDGE), pt(NOSE_TIP)) / scale,  # Nose length
        dist(pt(SUBNASALE), pt(CHIN_TIP)) / scale,  # Lower face height
        dist(pt(GLABELLA), pt(NOSE_BRIDGE)) / scale,  # Upper face height
        dist(pt(LEFT_EYE_OUTER), pt(CHIN_TIP)) / scale,  # Left eye to chin
        dist(pt(RIGHT_EYE_OUTER), pt(CHIN_TIP)) / scale,  # Right eye to chin
        dist(pt(LEFT_EYE_OUTER), pt(NOSE_TIP)) / scale,  # Left eye to nose
        dist(pt(RIGHT_EYE_OUTER), pt(NOSE_TIP)) / scale,  # Right eye to nose
        dist(pt(LEFT_TRAGUS), pt(CHIN_TIP)) / scale,  # Left ear to chin
        dist(pt(RIGHT_TRAGUS), pt(CHIN_TIP)) / scale,  # Right ear to chin
        dist(pt(LEFT_GONION), pt(CHIN_TIP)) / scale,  # Left jaw to chin
        dist(pt(RIGHT_GONION), pt(CHIN_TIP)) / scale,  # Right jaw to chin
        dist(pt(GLABELLA), pt(LEFT_TRAGUS)) / scale,  # Forehead to left ear
        dist(pt(GLABELLA), pt(RIGHT_TRAGUS)) / scale,  # Forehead to right ear
        dist(pt(LEFT_CHEEK), pt(RIGHT_CHEEK)) / scale,  # Midface cheek width
        dist(pt(LEFT_CHEEK), pt(CHIN_TIP)) / scale,  # Left cheek to chin
        dist(pt(RIGHT_CHEEK), pt(CHIN_TIP)) / scale,  # Right cheek to chin
    ]
    return features


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Calculate cosine similarity between two feature vectors."""
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class DriverRecognizer:
    """Recognizes known driver profiles from real-time facial landmark geometry."""

    def __init__(self, match_threshold: float = 0.94) -> None:
        self.match_threshold = match_threshold

    def match(
        self,
        landmarks: Sequence[Landmark],
        width: int,
        height: int,
        profiles: Sequence[DriverProfile],
    ) -> tuple[DriverProfile | None, float]:
        """Match current face against stored profiles.

        Returns:
            (best_matching_profile, confidence)
        """
        signature = extract_biometric_signature(landmarks, width, height)
        if signature is None or not profiles:
            return None, 0.0

        best_profile: DriverProfile | None = None
        best_sim = 0.0

        for profile in profiles:
            if not profile.biometric_signature:
                continue
            sim = cosine_similarity(signature, profile.biometric_signature)
            if sim > best_sim:
                best_sim = sim
                best_profile = profile

        if best_sim >= self.match_threshold:
            return best_profile, best_sim
        return None, best_sim


class ProfileManager:
    """Manages persistent driver profile storage, auto-calibration, and retrieval."""

    def __init__(self, storage_path: Path | None = None) -> None:
        if storage_path is None:
            # Default to data/profiles.json inside project root
            base_dir = Path(__file__).resolve().parents[3]
            data_dir = base_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self.storage_path = data_dir / "profiles.json"
        else:
            self.storage_path = storage_path
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self._profiles: dict[str, DriverProfile] = {}
        self._active_driver_id: str = "driver_default"
        self._load()

    def _load(self) -> None:
        """Load profiles from JSON file or create initial seed profile."""
        if self.storage_path.exists():
            try:
                with self.storage_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                profiles_data = data.get("profiles", {})
                self._profiles = {
                    pid: DriverProfile.from_dict(pdata) for pid, pdata in profiles_data.items()
                }
                self._active_driver_id = data.get("active_driver_id", "driver_default")
            except Exception as e:
                logger.warning(
                    "Could not read profiles file (%s), reinitializing: %s",
                    self.storage_path,
                    e,
                )
                self._create_default_seed()
        else:
            self._create_default_seed()

    def _create_default_seed(self) -> None:
        """Seed default driver profile if none exists."""
        default_profile = DriverProfile(
            driver_id="driver_default",
            name="Default Driver",
            resting_ear=0.31,
            ear_threshold=0.22,
            resting_mar=0.18,
            mar_threshold=0.60,
            calibrated=False,
        )
        self._profiles = {default_profile.driver_id: default_profile}
        self._active_driver_id = default_profile.driver_id
        self.save()

    def save(self) -> None:
        """Persist profiles to disk."""
        data = {
            "active_driver_id": self._active_driver_id,
            "profiles": {pid: p.to_dict() for pid, p in self._profiles.items()},
        }
        with self.storage_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_all(self) -> list[DriverProfile]:
        return list(self._profiles.values())

    def get(self, driver_id: str) -> DriverProfile | None:
        return self._profiles.get(driver_id)

    def get_active_profile(self) -> DriverProfile:
        return self._profiles.get(self._active_driver_id) or next(iter(self._profiles.values()))

    def set_active_profile(self, driver_id: str) -> bool:
        if driver_id in self._profiles:
            self._active_driver_id = driver_id
            self.save()
            return True
        return False

    def create(
        self,
        name: str,
        resting_ear: float = 0.30,
        ear_threshold: float = 0.22,
        resting_mar: float = 0.18,
        mar_threshold: float = 0.60,
        signature: list[float] | None = None,
    ) -> DriverProfile:
        """Create and store a new driver profile."""
        import uuid

        driver_id = f"driver_{uuid.uuid4().hex[:8]}"
        profile = DriverProfile(
            driver_id=driver_id,
            name=name.strip(),
            resting_ear=resting_ear,
            ear_threshold=ear_threshold,
            resting_mar=resting_mar,
            mar_threshold=mar_threshold,
            biometric_signature=signature,
            calibrated=signature is not None,
        )
        self._profiles[driver_id] = profile
        self.save()
        return profile

    def update_thresholds(
        self,
        driver_id: str,
        ear_threshold: float | None = None,
        mar_threshold: float | None = None,
        head_pitch_threshold: float | None = None,
    ) -> bool:
        profile = self.get(driver_id)
        if not profile:
            return False

        if ear_threshold is not None:
            profile.ear_threshold = ear_threshold
        if mar_threshold is not None:
            profile.mar_threshold = mar_threshold
        if head_pitch_threshold is not None:
            profile.head_pitch_threshold = head_pitch_threshold

        self.save()
        return True

    def auto_calibrate(
        self,
        driver_id: str,
        ear_samples: Sequence[float],
        mar_samples: Sequence[float],
        signature_samples: Sequence[list[float]] | None = None,
    ) -> DriverProfile | None:
        """Compute mean baseline geometry and calibrate personalized thresholds.

        Formula:
            ear_threshold = mean(EAR) * 0.70  (70% of resting state)
            mar_threshold = max(0.55, mean(MAR) * 2.8)  (detects yawning aperture)
        """
        profile = self.get(driver_id)
        if not profile or not ear_samples or not mar_samples:
            return None

        mean_ear = sum(ear_samples) / len(ear_samples)
        mean_mar = sum(mar_samples) / len(mar_samples)

        profile.resting_ear = round(mean_ear, 3)
        profile.resting_mar = round(mean_mar, 3)
        profile.ear_threshold = round(max(0.14, min(0.28, mean_ear * 0.70)), 2)
        profile.mar_threshold = round(max(0.52, min(0.75, mean_mar * 2.80)), 2)
        profile.calibrated = True

        if signature_samples:
            # Average biometric signatures across samples
            n = len(signature_samples)
            dim = len(signature_samples[0])
            avg_sig = [sum(sig[i] for sig in signature_samples) / n for i in range(dim)]
            profile.biometric_signature = avg_sig

        self.save()
        return profile
