"""Object-detection signals for phone use and smoking-related distractions."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml
from ultralytics import YOLO, YOLOWorld

from cockpit_sentinel.domain import DriverSignals
from cockpit_sentinel.utils.device import resolve_device

PHONE_CLASSES: frozenset[str] = frozenset({"cell phone", "phone", "mobile phone", "telephone"})


class ObjectDetector(Protocol):
    """Minimal callable interface shared by Ultralytics detection models."""

    def __call__(self, frame: np.ndarray, **kwargs: Any) -> Sequence[Any]: ...


@dataclass(frozen=True, slots=True)
class DistractionConfig:
    """Confidence thresholds, persistence frames, and text labels for distraction detection."""

    phone_confidence_threshold: float = 0.28
    smoking_confidence_threshold: float = 0.18
    smoking_labels: tuple[str, ...] = (
        "cigarette",
        "smoking cigarette",
        "smoking",
        "vape",
        "vaping",
        "cigar",
        "e-cigarette",
        "electronic cigarette",
        "holding cigarette",
    )
    persistence_frames: int = 12

    def __post_init__(self) -> None:
        if not 0 < self.phone_confidence_threshold <= 1:
            raise ValueError("phone_confidence_threshold must be between 0 and 1.")
        if not 0 < self.smoking_confidence_threshold <= 1:
            raise ValueError("smoking_confidence_threshold must be between 0 and 1.")
        if self.persistence_frames < 0:
            raise ValueError("persistence_frames must be non-negative.")
        if not self.smoking_labels or not all(self.smoking_labels):
            raise ValueError("smoking_labels must contain at least one non-empty label.")


@dataclass(frozen=True, slots=True)
class DistractionAnalysis:
    """Per-frame object-detection output for distraction signals."""

    signals: DriverSignals
    phone_confidence: float | None = None
    smoking_confidence: float | None = None


def load_distraction_config(path: Path) -> DistractionConfig:
    """Load object-detection thresholds and prompts from YAML."""

    with path.open(encoding="utf-8") as config_file:
        raw_config: Any = yaml.safe_load(config_file)
    if not isinstance(raw_config, dict):
        raise ValueError("Distraction configuration must be a YAML mapping.")

    labels = raw_config.get("smoking_labels")
    if labels is not None and not isinstance(labels, list):
        raise ValueError("smoking_labels must be a YAML list.")
    smoking_tuple = (
        tuple(labels)
        if labels
        else DistractionConfig.__dataclass_fields__["smoking_labels"].default  # type: ignore[attr-defined]
    )

    return DistractionConfig(
        phone_confidence_threshold=raw_config.get("phone_confidence_threshold", 0.28),
        smoking_confidence_threshold=raw_config.get("smoking_confidence_threshold", 0.18),
        smoking_labels=smoking_tuple,
        persistence_frames=raw_config.get("persistence_frames", 12),
    )


class DistractionDetector:
    """Detect phone use and smoking-related objects from BGR video frames."""

    def __init__(
        self,
        phone_model_path: Path,
        smoking_model_path: Path,
        config: DistractionConfig | None = None,
        *,
        phone_model: ObjectDetector | None = None,
        smoking_model: ObjectDetector | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or DistractionConfig()
        self.device = resolve_device(device) if device is not None else None
        self._phone_model = phone_model or self._load_phone_model(phone_model_path, self.device)
        self._smoking_model = smoking_model or self._load_smoking_model(
            smoking_model_path, self.device
        )
        self._phone_persistence: int = 0
        self._smoking_persistence: int = 0
        self._last_phone_conf: float | None = None
        self._last_smoking_conf: float | None = None

    def reset(self) -> None:
        """Reset temporal persistence buffers."""
        self._phone_persistence = 0
        self._smoking_persistence = 0
        self._last_phone_conf = None
        self._last_smoking_conf = None

    def close(self) -> None:
        """Release any device or model resources."""
        pass

    def __enter__(self) -> DistractionDetector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def analyze(self, frame: np.ndarray) -> DistractionAnalysis:
        """Run object detectors with temporal persistence to prevent signal flicker."""

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a three-channel BGR frame.")

        predict_kwargs: dict[str, Any] = {"verbose": False}
        if self.device is not None:
            predict_kwargs["device"] = self.device

        detected_phone_conf = _highest_confidence(
            self._phone_model(frame, conf=self.config.phone_confidence_threshold, **predict_kwargs),
            PHONE_CLASSES,
        )
        detected_smoking_conf = _highest_confidence(
            self._smoking_model(
                frame, conf=self.config.smoking_confidence_threshold, **predict_kwargs
            ),
            set(self.config.smoking_labels),
        )

        # Temporal persistence: sustain detection for N frames to eliminate flickering
        if detected_phone_conf is not None:
            self._phone_persistence = self.config.persistence_frames
            self._last_phone_conf = detected_phone_conf
            phone_active = True
            phone_confidence = detected_phone_conf
        elif self._phone_persistence > 0:
            self._phone_persistence -= 1
            phone_active = True
            phone_confidence = self._last_phone_conf
        else:
            self._last_phone_conf = None
            phone_active = False
            phone_confidence = None

        if detected_smoking_conf is not None:
            self._smoking_persistence = self.config.persistence_frames
            self._last_smoking_conf = detected_smoking_conf
            smoking_active = True
            smoking_confidence = detected_smoking_conf
        elif self._smoking_persistence > 0:
            self._smoking_persistence -= 1
            smoking_active = True
            smoking_confidence = self._last_smoking_conf
        else:
            self._last_smoking_conf = None
            smoking_active = False
            smoking_confidence = None

        return DistractionAnalysis(
            signals=DriverSignals(
                phone_detected=phone_active,
                smoking_detected=smoking_active,
            ),
            phone_confidence=phone_confidence,
            smoking_confidence=smoking_confidence,
        )

    @staticmethod
    def _load_phone_model(path: Path, device: str | None = None) -> YOLO:
        if not path.exists():
            raise FileNotFoundError(f"Phone detection model was not found: {path}")
        model = YOLO(path)
        if device is not None:
            with contextlib.suppress(Exception):
                model.to(device)
        return model

    def _load_smoking_model(self, path: Path, device: str | None = None) -> YOLOWorld:
        if not path.exists():
            raise FileNotFoundError(f"Smoking detection model was not found: {path}")
        model = YOLOWorld(path)
        model.set_classes(list(self.config.smoking_labels))
        if device is not None:
            with contextlib.suppress(Exception):
                model.to(device)
        return model


def _highest_confidence(
    results: Sequence[Any], target_labels: set[str] | frozenset[str]
) -> float | None:
    highest: float | None = None
    normalized_targets = {label.casefold() for label in target_labels}
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        names = getattr(result, "names", {})
        for box in boxes:
            class_index = int(box.cls[0])
            label = str(names[class_index]).casefold()
            if label not in normalized_targets:
                continue
            confidence = float(box.conf[0])
            if highest is None or confidence > highest:
                highest = confidence
    return highest
