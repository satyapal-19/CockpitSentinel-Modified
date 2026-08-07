"""Object-detection signals for phone use and smoking-related distractions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml
from ultralytics import YOLO, YOLOWorld

from cockpit_sentinel.domain import DriverSignals


class ObjectDetector(Protocol):
    """Minimal callable interface shared by Ultralytics detection models."""

    def __call__(self, frame: np.ndarray, **kwargs: Any) -> Sequence[Any]: ...


@dataclass(frozen=True, slots=True)
class DistractionConfig:
    """Confidence thresholds and text labels for distraction detection."""

    phone_confidence_threshold: float = 0.35
    smoking_confidence_threshold: float = 0.20
    smoking_labels: tuple[str, ...] = ("cigarette", "smoking", "vape")

    def __post_init__(self) -> None:
        if not 0 < self.phone_confidence_threshold <= 1:
            raise ValueError("phone_confidence_threshold must be between 0 and 1.")
        if not 0 < self.smoking_confidence_threshold <= 1:
            raise ValueError("smoking_confidence_threshold must be between 0 and 1.")
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

    expected_keys = set(DistractionConfig.__dataclass_fields__)
    if set(raw_config) != expected_keys:
        raise ValueError("Distraction configuration must define every setting.")
    labels = raw_config["smoking_labels"]
    if not isinstance(labels, list):
        raise ValueError("smoking_labels must be a YAML list.")

    return DistractionConfig(
        phone_confidence_threshold=raw_config["phone_confidence_threshold"],
        smoking_confidence_threshold=raw_config["smoking_confidence_threshold"],
        smoking_labels=tuple(labels),
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
    ) -> None:
        self.config = config or DistractionConfig()
        self._phone_model = phone_model or self._load_phone_model(phone_model_path)
        self._smoking_model = smoking_model or self._load_smoking_model(smoking_model_path)

    def analyze(self, frame: np.ndarray) -> DistractionAnalysis:
        """Run both object detectors and return only the relevant risk signals."""

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a three-channel BGR frame.")

        phone_confidence = _highest_confidence(
            self._phone_model(frame, conf=self.config.phone_confidence_threshold, verbose=False),
            {"cell phone"},
        )
        smoking_confidence = _highest_confidence(
            self._smoking_model(
                frame, conf=self.config.smoking_confidence_threshold, verbose=False
            ),
            set(self.config.smoking_labels),
        )
        return DistractionAnalysis(
            signals=DriverSignals(
                phone_detected=phone_confidence is not None,
                smoking_detected=smoking_confidence is not None,
            ),
            phone_confidence=phone_confidence,
            smoking_confidence=smoking_confidence,
        )

    @staticmethod
    def _load_phone_model(path: Path) -> YOLO:
        if not path.exists():
            raise FileNotFoundError(f"Phone detection model was not found: {path}")
        return YOLO(path)

    def _load_smoking_model(self, path: Path) -> YOLOWorld:
        if not path.exists():
            raise FileNotFoundError(f"Smoking detection model was not found: {path}")
        model = YOLOWorld(path)
        model.set_classes(list(self.config.smoking_labels))
        return model


def _highest_confidence(results: Sequence[Any], target_labels: set[str]) -> float | None:
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
