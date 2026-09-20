"""Sunglasses and eye occlusion detection via regional pixel contrast and luminance variance."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


class EyeOcclusionDetector:
    """Detects sunglasses, polarized lenses, and eye occlusion.

    Normal exposed eyes exhibit high spatial variance between the white sclera, dark
    iris/pupil, and surrounding skin. Sunglasses present a uniform dark or polarized
    lens with significantly lower standard deviation (contrast) and/or low mean luminance.
    When occlusion is detected, the system transitions to secondary fatigue indicators
    (head pose nodding / micro-droops and mouth yawning).
    """

    def __init__(
        self,
        *,
        contrast_threshold: float = 12.0,
        darkness_threshold: float = 28.0,
        hysteresis_frames: int = 3,
    ) -> None:
        if contrast_threshold <= 0:
            raise ValueError("contrast_threshold must be positive.")
        if darkness_threshold <= 0:
            raise ValueError("darkness_threshold must be positive.")
        if hysteresis_frames < 1:
            raise ValueError("hysteresis_frames must be at least 1.")

        self.contrast_threshold = contrast_threshold
        self.darkness_threshold = darkness_threshold
        self.hysteresis_frames = hysteresis_frames

        self._consecutive_occluded = 0
        self._consecutive_clear = 0
        self._is_occluded = False

    def update(
        self,
        frame: np.ndarray,
        left_eye_points: Sequence[tuple[float, float]],
        right_eye_points: Sequence[tuple[float, float]],
    ) -> tuple[bool, float, float]:
        """Analyze eye regions in the frame to determine whether eyes are occluded.

        Returns:
            (is_occluded, left_contrast, right_contrast)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        left_std, left_mean = self._analyze_eye_roi(gray, left_eye_points)
        right_std, right_mean = self._analyze_eye_roi(gray, right_eye_points)

        # Eye is considered occluded if contrast is below threshold or mean luminance is very dark
        left_occluded = (left_std < self.contrast_threshold) or (
            left_mean < self.darkness_threshold
        )
        right_occluded = (right_std < self.contrast_threshold) or (
            right_mean < self.darkness_threshold
        )
        raw_occluded = left_occluded and right_occluded

        if raw_occluded:
            self._consecutive_occluded += 1
            self._consecutive_clear = 0
            if self._consecutive_occluded >= self.hysteresis_frames:
                self._is_occluded = True
        else:
            self._consecutive_clear += 1
            self._consecutive_occluded = 0
            if self._consecutive_clear >= self.hysteresis_frames:
                self._is_occluded = False

        return self._is_occluded, left_std, right_std

    def _analyze_eye_roi(
        self, gray: np.ndarray, points: Sequence[tuple[float, float]]
    ) -> tuple[float, float]:
        """Extract rectangular eye ROI with margin and calculate pixel std and mean."""
        if not points:
            return 0.0, 0.0

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        w = max_x - min_x
        h = max_y - min_y

        margin_x = w * 0.20
        margin_y = h * 0.35

        h_img, w_img = gray.shape[:2]
        x1 = max(0, int(min_x - margin_x))
        y1 = max(0, int(min_y - margin_y))
        x2 = min(w_img, int(max_x + margin_x))
        y2 = min(h_img, int(max_y + margin_y))

        if (x2 - x1) < 4 or (y2 - y1) < 4:
            return 0.0, 0.0

        roi = gray[y1:y2, x1:x2]
        return float(np.std(roi)), float(np.mean(roi))

    def reset(self) -> None:
        """Reset internal hysteresis counters."""
        self._consecutive_occluded = 0
        self._consecutive_clear = 0
        self._is_occluded = False
