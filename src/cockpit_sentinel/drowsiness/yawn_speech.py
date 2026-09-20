"""Speech vs. Yawn discrimination using temporal MAR frequency and derivative analysis."""

from __future__ import annotations

import time
from collections import deque


class YawnSpeechDiscriminator:
    """Discriminates conversational speech from genuine yawning.

    Speech consists of rapid, oscillating mouth fluctuations (e.g. talking, singing,
    chewing) with short apertures. A genuine fatigue yawn exhibits a distinct slow
    expansion, sustained wide aperture (>= 1.5 - 2.0s), and minimal high-frequency reversals.
    """

    def __init__(
        self,
        *,
        yawn_mar_threshold: float = 0.60,
        yawn_min_duration_seconds: float = 1.5,
        speech_mar_threshold: float = 0.38,
        speech_oscillation_hz: float = 1.5,
        window_seconds: float = 3.0,
    ) -> None:
        if yawn_mar_threshold <= 0:
            raise ValueError("yawn_mar_threshold must be positive.")
        if yawn_min_duration_seconds <= 0:
            raise ValueError("yawn_min_duration_seconds must be positive.")
        if speech_mar_threshold <= 0:
            raise ValueError("speech_mar_threshold must be positive.")
        if speech_oscillation_hz <= 0:
            raise ValueError("speech_oscillation_hz must be positive.")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")

        self.yawn_mar_threshold = yawn_mar_threshold
        self.yawn_min_duration_seconds = yawn_min_duration_seconds
        self.speech_mar_threshold = speech_mar_threshold
        self.speech_oscillation_hz = speech_oscillation_hz
        self.window_seconds = window_seconds

        self._history: deque[tuple[float, float, float | None]] = deque()
        self._yawn_start_time: float | None = None

    def update(
        self,
        mar: float,
        ear: float | None = None,
        timestamp: float | None = None,
    ) -> tuple[bool, bool, float]:
        """Process one frame sample and determine (is_yawn, is_talking, sustained_duration).

        Returns:
            is_yawn: True if driver is genuinely yawning (sustained wide aperture).
            is_talking: True if driver is speaking/talking (rapid fluctuations).
            sustained_duration: Current continuous duration of MAR >= yawn_mar_threshold.
        """
        now = time.monotonic() if timestamp is None else timestamp
        self._history.append((now, mar, ear))

        # Purge entries outside observation window
        cutoff = now - self.window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        # Track continuous wide-aperture duration
        if mar >= self.yawn_mar_threshold:
            if self._yawn_start_time is None:
                self._yawn_start_time = now
            sustained_duration = now - self._yawn_start_time
        else:
            self._yawn_start_time = None
            sustained_duration = 0.0

        # Compute oscillation frequency and directional reversals
        is_talking = self._detect_speech()

        # A genuine yawn requires:
        # 1. Continuous aperture >= threshold for at least yawn_min_duration_seconds
        # 2. Absence of rapid conversational speech oscillations
        is_yawn = (sustained_duration >= self.yawn_min_duration_seconds) and not is_talking

        return is_yawn, is_talking, sustained_duration

    def _detect_speech(self) -> bool:
        """Analyze temporal derivatives (d(MAR)/dt) to detect conversational speech."""
        if len(self._history) < 4:
            return False

        # Extract samples within the recent window
        samples = list(self._history)
        total_time = samples[-1][0] - samples[0][0]
        if total_time < 0.3:
            return False

        # Count directional reversals (zero-crossings of velocity) above speech_mar_threshold
        reversals = 0
        prev_slope: float | None = None
        has_elevated_mar = False

        for i in range(1, len(samples)):
            t1, m1, _ = samples[i - 1]
            t2, m2, _ = samples[i]
            dt = t2 - t1
            if dt <= 0:
                continue

            if m2 >= self.speech_mar_threshold or m1 >= self.speech_mar_threshold:
                has_elevated_mar = True

            slope = (m2 - m1) / dt
            # Ignore minute sensor noise
            if abs(slope) > 0.1:
                if prev_slope is not None and (
                    (slope > 0 and prev_slope < 0) or (slope < 0 and prev_slope > 0)
                ):
                    reversals += 1
                prev_slope = slope

        oscillation_rate = reversals / total_time if total_time > 0 else 0.0

        # Driver is classified as talking if there are frequent mouth reversals while MAR is active
        return has_elevated_mar and (oscillation_rate >= self.speech_oscillation_hz)

    def reset(self) -> None:
        """Reset history and duration tracking."""
        self._history.clear()
        self._yawn_start_time = None
