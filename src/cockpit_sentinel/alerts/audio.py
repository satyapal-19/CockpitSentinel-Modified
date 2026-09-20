"""Three-tier contextual audio escalation subsystem for real-time alerts."""

from __future__ import annotations

import contextlib
import logging
import queue
import sys
import threading
import time

from cockpit_sentinel.domain import RiskLevel

logger = logging.getLogger(__name__)

try:
    import winsound
except ImportError:
    winsound = None  # type: ignore[assignment]


class AlertSoundTier:
    """Sound configuration for an alert tier."""

    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"


class AudioManager:
    """Non-blocking, tiered audio alert generator for vehicle monitoring.

    Executes audio cues on a daemon background thread so video capture and ML
    inference remain unblocked at full frame rate.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        caution_cooldown: float = 5.0,
        warning_cooldown: float = 6.0,
        critical_cooldown: float = 2.0,
    ) -> None:
        self.enabled = enabled
        self.caution_cooldown = caution_cooldown
        self.warning_cooldown = warning_cooldown
        self.critical_cooldown = critical_cooldown

        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=10)
        self._last_played: dict[str, float] = {
            AlertSoundTier.CAUTION: 0.0,
            AlertSoundTier.WARNING: 0.0,
            AlertSoundTier.CRITICAL: 0.0,
        }
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="AudioManagerWorker", daemon=True
        )
        self._worker_thread.start()

    def __enter__(self) -> AudioManager:
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def stop(self) -> None:
        """Signal the audio worker to terminate cleanly."""
        self._stop_event.set()
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

    def trigger(self, level: RiskLevel, reason: str | None = None) -> bool:
        """Evaluate current risk level and enqueue appropriate audio alert if cooled down.

        Returns True if a sound was enqueued, False otherwise.
        """
        if not self.enabled:
            return False

        tier: str | None = None
        cooldown = 0.0

        if level is RiskLevel.CRITICAL:
            tier = AlertSoundTier.CRITICAL
            cooldown = self.critical_cooldown
        elif level is RiskLevel.WARNING:
            tier = AlertSoundTier.WARNING
            cooldown = self.warning_cooldown
        elif level is RiskLevel.CAUTION:
            tier = AlertSoundTier.CAUTION
            cooldown = self.caution_cooldown

        if tier is None:
            return False

        now = time.monotonic()
        if (now - self._last_played[tier]) < cooldown:
            return False

        self._last_played[tier] = now
        try:
            self._queue.put_nowait(tier)
            logger.debug("Enqueued audio alert: tier=%s reason=%s", tier, reason)
            return True
        except queue.Full:
            logger.debug("Audio queue full; skipping alert tier=%s", tier)
            return False

    def play_caution(self) -> None:
        """Direct trigger for gentle caution chime."""
        self._emit_tone_sequence([(523, 120), (659, 150)])

    def play_warning(self) -> None:
        """Direct trigger for double-tone warning."""
        self._emit_tone_sequence([(784, 150), (0, 80), (784, 200)])

    def play_critical(self) -> None:
        """Direct trigger for high-urgency alarm buzzer."""
        self._emit_tone_sequence([(1046, 120), (0, 40), (1318, 120), (0, 40), (1046, 160)])

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                tier = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if tier is None or self._stop_event.is_set():
                break

            try:
                if tier == AlertSoundTier.CRITICAL:
                    self.play_critical()
                elif tier == AlertSoundTier.WARNING:
                    self.play_warning()
                elif tier == AlertSoundTier.CAUTION:
                    self.play_caution()
            except Exception as exc:
                logger.debug("Audio playback exception: %s", exc)
            finally:
                self._queue.task_done()

    @staticmethod
    def _emit_tone_sequence(tones: list[tuple[int, int]]) -> None:
        """Emit a sequence of (frequency_hz, duration_ms) tones."""
        for freq, duration in tones:
            if freq <= 0:
                time.sleep(duration / 1000.0)
            elif winsound is not None and sys.platform == "win32":
                try:
                    winsound.Beep(freq, duration)
                except Exception:
                    time.sleep(duration / 1000.0)
            else:
                # Portable fallback (terminal bell or silence sleep)
                sys.stdout.write("\a")
                sys.stdout.flush()
                time.sleep(duration / 1000.0)
