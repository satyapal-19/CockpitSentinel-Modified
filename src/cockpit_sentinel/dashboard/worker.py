"""Background worker running the real-time live monitoring pipeline for the dashboard."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from cockpit_sentinel.alerts.audio import AudioManager
from cockpit_sentinel.alerts.policy import load_alert_policy
from cockpit_sentinel.detection import (
    DistractionAnalysis,
    DistractionDetector,
    load_distraction_config,
)
from cockpit_sentinel.drowsiness import (
    DriverProfile,
    DrowsinessDetector,
    ProfileManager,
    load_drowsiness_config,
)
from cockpit_sentinel.pipeline.live_monitor import LiveMonitor, parse_source
from cockpit_sentinel.utils.device import resolve_device
from cockpit_sentinel.utils.paths import resolve_config_path, resolve_model_path

if TYPE_CHECKING:
    from cockpit_sentinel.dashboard.app import TelemetryState

logger = logging.getLogger(__name__)


class DecimatedDistractionDetector:
    """Wrapper that evaluates heavy YOLO detection every N frames to maintain high FPS."""

    def __init__(self, detector: DistractionDetector, interval: int = 5) -> None:
        self._detector = detector
        self._interval = max(1, interval)
        self._count = 0
        self._cached: DistractionAnalysis | None = None

    def analyze(self, frame: np.ndarray) -> DistractionAnalysis:
        self._count += 1
        if self._cached is None or self._count % self._interval == 0:
            self._cached = self._detector.analyze(frame)
        return self._cached

    def close(self) -> None:
        self._detector.close()


class MonitoringWorker:
    """Runs video capture and LiveMonitor in a dedicated background thread."""

    def __init__(
        self,
        source: int | str = 0,
        device: str = "auto",
        telemetry: TelemetryState | None = None,
        profile_manager: ProfileManager | None = None,
        enable_audio: bool = False,
        enable_distraction: bool = True,
        models_root: Path | None = None,
        config_dir: Path | None = None,
    ) -> None:
        self.source = parse_source(str(source))
        self.device_str = device
        self.telemetry = telemetry
        self.profile_manager = profile_manager or ProfileManager()
        self.enable_audio = enable_audio
        self.enable_distraction = enable_distraction
        self.models_root = models_root
        self.config_dir = config_dir

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._monitor: LiveMonitor | None = None
        self._detector: DrowsinessDetector | None = None
        self._audio_mgr: AudioManager | None = None
        self._distraction_detector: DistractionDetector | None = None
        self._fps: float = 0.0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="DashboardMonitoringWorker",
        )
        self._thread.start()
        logger.info("MonitoringWorker started on source: %s", self.source)

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._cleanup()
        logger.info("MonitoringWorker stopped.")

    def sync_active_profile(self, profile: DriverProfile) -> None:
        """Apply active profile boundaries directly to detector and monitor."""
        if self._detector and hasattr(self._detector, "config"):
            self._detector.config = replace(
                self._detector.config,
                eye_aspect_ratio_threshold=profile.ear_threshold,
                mouth_aspect_ratio_threshold=profile.mar_threshold,
            )
        if self._monitor:
            self._monitor.apply_profile(profile)

    def reset_recognition(self) -> None:
        """Reset driver recognition on the active live monitor."""
        if self._monitor:
            self._monitor.reset_recognition()

    def _initialize_pipeline(self) -> LiveMonitor:
        drowsiness_cfg_path = resolve_config_path("drowsiness.yaml", custom_dir=self.config_dir)
        alerts_cfg_path = resolve_config_path("alerts.yaml", custom_dir=self.config_dir)
        drowsiness_config = load_drowsiness_config(drowsiness_cfg_path)
        alert_policy = load_alert_policy(alerts_cfg_path)

        active_profile = self.profile_manager.get_active_profile()
        drowsiness_config = replace(
            drowsiness_config,
            eye_aspect_ratio_threshold=active_profile.ear_threshold,
            mouth_aspect_ratio_threshold=active_profile.mar_threshold,
        )

        target_device = resolve_device(self.device_str)
        model_path = resolve_model_path("face_landmarker.task", self.models_root)

        if not model_path.exists():
            raise FileNotFoundError(f"MediaPipe face landmarker model not found: {model_path}")

        self._detector = DrowsinessDetector(model_path, drowsiness_config, delegate=target_device)

        if self.enable_distraction:
            phone_path = resolve_model_path("yolov8n.pt", self.models_root)
            smoking_path = resolve_model_path("yolov8s-worldv2.pt", self.models_root)
            if phone_path.exists() and smoking_path.exists():
                distraction_cfg_path = resolve_config_path(
                    "distraction.yaml", custom_dir=self.config_dir
                )
                distraction_cfg = load_distraction_config(distraction_cfg_path)
                try:
                    raw_detector = DistractionDetector(
                        phone_path,
                        smoking_path,
                        distraction_cfg,
                        device=target_device,
                    )
                    self._distraction_detector = DecimatedDistractionDetector(
                        raw_detector, interval=6
                    )
                except Exception as exc:
                    logger.warning("Distraction detector initialization bypassed: %s", exc)

        if self.enable_audio:
            self._audio_mgr = AudioManager(enabled=True)

        self._monitor = LiveMonitor(
            detector=self._detector,
            policy=alert_policy,
            distraction_detector=self._distraction_detector,
            audio_manager=self._audio_mgr,
            profile_manager=self.profile_manager,
        )
        return self._monitor

    def _render_diagnostic_frame(self, header: str, detail: str) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (20, 160), (620, 320), (30, 30, 80), -1)
        cv2.rectangle(img, (20, 160), (620, 320), (0, 140, 255), 2)
        cv2.putText(
            img,
            header,
            (40, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            detail[:60],
            (40, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"Source: {self.source} | Device: {self.device_str}",
            (40, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        ok_enc, buf = cv2.imencode(".jpg", img)
        if ok_enc and self.telemetry is not None:
            self.telemetry.latest_jpeg = buf.tobytes()

    def _cleanup(self) -> None:
        if self._detector is not None:
            with contextlib.suppress(Exception):
                self._detector.close()
            self._detector = None

        if self._distraction_detector is not None:
            with contextlib.suppress(Exception):
                self._distraction_detector.close()
            self._distraction_detector = None

        if self._audio_mgr is not None:
            with contextlib.suppress(Exception):
                self._audio_mgr.close()
            self._audio_mgr = None

    def _run(self) -> None:
        self._render_diagnostic_frame(
            "CockpitSentinel Initializing...",
            "Loading neural network models and opening camera...",
        )
        try:
            monitor = self._initialize_pipeline()
        except Exception as exc:
            logger.error("Failed to initialize monitoring pipeline: %s", exc)
            self._render_diagnostic_frame("Pipeline Initialization Error", str(exc))
            return

        def _open_camera() -> cv2.VideoCapture | None:
            cap = None
            if isinstance(self.source, int):
                cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(self.source)
            else:
                cap = cv2.VideoCapture(self.source)
            return cap if (cap is not None and cap.isOpened()) else None

        capture = _open_camera()
        while capture is None and not self._stop_event.is_set():
            logger.warning("Could not open video source: %s. Retrying in 2 seconds...", self.source)
            self._render_diagnostic_frame(
                "Camera Connecting...",
                "Waiting for webcam access. Retrying automatically...",
            )
            time.sleep(2.0)
            capture = _open_camera()

        if self._stop_event.is_set() or capture is None:
            if capture is not None:
                capture.release()
            self._cleanup()
            return

        fps_timer: float | None = None
        frames_this_second = 0

        try:
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    if isinstance(self.source, str) and not self.source.isdigit():
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    time.sleep(0.04)
                    continue

                if fps_timer is None:
                    fps_timer = time.time()

                try:
                    processed = monitor.process(frame)
                except Exception as exc:
                    logger.warning("Error processing frame: %s", exc)
                    time.sleep(0.01)
                    continue

                frames_this_second += 1
                now = time.time()
                if now - fps_timer >= 1.0:
                    self._fps = frames_this_second / (now - fps_timer)
                    frames_this_second = 0
                    fps_timer = now

                if self.telemetry is not None:
                    self.telemetry.update_from_monitor(processed, monitor, fps=self._fps)
                    ok_enc, buf = cv2.imencode(
                        ".jpg",
                        processed.image,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                    )
                    if ok_enc:
                        self.telemetry.latest_jpeg = buf.tobytes()

                time.sleep(0.005)
        finally:
            if capture is not None:
                capture.release()
            self._cleanup()
