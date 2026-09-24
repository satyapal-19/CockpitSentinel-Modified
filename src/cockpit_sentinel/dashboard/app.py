"""FastAPI telematics and driver profile dashboard application."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import math
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from cockpit_sentinel.dashboard.worker import MonitoringWorker
from cockpit_sentinel.drowsiness import ProfileManager, extract_biometric_signature

if TYPE_CHECKING:
    from cockpit_sentinel.pipeline.live_monitor import LiveMonitor, MonitorFrame

logger = logging.getLogger(__name__)


def _safe_float(v: float | None, default: float = 0.0) -> float:
    """Ensure floating point values are valid numbers for JSON compliance."""
    if v is None or math.isnan(v) or math.isinf(v):
        return default
    return float(v)


class CreateProfileRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    resting_ear: float = 0.30
    ear_threshold: float = 0.22
    resting_mar: float = 0.18
    mar_threshold: float = 0.60


class UpdateThresholdsRequest(BaseModel):
    ear_threshold: float | None = None
    mar_threshold: float | None = None
    head_pitch_threshold: float | None = None


class TelemetryState:
    """Thread-safe snapshot of live vehicle monitoring telemetry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.ear: float = 0.31
        self.mar: float = 0.19
        self.pitch: float = 0.0
        self.yaw: float = 0.0
        self.perclos: float = 0.0
        self.level: str = "safe"
        self.score: int = 0
        self.eye_occluded: bool = False
        self.talking: bool = False
        self.head_nodding: bool = False
        self.microsleep: bool = False
        self.phone_detected: bool = False
        self.smoking_detected: bool = False
        self.phone_confidence: float | None = None
        self.smoking_confidence: float | None = None
        self.recognized_driver: str = "Default Driver"
        self.recognition_confidence: float = 0.95
        self.fps: float = 0.0
        self.camera_active: bool = False
        self.latest_jpeg: bytes | None = None
        self.ear_samples: deque[float] = deque(maxlen=180)
        self.mar_samples: deque[float] = deque(maxlen=180)
        self.signature_samples: deque[list[float]] = deque(maxlen=60)

    def update_from_monitor(
        self,
        processed: MonitorFrame,
        monitor: LiveMonitor,
        fps: float = 0.0,
    ) -> None:
        with self._lock:
            analysis = processed.analysis
            assessment = processed.assessment
            if analysis.eye_aspect_ratio is not None:
                self.ear = round(_safe_float(analysis.eye_aspect_ratio, 0.31), 3)
                self.ear_samples.append(self.ear)
            if analysis.mouth_aspect_ratio is not None:
                self.mar = round(_safe_float(analysis.mouth_aspect_ratio, 0.19), 3)
                self.mar_samples.append(self.mar)
            if analysis.head_pitch_degrees is not None:
                self.pitch = round(_safe_float(analysis.head_pitch_degrees), 1)
            if analysis.head_yaw_degrees is not None:
                self.yaw = round(_safe_float(analysis.head_yaw_degrees), 1)
            if analysis.perclos is not None:
                self.perclos = round(_safe_float(analysis.perclos), 4)

            # Extract biometric signature for auto-driver enrollment
            if analysis.landmarks is not None and len(analysis.landmarks) >= 468:
                h, w = processed.image.shape[:2]
                sig = extract_biometric_signature(analysis.landmarks, w, h)
                if sig is not None:
                    self.signature_samples.append(sig)

            self.level = str(assessment.level.value)
            self.score = int(assessment.score)
            self.eye_occluded = False
            self.talking = bool(analysis.talking)
            self.head_nodding = False
            self.microsleep = bool(analysis.microsleep_detected)

            # Object-detection distraction signals
            if processed.distraction is not None:
                self.phone_detected = bool(processed.distraction.signals.phone_detected)
                self.smoking_detected = bool(processed.distraction.signals.smoking_detected)
                self.phone_confidence = (
                    round(float(processed.distraction.phone_confidence), 2)
                    if processed.distraction.phone_confidence is not None
                    else None
                )
                self.smoking_confidence = (
                    round(float(processed.distraction.smoking_confidence), 2)
                    if processed.distraction.smoking_confidence is not None
                    else None
                )

            if monitor._recognized_driver_name:
                self.recognized_driver = monitor._recognized_driver_name
                self.recognition_confidence = round(
                    _safe_float(monitor._recognition_confidence, 0.95), 2
                )
            self.fps = round(_safe_float(fps), 1)
            self.camera_active = True

    def get_recent_samples(self, count: int = 90) -> tuple[list[float], list[float]]:
        with self._lock:
            ears = list(self.ear_samples)
            mars = list(self.mar_samples)
        if len(ears) < 10:
            ears = [self.ear] * count
            mars = [self.mar] * count
        elif len(ears) < count:
            ears = (ears * (count // len(ears) + 1))[:count]
            mars = (mars * (count // len(mars) + 1))[:count]
        else:
            ears = ears[-count:]
            mars = mars[-count:]
        return ears, mars

    def get_recent_signatures(self, count: int = 30) -> list[list[float]]:
        with self._lock:
            return list(self.signature_samples)[-count:]

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ear": _safe_float(self.ear, 0.31),
                "mar": _safe_float(self.mar, 0.19),
                "pitch": _safe_float(self.pitch, 0.0),
                "yaw": _safe_float(self.yaw, 0.0),
                "perclos": _safe_float(self.perclos, 0.0),
                "level": self.level,
                "score": self.score,
                "eye_occluded": False,
                "talking": self.talking,
                "head_nodding": False,
                "microsleep": self.microsleep,
                "phone_detected": self.phone_detected,
                "smoking_detected": self.smoking_detected,
                "phone_confidence": self.phone_confidence,
                "smoking_confidence": self.smoking_confidence,
                "recognized_driver": self.recognized_driver,
                "recognition_confidence": _safe_float(self.recognition_confidence, 0.95),
                "fps": _safe_float(self.fps, 0.0),
                "camera_active": self.camera_active,
            }


def create_app(
    profile_manager: ProfileManager | None = None,
    telemetry: TelemetryState | None = None,
    worker: MonitoringWorker | None = None,
    start_camera: bool = False,
    source: int | str = 0,
    device: str = "auto",
    no_audio: bool = False,
) -> FastAPI:
    """Create configured FastAPI application for driver telematics."""
    pm = profile_manager or ProfileManager()
    tel = telemetry or TelemetryState()

    active_worker = worker
    if active_worker is None and start_camera:
        active_worker = MonitoringWorker(
            source=source,
            device=device,
            telemetry=tel,
            profile_manager=pm,
            enable_audio=not no_audio,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if active_worker is not None and start_camera:
            active_worker.start()
        yield
        if active_worker is not None and active_worker.is_running:
            active_worker.stop()

    app = FastAPI(title="CockpitSentinel Telematics Dashboard", lifespan=lifespan)
    active_websockets: list[WebSocket] = []
    template_path = Path(__file__).resolve().parent / "templates" / "index.html"

    @app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        if not template_path.exists():
            return HTMLResponse(
                "<h1>CockpitSentinel Dashboard Template Not Found</h1>",
                status_code=404,
            )
        content = template_path.read_text(encoding="utf-8")
        return HTMLResponse(content)

    @app.get("/api/profiles", response_model=list[dict[str, Any]])
    async def list_profiles() -> list[dict[str, Any]]:
        return [p.to_dict() for p in pm.get_all()]

    @app.post("/api/profiles", response_model=dict[str, Any])
    async def create_profile(req: CreateProfileRequest) -> dict[str, Any]:
        p = pm.create(
            name=req.name,
            resting_ear=req.resting_ear,
            ear_threshold=req.ear_threshold,
            resting_mar=req.resting_mar,
            mar_threshold=req.mar_threshold,
        )
        return p.to_dict()

    @app.get("/api/profiles/{driver_id}", response_model=dict[str, Any])
    async def get_profile(driver_id: str) -> dict[str, Any]:
        p = pm.get(driver_id)
        if not p:
            raise HTTPException(status_code=404, detail="Profile not found")
        return p.to_dict()

    @app.put("/api/profiles/{driver_id}", response_model=dict[str, Any])
    async def update_profile_thresholds(
        driver_id: str, req: UpdateThresholdsRequest
    ) -> dict[str, Any]:
        success = pm.update_thresholds(
            driver_id,
            ear_threshold=req.ear_threshold,
            mar_threshold=req.mar_threshold,
            head_pitch_threshold=req.head_pitch_threshold,
        )
        if not success:
            raise HTTPException(status_code=404, detail="Profile not found")
        p = pm.get(driver_id)
        if p and active_worker is not None and driver_id == pm.get_active_profile().driver_id:
            active_worker.sync_active_profile(p)
        return p.to_dict() if p else {}

    @app.post("/api/profiles/{driver_id}/activate", response_model=dict[str, Any])
    async def activate_profile(driver_id: str) -> dict[str, Any]:
        if not pm.set_active_profile(driver_id):
            raise HTTPException(status_code=404, detail="Profile not found")
        p = pm.get_active_profile()
        tel.recognized_driver = p.name
        if active_worker is not None:
            active_worker.sync_active_profile(p)
            active_worker.reset_recognition()
        return p.to_dict()

    @app.post("/api/profiles/{driver_id}/calibrate", response_model=dict[str, Any])
    async def calibrate_profile(driver_id: str) -> dict[str, Any]:
        p = pm.get(driver_id)
        if not p:
            raise HTTPException(status_code=404, detail="Profile not found")

        sample_ear, sample_mar = tel.get_recent_samples(count=90)
        sample_sigs = tel.get_recent_signatures(count=30)
        calibrated = pm.auto_calibrate(
            driver_id,
            sample_ear,
            sample_mar,
            signature_samples=sample_sigs or None,
        )
        if not calibrated:
            raise HTTPException(status_code=400, detail="Calibration failed")
        if active_worker is not None:
            active_worker.sync_active_profile(calibrated)
            active_worker.reset_recognition()
        return calibrated.to_dict()

    @app.get("/api/telemetry")
    async def get_telemetry() -> dict[str, Any]:
        return tel.to_dict()

    @app.get("/api/video_feed")
    def video_feed() -> StreamingResponse:
        def frame_generator():
            try:
                while True:
                    if tel.latest_jpeg is not None:
                        frame_bytes = tel.latest_jpeg
                    else:
                        img = np.zeros((360, 640, 3), dtype=np.uint8)
                        cv2.putText(
                            img,
                            "Connecting to CockpitSentinel Camera...",
                            (100, 160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (200, 200, 200),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            img,
                            f"Active Profile: {tel.recognized_driver}",
                            (140, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (100, 200, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        _, encoded = cv2.imencode(".jpg", img)
                        frame_bytes = encoded.tobytes()

                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                    time.sleep(0.033)
            except (GeneratorExit, asyncio.CancelledError, ConnectionResetError):
                pass

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.websocket("/ws/telemetry")
    async def ws_telemetry(websocket: WebSocket) -> None:
        await websocket.accept()
        active_websockets.append(websocket)
        try:
            while True:
                await websocket.send_json(tel.to_dict())
                await asyncio.sleep(0.04)
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        finally:
            with contextlib.suppress(ValueError):
                active_websockets.remove(websocket)

    return app


def main() -> None:
    """Launch the dashboard telematics server."""
    parser = argparse.ArgumentParser(description="CockpitSentinel Telematics Web Dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument(
        "--source", default="0", help="Webcam index or video file path (default: 0)"
    )
    parser.add_argument("--device", default="auto", help="Inference device: 'auto', 'cpu', 'cuda'")
    parser.add_argument(
        "--no-audio", action="store_true", help="Disable acoustic alerts (default: audio enabled)"
    )
    parser.add_argument(
        "--no-camera", action="store_true", help="Run dashboard without opening camera"
    )
    args = parser.parse_args()

    app = create_app(
        start_camera=not args.no_camera,
        source=args.source,
        device=args.device,
        no_audio=args.no_audio,
    )
    print("\n=======================================================")
    print("[*] CockpitSentinel Telematics Dashboard Running at:")
    print(f"    http://{args.host}:{args.port}")
    print(f"    Camera Source: {args.source} | Device: {args.device}")
    print("=======================================================\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
