"""FastAPI telematics and driver profile dashboard application."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from cockpit_sentinel.drowsiness import ProfileManager

logger = logging.getLogger(__name__)


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
        self.ear: float = 0.31
        self.mar: float = 0.19
        self.pitch: float = 2.4
        self.yaw: float = -1.2
        self.perclos: float = 0.042
        self.level: str = "safe"
        self.score: int = 0
        self.eye_occluded: bool = False
        self.talking: bool = False
        self.head_nodding: bool = False
        self.microsleep: bool = False
        self.recognized_driver: str = "Default Driver"
        self.recognition_confidence: float = 0.96
        self.latest_jpeg: bytes | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ear": self.ear,
            "mar": self.mar,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "perclos": self.perclos,
            "level": self.level,
            "score": self.score,
            "eye_occluded": self.eye_occluded,
            "talking": self.talking,
            "head_nodding": self.head_nodding,
            "microsleep": self.microsleep,
            "recognized_driver": self.recognized_driver,
            "recognition_confidence": self.recognition_confidence,
        }


def create_app(
    profile_manager: ProfileManager | None = None,
    telemetry: TelemetryState | None = None,
) -> FastAPI:
    """Create configured FastAPI application for driver telematics."""
    app = FastAPI(title="CockpitSentinel Telematics Dashboard")
    pm = profile_manager or ProfileManager()
    tel = telemetry or TelemetryState()

    active_websockets: list[WebSocket] = []
    template_path = Path(__file__).resolve().parent / "templates" / "index.html"

    @app.get("/", response_class=HTMLResponse)
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
        return p.to_dict() if p else {}

    @app.post("/api/profiles/{driver_id}/activate", response_model=dict[str, Any])
    async def activate_profile(driver_id: str) -> dict[str, Any]:
        if not pm.set_active_profile(driver_id):
            raise HTTPException(status_code=404, detail="Profile not found")
        p = pm.get_active_profile()
        tel.recognized_driver = p.name
        return p.to_dict()

    @app.post("/api/profiles/{driver_id}/calibrate", response_model=dict[str, Any])
    async def calibrate_profile(driver_id: str) -> dict[str, Any]:
        # Simulate / trigger 90-frame calibration calculation
        p = pm.get(driver_id)
        if not p:
            raise HTTPException(status_code=404, detail="Profile not found")

        # Use current telemetry or baseline samples
        sample_ear = [tel.ear * (0.95 + 0.1 * (i % 5) / 5) for i in range(90)]
        sample_mar = [tel.mar * (0.95 + 0.1 * (i % 5) / 5) for i in range(90)]

        calibrated = pm.auto_calibrate(driver_id, sample_ear, sample_mar)
        if not calibrated:
            raise HTTPException(status_code=400, detail="Calibration failed")
        return calibrated.to_dict()

    @app.get("/api/telemetry")
    async def get_telemetry() -> dict[str, Any]:
        return tel.to_dict()

    @app.get("/api/video_feed")
    def video_feed() -> StreamingResponse:
        def frame_generator():
            while True:
                if tel.latest_jpeg is not None:
                    frame_bytes = tel.latest_jpeg
                else:
                    # Synthetic placeholder frame if camera is not active
                    import numpy as np

                    img = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        img,
                        "CockpitSentinel Video Stream",
                        (140, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (200, 200, 200),
                        2,
                    )
                    cv2.putText(
                        img,
                        f"Active Driver: {tel.recognized_driver}",
                        (170, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (100, 200, 255),
                        1,
                    )
                    _, encoded = cv2.imencode(".jpg", img)
                    frame_bytes = encoded.tobytes()

                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                import time

                time.sleep(0.04)

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
                await asyncio.sleep(0.05)
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
    args = parser.parse_args()

    app = create_app()
    print("\n=======================================================")
    print("🚀 CockpitSentinel Telematics Dashboard Running at:")
    print(f"   http://{args.host}:{args.port}")
    print("=======================================================\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
