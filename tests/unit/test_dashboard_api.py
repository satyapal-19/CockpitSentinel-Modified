"""Unit tests for FastAPI dashboard REST endpoints and telemetry."""

import pytest
from starlette.testclient import TestClient

from cockpit_sentinel.dashboard.app import TelemetryState, create_app
from cockpit_sentinel.drowsiness.recognition import ProfileManager


@pytest.fixture
def client(tmp_path):
    storage = tmp_path / "profiles.json"
    pm = ProfileManager(storage_path=storage)
    tel = TelemetryState()
    app = create_app(profile_manager=pm, telemetry=tel)
    return TestClient(app)


def test_dashboard_index_route_returns_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "CockpitSentinel" in response.text


def test_dashboard_api_list_profiles(client):
    response = client.get("/api/profiles")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["name"] == "Default Driver"


def test_dashboard_api_create_and_update_profile(client):
    # Create profile
    res = client.post(
        "/api/profiles",
        json={"name": "Yashodhan Jadhav", "resting_ear": 0.33, "ear_threshold": 0.23},
    )
    assert res.status_code == 200
    created = res.json()
    assert created["name"] == "Yashodhan Jadhav"
    driver_id = created["driver_id"]

    # Update thresholds
    update_res = client.put(
        f"/api/profiles/{driver_id}",
        json={"ear_threshold": 0.25, "mar_threshold": 0.65},
    )
    assert update_res.status_code == 200
    updated = update_res.json()
    assert updated["ear_threshold"] == 0.25
    assert updated["mar_threshold"] == 0.65


def test_dashboard_api_calibrate_profile(client):
    # Create driver
    res = client.post("/api/profiles", json={"name": "AutoCalib Driver"})
    driver_id = res.json()["driver_id"]

    # Trigger calibrate
    calib_res = client.post(f"/api/profiles/{driver_id}/calibrate")
    assert calib_res.status_code == 200
    calib_data = calib_res.json()
    assert calib_data["calibrated"] is True
    assert calib_data["ear_threshold"] > 0.0


def test_dashboard_api_telemetry_snapshot(client):
    res = client.get("/api/telemetry")
    assert res.status_code == 200
    data = res.json()
    assert "ear" in data
    assert "mar" in data
    assert "level" in data
    assert "fps" in data
    assert "camera_active" in data


def test_telemetry_state_update_and_sampling():
    tel = TelemetryState()
    assert tel.ear == 0.31

    tel.ear_samples.append(0.28)
    tel.mar_samples.append(0.19)
    ears, mars = tel.get_recent_samples(count=10)
    assert len(ears) == 10
    assert len(mars) == 10

    d = tel.to_dict()
    assert "fps" in d
    assert "camera_active" in d
