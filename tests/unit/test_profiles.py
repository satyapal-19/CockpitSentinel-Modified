"""Unit tests for ProfileManager and driver boundary persistence."""

import pytest

from cockpit_sentinel.drowsiness.recognition import ProfileManager


def test_profile_manager_seeds_default_profile(tmp_path):
    storage = tmp_path / "profiles.json"
    pm = ProfileManager(storage_path=storage)

    profiles = pm.get_all()
    assert len(profiles) >= 1
    active = pm.get_active_profile()
    assert active is not None
    assert active.driver_id == "driver_default"
    assert storage.exists()


def test_profile_manager_create_and_retrieve(tmp_path):
    storage = tmp_path / "profiles.json"
    pm = ProfileManager(storage_path=storage)

    p = pm.create(
        name="Darshan Patil",
        resting_ear=0.25,
        ear_threshold=0.18,
        resting_mar=0.20,
        mar_threshold=0.58,
    )
    assert p.name == "Darshan Patil"
    assert p.ear_threshold == 0.18

    retrieved = pm.get(p.driver_id)
    assert retrieved is not None
    assert retrieved.name == "Darshan Patil"

    # Re-initialize ProfileManager from file to verify disk persistence
    pm2 = ProfileManager(storage_path=storage)
    p2 = pm2.get(p.driver_id)
    assert p2 is not None
    assert p2.name == "Darshan Patil"


def test_profile_manager_update_thresholds(tmp_path):
    storage = tmp_path / "profiles.json"
    pm = ProfileManager(storage_path=storage)

    p = pm.create(name="Test Driver")
    success = pm.update_thresholds(p.driver_id, ear_threshold=0.24, mar_threshold=0.62)
    assert success

    updated = pm.get(p.driver_id)
    assert updated.ear_threshold == 0.24
    assert updated.mar_threshold == 0.62

    # Non-existent ID returns False
    assert not pm.update_thresholds("non_existent_id", ear_threshold=0.25)


def test_profile_manager_auto_calibrate(tmp_path):
    storage = tmp_path / "profiles.json"
    pm = ProfileManager(storage_path=storage)

    p = pm.create(name="Calib Driver")
    # Simulate 90 frames of neutral face: mean EAR ~0.30, mean MAR ~0.20
    ear_samples = [0.30] * 90
    mar_samples = [0.20] * 90

    calibrated = pm.auto_calibrate(p.driver_id, ear_samples, mar_samples)
    assert calibrated is not None
    assert calibrated.calibrated
    assert calibrated.resting_ear == 0.30
    assert calibrated.resting_mar == 0.20
    # Dynamic threshold: 0.30 * 0.70 = 0.21
    assert calibrated.ear_threshold == pytest.approx(0.21)
    # Dynamic MAR threshold: 0.20 * 2.80 = 0.56
    assert calibrated.mar_threshold == pytest.approx(0.56)
