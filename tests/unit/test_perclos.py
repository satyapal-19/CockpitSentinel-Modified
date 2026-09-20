"""Unit tests for the PERCLOS rolling-window fatigue and micro-sleep tracker."""

import pytest

from cockpit_sentinel.drowsiness.detector import PERCLOSTracker


def test_perclos_tracker_initializes_with_valid_parameters():
    tracker = PERCLOSTracker(window_seconds=30.0, fatigue_threshold=0.20, microsleep_seconds=1.5)
    assert tracker.window_seconds == 30.0
    assert tracker.fatigue_threshold == 0.20
    assert tracker.microsleep_seconds == 1.5


def test_perclos_tracker_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="window_seconds"):
        PERCLOSTracker(window_seconds=0)

    with pytest.raises(ValueError, match="fatigue_threshold"):
        PERCLOSTracker(fatigue_threshold=0)

    with pytest.raises(ValueError, match="microsleep_seconds"):
        PERCLOSTracker(microsleep_seconds=-1)


def test_perclos_all_open_eyes_yields_zero():
    tracker = PERCLOSTracker(window_seconds=10.0, fatigue_threshold=0.20)

    for t in range(10):
        perclos, closure_dur, is_fatigued, is_microsleep = tracker.update(
            eyes_closed=False, timestamp=float(t)
        )
        assert perclos == 0.0
        assert closure_dur == 0.0
        assert not is_fatigued
        assert not is_microsleep


def test_perclos_calculates_correct_ratio():
    tracker = PERCLOSTracker(window_seconds=10.0, fatigue_threshold=0.40)

    # 4 closed, 6 open in a 10s window (simulated 1s intervals)
    for t in range(4):
        tracker.update(eyes_closed=True, timestamp=float(t))
    for t in range(4, 10):
        perclos, _, is_fatigued, _ = tracker.update(eyes_closed=False, timestamp=float(t))

    assert perclos == pytest.approx(0.4)
    assert is_fatigued


def test_perclos_purges_old_entries_from_rolling_window():
    tracker = PERCLOSTracker(window_seconds=5.0, fatigue_threshold=0.50)

    # 5 frames of closed eyes at t=0, 1, 2, 3, 4
    for t in range(5):
        tracker.update(eyes_closed=True, timestamp=float(t))

    # At t=4, all 5 frames in window [0..4] are closed -> PERCLOS = 1.0
    perclos, _, is_fatigued, _ = tracker.update(eyes_closed=True, timestamp=4.0)
    assert perclos == 1.0

    # Advance time to t=10 and insert 5 open eye frames
    for t in range(6, 11):
        perclos, _, is_fatigued, _ = tracker.update(eyes_closed=False, timestamp=float(t))

    # All closed frames from t<=4 are purged; PERCLOS should now be 0.0
    assert perclos == 0.0
    assert not is_fatigued


def test_microsleep_triggers_when_closed_exceeds_threshold():
    tracker = PERCLOSTracker(window_seconds=30.0, microsleep_seconds=1.5)

    # Eye closure begins at t=1.0
    _, dur1, _, micro1 = tracker.update(eyes_closed=True, timestamp=1.0)
    assert dur1 == 0.0
    assert not micro1

    # At t=2.0 (duration 1.0s < 1.5s)
    _, dur2, _, micro2 = tracker.update(eyes_closed=True, timestamp=2.0)
    assert dur2 == pytest.approx(1.0)
    assert not micro2

    # At t=2.6 (duration 1.6s >= 1.5s) -> Micro-sleep triggered!
    _, dur3, _, micro3 = tracker.update(eyes_closed=True, timestamp=2.6)
    assert dur3 == pytest.approx(1.6)
    assert micro3

    # Eyes open at t=2.7 -> Micro-sleep cleared, duration resets
    _, dur4, _, micro4 = tracker.update(eyes_closed=False, timestamp=2.7)
    assert dur4 == 0.0
    assert not micro4


def test_perclos_reset_clears_all_history():
    tracker = PERCLOSTracker(window_seconds=10.0, microsleep_seconds=1.0)
    tracker.update(eyes_closed=True, timestamp=1.0)
    tracker.update(eyes_closed=True, timestamp=2.5)

    tracker.reset()

    perclos, dur, is_fatigued, is_microsleep = tracker.update(eyes_closed=False, timestamp=3.0)
    assert perclos == 0.0
    assert dur == 0.0
    assert not is_fatigued
    assert not is_microsleep
