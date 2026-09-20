"""Tests for the shared monitoring risk contract."""

from cockpit_sentinel.domain import DriverSignals, RiskLevel, assess_risk


def test_no_signals_is_safe():
    assessment = assess_risk(DriverSignals())

    assert assessment.score == 0
    assert assessment.level is RiskLevel.SAFE
    assert assessment.reasons == ()


def test_yawning_is_caution():
    assessment = assess_risk(DriverSignals(yawning=True))

    assert assessment.score == 2
    assert assessment.level is RiskLevel.CAUTION
    assert assessment.reasons == ("yawning",)


def test_phone_detection_is_warning():
    assessment = assess_risk(DriverSignals(phone_detected=True))

    assert assessment.score == 4
    assert assessment.level is RiskLevel.WARNING


def test_combined_risk_is_critical_and_explained():
    assessment = assess_risk(DriverSignals(eyes_closed=True, phone_detected=True))

    assert assessment.score == 7
    assert assessment.level is RiskLevel.CRITICAL
    assert assessment.reasons == ("eyes closed", "phone detected")


def test_perclos_fatigue_is_caution():
    assessment = assess_risk(DriverSignals(perclos_fatigue=True))

    assert assessment.score == 3
    assert assessment.level is RiskLevel.CAUTION
    assert assessment.reasons == ("high PERCLOS fatigue",)


def test_microsleep_immediately_triggers_critical():
    assessment = assess_risk(DriverSignals(microsleep_detected=True))

    assert assessment.score == 6
    assert assessment.level is RiskLevel.CRITICAL
    assert assessment.reasons == ("microsleep detected",)
