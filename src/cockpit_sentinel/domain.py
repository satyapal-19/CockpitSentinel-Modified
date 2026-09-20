"""Shared data contracts for the real-time monitoring pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class RiskLevel(StrEnum):
    """Ordered monitoring states shown to the driver and recorded in events."""

    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class DriverSignals:
    """Boolean events produced for one frame by the detector modules."""

    eyes_closed: bool = False
    yawning: bool = False
    looking_away: bool = False
    phone_detected: bool = False
    smoking_detected: bool = False
    perclos_fatigue: bool = False
    microsleep_detected: bool = False


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    """A transparent assessment built from the current driver signals."""

    score: int
    level: RiskLevel
    reasons: tuple[str, ...]


SIGNAL_WEIGHTS: dict[str, int] = {
    "eyes_closed": 3,
    "yawning": 2,
    "looking_away": 2,
    "phone_detected": 4,
    "smoking_detected": 3,
    "perclos_fatigue": 3,
    "microsleep_detected": 6,
}


SIGNAL_LABELS: dict[str, str] = {
    "eyes_closed": "eyes closed",
    "yawning": "yawning",
    "looking_away": "looking away",
    "phone_detected": "phone detected",
    "smoking_detected": "smoking detected",
    "perclos_fatigue": "high PERCLOS fatigue",
    "microsleep_detected": "microsleep detected",
}


def assess_risk(
    signals: DriverSignals,
    *,
    weights: Mapping[str, int] = SIGNAL_WEIGHTS,
    caution_score: int = 2,
    warning_score: int = 4,
    critical_score: int = 6,
) -> RiskAssessment:
    """Convert independent detector signals into one reproducible risk result."""

    active_signals = [name for name in SIGNAL_LABELS if getattr(signals, name)]
    score = sum(weights[name] for name in active_signals)

    if score >= critical_score:
        level = RiskLevel.CRITICAL
    elif score >= warning_score:
        level = RiskLevel.WARNING
    elif score >= caution_score:
        level = RiskLevel.CAUTION
    else:
        level = RiskLevel.SAFE

    return RiskAssessment(
        score=score,
        level=level,
        reasons=tuple(SIGNAL_LABELS[name] for name in active_signals),
    )
