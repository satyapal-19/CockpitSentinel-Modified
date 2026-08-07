"""Tests for configurable alert policy behavior."""

from pathlib import Path

import pytest

from cockpit_sentinel.alerts.policy import PolicyConfigurationError, load_alert_policy
from cockpit_sentinel.domain import DriverSignals, RiskLevel


def test_committed_alert_policy_scores_phone_detection():
    policy = load_alert_policy(Path("configs/alerts.yaml"))

    assessment = policy.assess(DriverSignals(phone_detected=True))

    assert assessment.score == 4
    assert assessment.level is RiskLevel.WARNING


def test_policy_rejects_non_increasing_thresholds(tmp_path):
    invalid_policy = tmp_path / "alerts.yaml"
    invalid_policy.write_text(
        """weights:
  eyes_closed: 3
  yawning: 2
  looking_away: 2
  phone_detected: 4
  smoking_detected: 3
thresholds:
  caution: 4
  warning: 4
  critical: 6
""",
        encoding="utf-8",
    )

    with pytest.raises(PolicyConfigurationError, match="increase"):
        load_alert_policy(invalid_policy)
