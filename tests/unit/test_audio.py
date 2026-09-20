"""Unit tests for the three-tier audio escalation subsystem."""

from unittest.mock import patch

from cockpit_sentinel.alerts.audio import AlertSoundTier, AudioManager
from cockpit_sentinel.domain import RiskLevel


def test_audio_manager_initializes_with_defaults():
    with AudioManager() as audio:
        assert audio.enabled
        assert audio.caution_cooldown > 0
        assert audio.warning_cooldown > 0
        assert audio.critical_cooldown > 0


def test_audio_manager_does_not_trigger_for_safe_level():
    with AudioManager() as audio:
        assert not audio.trigger(RiskLevel.SAFE)


def test_audio_manager_triggers_and_cooldown_throttles():
    with (
        AudioManager(caution_cooldown=2.0) as audio,
        patch.object(audio, "play_caution") as mock_caution,
    ):
        triggered = audio.trigger(RiskLevel.CAUTION, reason="yawning")
        assert triggered

        # Immediate second trigger should be throttled by cooldown
        assert not audio.trigger(RiskLevel.CAUTION, reason="yawning")

        # Warning tier has its own cooldown timer and can trigger
        assert audio.trigger(RiskLevel.WARNING, reason="phone detected")
        del mock_caution


def test_audio_manager_disabled_mode_suppresses_triggers():
    with AudioManager(enabled=False) as audio:
        assert not audio.trigger(RiskLevel.CRITICAL, reason="microsleep detected")
        assert not audio.trigger(RiskLevel.WARNING)
        assert not audio.trigger(RiskLevel.CAUTION)


def test_audio_manager_tier_mapping():
    with AudioManager(
        caution_cooldown=0.001, warning_cooldown=0.001, critical_cooldown=0.001
    ) as audio:
        assert audio.trigger(RiskLevel.CAUTION)
        assert audio.trigger(RiskLevel.WARNING)
        assert audio.trigger(RiskLevel.CRITICAL)
        assert AlertSoundTier.CAUTION == "caution"
        assert AlertSoundTier.WARNING == "warning"
        assert AlertSoundTier.CRITICAL == "critical"
