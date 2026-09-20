"""Unit tests for YawnSpeechDiscriminator (Speech vs. Yawn discrimination)."""

import pytest

from cockpit_sentinel.drowsiness.yawn_speech import YawnSpeechDiscriminator


def test_yawn_speech_discriminator_validates_parameters():
    with pytest.raises(ValueError, match="yawn_mar_threshold"):
        YawnSpeechDiscriminator(yawn_mar_threshold=0)
    with pytest.raises(ValueError, match="yawn_min_duration_seconds"):
        YawnSpeechDiscriminator(yawn_min_duration_seconds=-1)
    with pytest.raises(ValueError, match="speech_mar_threshold"):
        YawnSpeechDiscriminator(speech_mar_threshold=0)
    with pytest.raises(ValueError, match="speech_oscillation_hz"):
        YawnSpeechDiscriminator(speech_oscillation_hz=-0.5)
    with pytest.raises(ValueError, match="window_seconds"):
        YawnSpeechDiscriminator(window_seconds=0)


def test_rapid_mouth_oscillation_detected_as_speech_not_yawn():
    discriminator = YawnSpeechDiscriminator(
        yawn_mar_threshold=0.60,
        yawn_min_duration_seconds=1.5,
        speech_mar_threshold=0.38,
        speech_oscillation_hz=1.5,
    )

    # Simulate speaking: syllables opening and closing between 0.25 and 0.48 every 0.25 seconds
    t = 0.0
    for _ in range(8):
        # Open mouth syllable
        is_yawn, is_talking, _ = discriminator.update(mar=0.48, timestamp=t)
        t += 0.15
        # Close mouth
        is_yawn, is_talking, _ = discriminator.update(mar=0.25, timestamp=t)
        t += 0.15

    # Should detect conversational speech and suppress yawn
    assert is_talking
    assert not is_yawn


def test_sustained_wide_aperture_detected_as_yawn():
    discriminator = YawnSpeechDiscriminator(
        yawn_mar_threshold=0.60,
        yawn_min_duration_seconds=1.5,
        speech_mar_threshold=0.38,
    )

    # Initial closed mouth
    is_yawn, is_talking, _ = discriminator.update(mar=0.20, timestamp=0.0)
    assert not is_yawn
    assert not is_talking

    # Driver starts yawning (slow expansion)
    discriminator.update(mar=0.45, timestamp=0.5)
    is_yawn, _, duration = discriminator.update(mar=0.65, timestamp=1.0)
    assert not is_yawn
    assert duration == pytest.approx(0.0)

    # Sustained yawn aperture for 1.6 seconds
    discriminator.update(mar=0.72, timestamp=2.0)
    is_yawn, is_talking, duration = discriminator.update(mar=0.68, timestamp=2.6)

    assert is_yawn
    assert not is_talking
    assert duration >= 1.5


def test_discriminator_reset_clears_state():
    discriminator = YawnSpeechDiscriminator(yawn_min_duration_seconds=1.5)
    discriminator.update(mar=0.75, timestamp=10.0)
    discriminator.update(mar=0.75, timestamp=11.6)

    is_yawn, _, _ = discriminator.update(mar=0.75, timestamp=11.7)
    assert is_yawn

    discriminator.reset()
    is_yawn, is_talking, duration = discriminator.update(mar=0.75, timestamp=12.0)
    assert not is_yawn
    assert duration == 0.0
