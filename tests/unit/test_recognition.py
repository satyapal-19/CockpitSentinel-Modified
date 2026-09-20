"""Unit tests for DriverRecognizer and facial biometric signature extraction."""

from dataclasses import dataclass

import pytest

from cockpit_sentinel.drowsiness.recognition import (
    DriverProfile,
    DriverRecognizer,
    cosine_similarity,
    extract_biometric_signature,
)


@dataclass
class DummyLandmark:
    x: float
    y: float


def _create_synthetic_landmarks(scale_factor: float = 1.0) -> list[DummyLandmark]:
    """Create synthetic face landmarks with distinct 2D geometry."""
    landmarks = [DummyLandmark(x=0.5, y=0.5) for _ in range(478)]
    # Anchor points with distinct non-degenerate positions
    landmarks[10] = DummyLandmark(x=0.50 * scale_factor, y=0.15 * scale_factor)  # Glabella
    landmarks[168] = DummyLandmark(x=0.50 * scale_factor, y=0.35 * scale_factor)  # Nose bridge
    landmarks[1] = DummyLandmark(x=0.50 * scale_factor, y=0.50 * scale_factor)  # Nose tip
    landmarks[2] = DummyLandmark(x=0.50 * scale_factor, y=0.60 * scale_factor)  # Subnasale
    landmarks[152] = DummyLandmark(x=0.50 * scale_factor, y=0.88 * scale_factor)  # Chin
    landmarks[33] = DummyLandmark(x=0.30 * scale_factor, y=0.35 * scale_factor)  # Left eye outer
    landmarks[133] = DummyLandmark(x=0.42 * scale_factor, y=0.35 * scale_factor)  # Left eye inner
    landmarks[362] = DummyLandmark(x=0.58 * scale_factor, y=0.35 * scale_factor)  # Right eye inner
    landmarks[263] = DummyLandmark(x=0.70 * scale_factor, y=0.35 * scale_factor)  # Right eye outer
    landmarks[234] = DummyLandmark(x=0.20 * scale_factor, y=0.45 * scale_factor)  # Left tragus
    landmarks[454] = DummyLandmark(x=0.80 * scale_factor, y=0.45 * scale_factor)  # Right tragus
    landmarks[172] = DummyLandmark(x=0.28 * scale_factor, y=0.75 * scale_factor)  # Left gonion
    landmarks[397] = DummyLandmark(x=0.72 * scale_factor, y=0.75 * scale_factor)  # Right gonion
    landmarks[116] = DummyLandmark(x=0.25 * scale_factor, y=0.55 * scale_factor)  # Left cheek
    landmarks[345] = DummyLandmark(x=0.75 * scale_factor, y=0.55 * scale_factor)  # Right cheek
    return landmarks


def test_cosine_similarity_identity_and_orthogonality():
    v1 = [1.0, 2.0, 3.0]
    assert cosine_similarity(v1, v1) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([], []) == 0.0


def test_biometric_signature_is_scale_invariant():
    """Verify that scaling the face produces identical signatures (distance invariant)."""
    face_near = _create_synthetic_landmarks(scale_factor=1.0)
    face_far = _create_synthetic_landmarks(scale_factor=0.6)

    sig_near = extract_biometric_signature(face_near, width=640, height=480)
    sig_far = extract_biometric_signature(face_far, width=640, height=480)

    assert sig_near is not None
    assert sig_far is not None
    assert len(sig_near) == 20

    # Near and Far face must have >= 0.999 cosine similarity (scale invariance)
    similarity = cosine_similarity(sig_near, sig_far)
    assert similarity >= 0.999


def test_driver_recognizer_matches_known_driver():
    face = _create_synthetic_landmarks(scale_factor=1.0)
    sig = extract_biometric_signature(face, width=640, height=480)

    known_profile = DriverProfile(
        driver_id="driver_01",
        name="Satyapal Gaikwad",
        resting_ear=0.32,
        ear_threshold=0.22,
        resting_mar=0.18,
        mar_threshold=0.60,
        biometric_signature=sig,
    )

    other_profile = DriverProfile(
        driver_id="driver_02",
        name="Different Person",
        biometric_signature=[s * 0.5 for s in sig],
    )

    recognizer = DriverRecognizer(match_threshold=0.92)
    matched, conf = recognizer.match(
        face, width=640, height=480, profiles=[known_profile, other_profile]
    )

    assert matched is not None
    assert matched.driver_id == "driver_01"
    assert conf >= 0.95


def test_driver_recognizer_rejects_unknown_face():
    face = _create_synthetic_landmarks(scale_factor=1.0)

    # Completely different feature signature
    unmatched_profile = DriverProfile(
        driver_id="driver_99",
        name="Unmatched Driver",
        biometric_signature=[(i * 0.3) % 1.0 for i in range(20)],
    )

    recognizer = DriverRecognizer(match_threshold=0.95)
    matched, conf = recognizer.match(face, width=640, height=480, profiles=[unmatched_profile])

    assert matched is None
    assert conf < 0.95
