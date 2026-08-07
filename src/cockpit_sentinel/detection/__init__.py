"""Object-detection components for driver distraction."""

from cockpit_sentinel.detection.distraction import (
    DistractionAnalysis,
    DistractionConfig,
    DistractionDetector,
    load_distraction_config,
)

__all__ = [
    "DistractionAnalysis",
    "DistractionConfig",
    "DistractionDetector",
    "load_distraction_config",
]
