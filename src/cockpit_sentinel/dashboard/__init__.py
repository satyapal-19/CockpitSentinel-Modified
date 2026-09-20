"""CockpitSentinel telematics dashboard package."""

from cockpit_sentinel.dashboard.app import TelemetryState, create_app, main
from cockpit_sentinel.dashboard.worker import MonitoringWorker

__all__ = ["MonitoringWorker", "TelemetryState", "create_app", "main"]
