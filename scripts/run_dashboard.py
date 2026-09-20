"""Launch the CockpitSentinel telematics web dashboard from repository root."""

from __future__ import annotations

import sys
from pathlib import Path


def run() -> None:
    """Add the source package and launch the FastAPI telematics server."""
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from cockpit_sentinel.dashboard.app import main

    main()


if __name__ == "__main__":
    run()
