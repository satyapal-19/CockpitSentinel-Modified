"""Launch the local webcam/video drowsiness monitor from the repository root."""

from __future__ import annotations

import sys
from pathlib import Path


def run() -> None:
    """Add the source package and hand off to the monitor command."""

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from cockpit_sentinel.pipeline.live_monitor import main

    main()


if __name__ == "__main__":
    run()
