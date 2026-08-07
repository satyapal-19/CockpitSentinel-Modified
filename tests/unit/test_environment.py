"""Smoke tests for environment and package."""

import cockpit_sentinel


def test_version():
    assert cockpit_sentinel.__version__ == "0.1.0"


def test_import_package():
    import cockpit_sentinel  # noqa: F401
