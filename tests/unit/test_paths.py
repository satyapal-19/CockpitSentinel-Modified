"""Unit tests for path and configuration resolution utilities."""

from cockpit_sentinel.utils.paths import (
    find_project_root,
    resolve_config_path,
    resolve_model_path,
)


def test_find_project_root_contains_markers():
    root = find_project_root()
    assert root.exists()
    assert (root / "pyproject.toml").exists() or (root / ".git").exists()


def test_resolve_config_path_locates_existing_configs():
    for name in ["alerts.yaml", "drowsiness.yaml", "distraction.yaml"]:
        path = resolve_config_path(name)
        assert path.exists(), f"Configuration file not found: {path}"


def test_resolve_config_path_respects_explicit_path(tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text("test: true\n", encoding="utf-8")

    resolved = resolve_config_path("ignored.yaml", explicit_file=custom)
    assert resolved == custom


def test_resolve_model_path_with_custom_root(tmp_path):
    models_dir = tmp_path / "models"
    resolved = resolve_model_path("test_model.pt", models_root=models_dir)
    assert resolved == models_dir / "pretrained" / "test_model.pt"
