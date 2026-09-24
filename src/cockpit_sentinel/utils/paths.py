"""Path and configuration resolution utilities."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import dotenv_values
except ImportError:
    dotenv_values = None  # type: ignore[assignment]


def find_project_root(start_path: Path | None = None) -> Path:
    """Locate the root directory of the CockpitSentinel project.

    Searches upwards from the given path (or this file) for standard project markers
    such as 'pyproject.toml' or '.git', or checks the 'COCKPIT_ROOT' environment variable.
    """
    env_root = os.environ.get("COCKPIT_ROOT")
    if env_root and Path(env_root).is_dir():
        return Path(env_root).resolve()

    current = (start_path or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate

    # Fallback to repo root assumption (3 levels above src/cockpit_sentinel/utils)
    return Path(__file__).resolve().parents[3]


def resolve_config_path(
    config_name: str,
    custom_dir: Path | None = None,
    explicit_file: Path | None = None,
) -> Path:
    """Resolve the location of a YAML configuration file.

    Lookup priority:
    1. explicit_file if specified
    2. custom_dir / config_name if custom_dir is specified
    3. Environment variable 'COCKPIT_CONFIG_DIR' / config_name
    4. <project_root> / 'configs' / config_name
    5. <current_working_directory> / 'configs' / config_name
    """
    if explicit_file is not None:
        return Path(explicit_file)

    if custom_dir is not None:
        return Path(custom_dir) / config_name

    env_config_dir = os.environ.get("COCKPIT_CONFIG_DIR")
    if env_config_dir:
        candidate = Path(env_config_dir) / config_name
        if candidate.exists():
            return candidate

    project_root = find_project_root()
    candidate = project_root / "configs" / config_name
    if candidate.exists():
        return candidate

    cwd_candidate = Path.cwd() / "configs" / config_name
    if cwd_candidate.exists():
        return cwd_candidate

    return project_root / "configs" / config_name


def resolve_model_path(
    filename: str,
    models_root: Path | None = None,
) -> Path:
    """Resolve the path of a pretrained model file.

    Lookup priority:
    1. models_root / 'pretrained' / filename if specified
    2. 'COCKPIT_MODELS_ROOT' environment variable / 'pretrained' / filename
    3. 'COCKPIT_MODELS_ROOT' defined in <project_root>/.env
    4. <project_root> / 'models' / 'pretrained' / filename
    """
    project_root = find_project_root()

    if models_root is not None:
        root_path = Path(models_root)
    elif os.environ.get("COCKPIT_MODELS_ROOT"):
        root_path = Path(os.environ["COCKPIT_MODELS_ROOT"])
    else:
        env_file = project_root / ".env"
        env_values = dotenv_values(env_file) if (dotenv_values and env_file.exists()) else {}
        env_models = env_values.get("COCKPIT_MODELS_ROOT")
        root_path = Path(env_models) if env_models else project_root / "models"

    if root_path.name == "pretrained":
        candidate = root_path / filename
    else:
        candidate = root_path / "pretrained" / filename
    if candidate.exists():
        return candidate

    # Fallback paths on Windows / local setup
    fallback_roots = [
        Path(r"D:\CockpitSentinel\models"),
        Path(r"C:\CockpitSentinel\models"),
        Path(r"C:\Users\SATYAPAL\CockpitSentinel\models"),
        project_root / "models",
    ]
    for fb in fallback_roots:
        p1 = fb / "pretrained" / filename
        if p1.exists():
            return p1
        p2 = fb / filename
        if p2.exists():
            return p2

    return candidate
