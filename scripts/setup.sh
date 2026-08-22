#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    python3 -m venv "$REPO_ROOT/.venv"
fi

"$VENV_PYTHON" -m pip install --quiet --disable-pip-version-check python-dotenv
"$VENV_PYTHON" "$REPO_ROOT/scripts/setup_environment.py"
