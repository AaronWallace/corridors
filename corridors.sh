#!/usr/bin/env bash
# Launcher for corridors on Linux/WSL. Run: ./corridors.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" exec "$PYTHON" -m corridors "$@"
