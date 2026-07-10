#!/usr/bin/env bash
# Launcher for corridors on Linux/WSL. Run: ./corridors.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" exec python3 -m corridors "$@"
