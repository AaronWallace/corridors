#!/usr/bin/env bash
set -euo pipefail

# Start the local Corridors web app. Optional: ./corridors-web.sh --port 9000
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    PYTHON="$(command -v python)"
fi

exec "$PYTHON" -m corridors.web "$@"
