#!/usr/bin/env bash
# Launcher for corridors on Linux/WSL. Run: ./corridors.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Find Python 3.10+
_find_python() {
    for c in "${PYTHON:-}" python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
        [ -z "$c" ] && continue
        if command -v "$c" &>/dev/null; then
            local v
            v=$("$c" -c 'import sys; v=sys.version_info; print(v.major*100+v.minor)' 2>/dev/null) || continue
            [ "$v" -ge 310 ] && { echo "$c"; return 0; }
        fi
    done
    return 1
}
PY=$(_find_python) || { echo "Python 3.10+ not found"; exit 1; }

PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" exec "$PY" -m corridors "$@"
