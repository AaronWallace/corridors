#!/usr/bin/env bash
set -euo pipefail

# Corridors — Linux/WSL setup script
# Installs dependencies and verifies the install.
# Skips venv creation when running as root (e.g. inside a container).

echo "=== Corridors setup ==="

# Find a suitable Python (3.10+). Check explicit $PYTHON, then common names.
_find_python() {
    for candidate in "${PYTHON:-}" python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
        [ -z "$candidate" ] && continue
        if command -v "$candidate" &>/dev/null; then
            local ver
            ver=$("$candidate" -c 'import sys; v=sys.version_info; print(v.major*100+v.minor)' 2>/dev/null) || continue
            if [ "$ver" -ge 310 ]; then
                echo "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON=$(_find_python) || {
    echo "ERROR: Python 3.10+ not found. Install it or set PYTHON=/path/to/python3.13"
    exit 1
}

PY_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Using $PYTHON ($PY_VERSION)"

# Venv — skip if already in one or running as root (containers)
if [ -z "${VIRTUAL_ENV:-}" ] && [ "$(id -u)" -ne 0 ]; then
    VENV_DIR=".venv"
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment..."
        "$PYTHON" -m venv "$VENV_DIR"
    fi
    echo "Activating venv..."
    source "$VENV_DIR/bin/activate"
fi

# Install base dependencies (use $PYTHON -m pip to match the right interpreter)
echo "Installing dependencies..."
"$PYTHON" -m pip install --upgrade pip -q
"$PYTHON" -m pip install -e . -q

# Install PyTorch with CUDA if nvidia-smi is available
if command -v nvidia-smi &>/dev/null; then
    echo "NVIDIA GPU detected — installing PyTorch with CUDA..."
    "$PYTHON" -m pip install torch --index-url https://download.pytorch.org/whl/cu126 -q
    "$PYTHON" -c 'import torch; print(f"  PyTorch CUDA: {torch.cuda.is_available()}")'
else
    echo "No NVIDIA GPU detected — skipping PyTorch (install manually if needed)."
fi

# Verify
echo ""
echo "=== Verifying ==="
"$PYTHON" -c "from corridors.game import State; print('  game engine: OK')"
"$PYTHON" -c "from corridors.solver import best_move; print('  solver:      OK')"
"$PYTHON" -c "from corridors.nn.encoding import encode_state; print('  nn encoding: OK')"
echo ""
echo "Setup complete. Run with:"
echo "  ./corridors.sh"
echo "  # or: python3 -m corridors"
