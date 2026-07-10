#!/usr/bin/env bash
set -euo pipefail

# Corridors — Linux/WSL setup script
# Installs dependencies and verifies the install.
# Skips venv creation when running as root (e.g. inside a container).

PYTHON="${PYTHON:-python3}"

echo "=== Corridors setup ==="

# Find Python
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: $PYTHON not found. Install Python 3.10+ and re-run."
    echo "  You can also set PYTHON=python3.13 ./setup.sh"
    exit 1
fi

PY_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Using Python $PY_VERSION"

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

# Install base dependencies
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -e . -q

# Install PyTorch with CUDA if nvidia-smi is available
if command -v nvidia-smi &>/dev/null; then
    echo "NVIDIA GPU detected — installing PyTorch with CUDA..."
    pip install torch --index-url https://download.pytorch.org/whl/cu126 -q
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
