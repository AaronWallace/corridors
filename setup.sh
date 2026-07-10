#!/usr/bin/env bash
set -euo pipefail

# Corridors — Linux/WSL setup script
# Creates a venv, installs dependencies, and verifies the install.

PYTHON="${PYTHON:-python3}"
VENV_DIR=".venv"

echo "=== Corridors setup ==="

# Check Python version (need 3.10+)
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: $PYTHON not found. Install Python 3.10+ and re-run."
    exit 1
fi

PY_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$("$PYTHON" -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$("$PYTHON" -c 'import sys; print(sys.version_info.minor)')

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "ERROR: Python 3.10+ required, found $PY_VERSION"
    exit 1
fi
echo "Using Python $PY_VERSION"

# Create venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Install base dependencies
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -e . -q

# Install PyTorch with CUDA if nvidia-smi is available
if command -v nvidia-smi &>/dev/null; then
    echo "NVIDIA GPU detected — installing PyTorch with CUDA..."
    pip install torch --index-url https://download.pytorch.org/whl/cu126 -q
    echo "  PyTorch CUDA: $("$PYTHON" -c 'import torch; print(torch.cuda.is_available())')"
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
echo "Setup complete. Activate the venv with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Run with:"
echo "  python -m corridors"
