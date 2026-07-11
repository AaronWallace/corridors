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

    # Pick a wheel channel that ships kernels for the installed GPU. Newer GPUs
    # (e.g. Blackwell / RTX 50-series, sm_120) need CUDA 12.8+ builds; a cu126
    # wheel installs cleanly and even reports cuda.is_available()==True, but has
    # no kernels for sm_120 and dies at the first CUDA op. Choose by the driver's
    # max CUDA version (override with TORCH_CUDA_CHANNEL=cu129 etc).
    CH="${TORCH_CUDA_CHANNEL:-}"
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oiE 'CUDA Version:[[:space:]]*[0-9]+\.[0-9]+' \
        | grep -oE '[0-9]+\.[0-9]+' | head -1) || true
    if [ -z "$CH" ]; then
        cuda_int=$(awk -v v="${CUDA_VER:-0}" 'BEGIN{n=split(v,a,"."); print (n>=2)?a[1]*100+a[2]:0}')
        if   [ "$cuda_int" -ge 1302 ]; then CH=cu132
        elif [ "$cuda_int" -ge 1300 ]; then CH=cu130
        elif [ "$cuda_int" -ge 1208 ]; then CH=cu129
        else CH=cu126
        fi
    fi
    echo "  driver CUDA: ${CUDA_VER:-unknown} — using wheel channel: $CH"
    "$PYTHON" -m pip install torch --index-url "https://download.pytorch.org/whl/$CH" -q

    # Verify with a real CUDA op (is_available() alone doesn't catch missing
    # kernels). If it fails, the wrong build is present — force a clean reinstall.
    _gpu_ok() {
        "$PYTHON" - <<'PY' >/dev/null 2>&1
import torch
assert torch.cuda.is_available()
(torch.zeros(8, device="cuda") + 1).sum().item()
PY
    }
    if ! _gpu_ok; then
        echo "  installed build has no kernels for this GPU — forcing reinstall on $CH..."
        "$PYTHON" -m pip install --force-reinstall --no-cache-dir torch \
            --index-url "https://download.pytorch.org/whl/$CH" -q
    fi
    if _gpu_ok; then
        "$PYTHON" -c 'import torch; print(f"  PyTorch {torch.__version__} · CUDA ops: OK")'
    else
        echo "  WARNING: CUDA ops still failing. Try a different channel, e.g.:"
        echo "    TORCH_CUDA_CHANNEL=cu130 ./setup.sh"
    fi
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
