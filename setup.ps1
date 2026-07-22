# Corridors — Windows setup script (PowerShell 5.1+). Mirrors setup.sh.
# Installs dependencies, builds the Cython engine in place, and verifies.
# Run from anywhere:  .\setup.ps1   (re-run any time; every step is idempotent)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "=== Corridors setup (Windows) ==="

# --- Find a suitable Python (3.10+). Honors $env:PYTHON. -------------------
function Find-Python {
    $specs = @()
    if ($env:PYTHON) { $specs += $env:PYTHON }
    $specs += 'py -3.14', 'py -3.13', 'py -3.12', 'py -3.11', 'py -3.10',
              'python', 'py -3'
    foreach ($spec in $specs) {
        $parts = $spec -split ' '
        $exe = $parts[0]
        if ($parts.Count -gt 1) { $extra = $parts[1..($parts.Count - 1)] }
        else { $extra = @() }
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
        try {
            $resolved = & $exe @extra -c "import sys; v = sys.version_info; print(sys.executable if v >= (3, 10) else '')"
        } catch { continue }
        if ($LASTEXITCODE -eq 0 -and $resolved) { return $resolved.Trim() }
    }
    return $null
}

$py = Find-Python
if (-not $py) {
    Write-Host "ERROR: Python 3.10+ not found. Install it (python.org or 'winget install Python.Python.3.14') or set `$env:PYTHON." -ForegroundColor Red
    exit 1
}
$pyVersion = & $py -c "import sys; print('{}.{}'.format(*sys.version_info[:2]))"
Write-Host "Using $py ($pyVersion)"

# --- C compiler check (needed for the Cython engine; optional). ------------
function Test-MSVC {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { return $false }
    $path = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    return [bool]$path
}

$hasMSVC = Test-MSVC
if (-not $hasMSVC) {
    Write-Host "WARNING: no MSVC C++ toolchain found. The Cython engine will not build" -ForegroundColor Yellow
    Write-Host "         (pure-Python fallback works, but search is ~14x slower)." -ForegroundColor Yellow
    Write-Host "         Fix: open 'Visual Studio Installer' and add the workload" -ForegroundColor Yellow
    Write-Host "         'Desktop development with C++', then re-run .\setup.ps1" -ForegroundColor Yellow
}

# --- Install base dependencies (same path as setup.sh: editable install). --
Write-Host "Installing dependencies..."
& $py -m pip install --upgrade pip -q
& $py -m pip install -e . -q --no-warn-script-location
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip install -e . failed." -ForegroundColor Red
    exit 1
}
# Editable installs do not rebuild the extension when _engine.pyx changes;
# an explicit in-place build is cheap, idempotent, and covers that case.
if ($hasMSVC) {
    $buildOut = & $py setup.py build_ext --inplace
    $notable = $buildOut | Where-Object {
        $_ -match 'NOTE|WARNING|error|copying|could not|remains|loaded|re-run' }
    foreach ($line in $notable) { Write-Host "  $line" }
}

# --- PyTorch (CUDA when an NVIDIA driver is present, else CPU wheel). ------
function Test-GpuTorch {
    try {
        & $py -c "import torch; assert torch.cuda.is_available(); (torch.zeros(8, device='cuda') + 1).sum().item()" 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "NVIDIA GPU detected - installing PyTorch with CUDA..."
    # Pick a wheel channel with kernels for this GPU by the driver's max CUDA
    # version (same logic as setup.sh; override with $env:TORCH_CUDA_CHANNEL).
    $channel = $env:TORCH_CUDA_CHANNEL
    if (-not $channel) {
        $cudaVer = ''
        # Out-String: -match must see one string, not an array of lines,
        # for $Matches to be populated.
        $smi = (& nvidia-smi | Out-String)
        if ($smi -match 'CUDA Version:\s*([0-9]+)\.([0-9]+)') {
            $cudaInt = [int]$Matches[1] * 100 + [int]$Matches[2]
            $cudaVer = "$($Matches[1]).$($Matches[2])"
        } else { $cudaInt = 0 }
        if ($cudaInt -ge 1302) { $channel = 'cu132' }
        elseif ($cudaInt -ge 1300) { $channel = 'cu130' }
        elseif ($cudaInt -ge 1208) { $channel = 'cu129' }
        else { $channel = 'cu126' }
        Write-Host "  driver CUDA: $cudaVer - using wheel channel: $channel"
    }
    & $py -m pip install torch --index-url "https://download.pytorch.org/whl/$channel" -q
    if (-not (Test-GpuTorch)) {
        Write-Host "  installed build has no kernels for this GPU - forcing reinstall on $channel..."
        & $py -m pip install --force-reinstall --no-cache-dir torch --index-url "https://download.pytorch.org/whl/$channel" -q
    }
    if (Test-GpuTorch) {
        & $py -c "import torch; print('  PyTorch {} - CUDA ops: OK'.format(torch.__version__))"
    } else {
        Write-Host "  WARNING: CUDA ops still failing. Try another channel, e.g.:" -ForegroundColor Yellow
        Write-Host "    `$env:TORCH_CUDA_CHANNEL = 'cu130'; .\setup.ps1" -ForegroundColor Yellow
    }
} else {
    Write-Host "No NVIDIA GPU detected - installing CPU PyTorch..."
    & $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu -q
    try {
        & $py -c "import torch; print('  PyTorch {} (CPU): OK'.format(torch.__version__))"
    } catch {
        Write-Host "  WARNING: torch install failed - training/tournaments will not run." -ForegroundColor Yellow
    }
}

# --- Verify -----------------------------------------------------------------
Write-Host ""
Write-Host "=== Verifying ==="
$env:PYTHONPATH = (Join-Path $root 'src') + ';' + $env:PYTHONPATH
& $py -c "from corridors.game import State; print('  game engine: OK')"
& $py -c @"
from corridors import game
if game._ENGINE is not None:
    print('  cython engine: OK (compiled hot path active)')
else:
    print('  cython engine: MISSING - pure-Python fallback (slow search).')
    print('                 Install the MSVC C++ workload and re-run setup.ps1')
"@
& $py -c "from corridors.solver import best_move; print('  solver:      OK')"
& $py -c "from corridors.nn.encoding import encode_state; print('  nn encoding: OK')"
& $py -c "import safetensors; print('  safetensors: OK')"
$torchOk = $false
try {
    & $py -c "import torch" 2>$null
    if ($LASTEXITCODE -eq 0) { $torchOk = $true }
} catch { }
if ($torchOk) {
    Write-Host "  torch:       OK (training + tournaments enabled)"
} else {
    Write-Host "  torch:       MISSING - CPU self-play works, but training/tournaments need it"
}
Write-Host ""
Write-Host "Setup complete. Run with:"
Write-Host "  .\corridors.ps1"
Write-Host "  # or: py -m corridors   (with PYTHONPATH=src)"
