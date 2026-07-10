#!/usr/bin/env pwsh
# Launcher for corridors. Run: .\corridors.ps1  (pass any flags after)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = (Join-Path $root 'src') + ';' + $env:PYTHONPATH
& py -m corridors @args
exit $LASTEXITCODE
