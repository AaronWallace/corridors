#!/usr/bin/env pwsh
# Start the local Corridors web app. Optional: .\corridors-web.ps1 --port 9000
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = (Join-Path $root 'src') + ';' + $env:PYTHONPATH

Set-Location $root
& py -m corridors.web @args
exit $LASTEXITCODE
