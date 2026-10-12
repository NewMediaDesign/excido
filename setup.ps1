# Excido setup for Windows. Same as:  uv sync ; uv run excido setup
#
#   .\setup.ps1                 core models (~3.7 GB)
#   .\setup.ps1 -Models all     all 16 models (~10 GB)
#   .\setup.ps1 -Models none    no models (download them from the app)

param([ValidateSet('core', 'all', 'none')] [string] $Models = 'core')

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host 'uv is required. Install it, open a new terminal and run this script again:' -ForegroundColor Yellow
    Write-Host '  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
    exit 1
}

uv sync
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
uv run --frozen excido setup --models $Models
exit $LASTEXITCODE
