# Start Excido on http://127.0.0.1:7860 and open the browser. Same as:  uv run excido
#
#   .\run.ps1 -Port 8000 -NoBrowser -Offline

param(
    [int] $Port = 7860,
    [switch] $Offline,
    [switch] $NoBrowser
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$cliArgs = @('run', '--frozen', 'excido', 'serve', '--port', "$Port")
if ($Offline) { $cliArgs += '--offline' }
if ($NoBrowser) { $cliArgs += '--no-browser' }
uv @cliArgs
exit $LASTEXITCODE
