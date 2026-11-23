$ErrorActionPreference = "Stop"

$desktopRoot = Split-Path -Parent $PSScriptRoot
$runtimePython = Join-Path $desktopRoot "runtime\python\python.exe"
$launcher = Join-Path $PSScriptRoot "launcher.py"

if (-not (Test-Path -LiteralPath $runtimePython)) {
    throw "The portable runtime is missing. Prepare DesktopApplication\runtime before launching."
}

Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue
Remove-Item Env:\PYTHONHOME -ErrorAction SilentlyContinue
$env:PYTHONNOUSERSITE = "1"
Push-Location $desktopRoot
try {
& $runtimePython $launcher
if ($LASTEXITCODE -ne 0) {
    throw "KalapakTTS Desktop exited with code $LASTEXITCODE."
}
}
finally {
    Pop-Location
}
