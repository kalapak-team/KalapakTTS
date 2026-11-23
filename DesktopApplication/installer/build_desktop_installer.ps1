$ErrorActionPreference = "Stop"

$desktopRoot = Split-Path -Parent $PSScriptRoot
$compiler = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if (-not $compiler) {
    $knownCompilerPath = Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"
    if (Test-Path -LiteralPath $knownCompilerPath -PathType Leaf) {
        $compilerPath = $knownCompilerPath
    }
    else {
        throw "Inno Setup 6 compiler is missing. Install it before building the desktop installer."
    }
}
else {
    $compilerPath = $compiler.Source
}

$runtimePython = Join-Path $desktopRoot "runtime\python\python.exe"
$pythonw = Join-Path $desktopRoot "runtime\python\pythonw.exe"
$ffmpeg = Join-Path $desktopRoot "runtime\bin\ffmpeg.exe"
$uiIndex = Join-Path $desktopRoot "ui\index.html"
$webviewScript = Join-Path $PSScriptRoot "prepare_webview2.ps1"
$webviewRuntime = Join-Path $desktopRoot "build\webview2\MicrosoftEdgeWebView2RuntimeInstallerX64.exe"
$outputDirectory = Join-Path $desktopRoot "build\installer"

foreach ($requiredFile in @($runtimePython, $pythonw, $ffmpeg, $uiIndex)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required installer input is missing: $requiredFile"
    }
}

if (-not (Test-Path -LiteralPath $webviewRuntime -PathType Leaf)) {
    & $webviewScript
    if (-not (Test-Path -LiteralPath $webviewRuntime -PathType Leaf)) {
        throw "Preparing the offline WebView2 runtime did not produce the required installer."
    }
}

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$installerScript = Join-Path $PSScriptRoot "KalapakTTS.iss"
& $compilerPath "/O$outputDirectory" $installerScript
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup compilation failed with exit code $LASTEXITCODE."
}

$installer = Get-ChildItem -LiteralPath $outputDirectory -Filter "KalapakTTS-Desktop-Setup-*.exe" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $installer) {
    throw "Inno Setup completed without producing the expected installer."
}

Write-Host "Installer: $($installer.FullName)"
Write-Host "Installer bytes: $($installer.Length)"
Write-Host "The installer was built only; it was not run."
