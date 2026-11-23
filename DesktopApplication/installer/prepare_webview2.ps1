$ErrorActionPreference = "Stop"

$desktopRoot = Split-Path -Parent $PSScriptRoot
$destinationDirectory = Join-Path $desktopRoot "build\webview2"
$destination = Join-Path $destinationDirectory "MicrosoftEdgeWebView2RuntimeInstallerX64.exe"
$downloadUrl = "https://go.microsoft.com/fwlink/p/?LinkId=2124701"

New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null
if (-not (Test-Path -LiteralPath $destination -PathType Leaf)) {
    Invoke-WebRequest -Uri $downloadUrl -OutFile $destination -UseBasicParsing
}

$signature = Get-AuthenticodeSignature -LiteralPath $destination
if ($signature.Status -ne "Valid" -or
    $signature.SignerCertificate.Subject -notlike "CN=Microsoft Corporation,*") {
    throw "The WebView2 standalone runtime is not signed by Microsoft or its signature is invalid."
}

$item = Get-Item -LiteralPath $destination
Write-Host "Verified Microsoft-signed WebView2 Evergreen Standalone x64: $($item.Length) bytes"
