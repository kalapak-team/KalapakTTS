$ErrorActionPreference = "Stop"

$packRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$manifestPath = Join-Path $packRoot "manifest.json"
$manifestHashPath = Join-Path $packRoot "manifest.sha256"

if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Model pack manifest.json is missing."
}
if (-not (Test-Path -LiteralPath $manifestHashPath -PathType Leaf)) {
    throw "Model pack manifest.sha256 is missing."
}

$manifestHashLine = (Get-Content -LiteralPath $manifestHashPath -Raw).Trim()
if ($manifestHashLine -notmatch '^([0-9a-fA-F]{64})\s+\*?manifest\.json$') {
    throw "Model pack manifest.sha256 has an invalid format."
}
$expectedManifestHash = $Matches[1]
$actualManifestHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash
if ($actualManifestHash -ne $expectedManifestHash) {
    throw "Model pack manifest SHA-256 verification failed."
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.schema_version -ne 1 -or -not $manifest.files) {
    throw "Model pack manifest has an unsupported or empty structure."
}

$seenPaths = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)
$requiredPaths = @(
    "VoxCPM2/.gitattributes",
    "VoxCPM2/audiovae.pth",
    "VoxCPM2/config.json",
    "VoxCPM2/manifest.json",
    "VoxCPM2/model.safetensors",
    "VoxCPM2/README.md",
    "VoxCPM2/special_tokens_map.json",
    "VoxCPM2/tokenization_voxcpm2.py",
    "VoxCPM2/tokenizer.json",
    "VoxCPM2/tokenizer_config.json",
    "voices/vanna.wav"
)
$verifiedFiles = @()
$packPrefix = $packRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) +
    [System.IO.Path]::DirectorySeparatorChar
foreach ($entry in $manifest.files) {
    $relativePath = [string]$entry.path
    if ([string]::IsNullOrWhiteSpace($relativePath) -or [System.IO.Path]::IsPathRooted($relativePath)) {
        throw "Model pack contains an invalid relative path."
    }
    $segments = $relativePath -split '[/\\]'
    if ($segments -contains ".." -or $segments -contains "." -or $segments -contains "") {
        throw "Model pack contains a path traversal entry: $relativePath"
    }
    if (-not ($relativePath.StartsWith("VoxCPM2/", [System.StringComparison]::OrdinalIgnoreCase) -or
        $relativePath.StartsWith("voices/", [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "Model pack contains a file outside VoxCPM2/ or voices/: $relativePath"
    }
    if (-not $seenPaths.Add($relativePath)) {
        throw "Model pack contains a duplicate path: $relativePath"
    }

    $sourcePath = [System.IO.Path]::GetFullPath(
        [System.IO.Path]::Combine($packRoot, $relativePath.Replace("/", [System.IO.Path]::DirectorySeparatorChar))
    )
    if (-not $sourcePath.StartsWith($packPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Model pack entry escapes the pack directory: $relativePath"
    }
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        throw "Model pack file is missing: $relativePath"
    }
    $sourceItem = Get-Item -LiteralPath $sourcePath
    if (($sourceItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Model pack symlinks are not supported: $relativePath"
    }
    if ($sourceItem.Length -ne [int64]$entry.size_bytes) {
        throw "Model pack file has an unexpected size: $relativePath"
    }
    $sourceHash = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash
    if ($sourceHash -ne [string]$entry.sha256) {
        throw "Model pack SHA-256 verification failed: $relativePath"
    }
    $verifiedFiles += [PSCustomObject]@{
        RelativePath = $relativePath
        SourcePath = $sourcePath
        Sha256 = [string]$entry.sha256
        Size = [int64]$entry.size_bytes
    }
}

if (-not $seenPaths.Contains("VoxCPM2/manifest.json") -or
    -not $seenPaths.Contains("voices/vanna.wav") -or
    $seenPaths.Count -ne $requiredPaths.Count) {
    throw "Model pack is missing required model files or contains unexpected entries."
}
foreach ($requiredPath in $requiredPaths) {
    if (-not $seenPaths.Contains($requiredPath)) {
        throw "Model pack is missing a required asset: $requiredPath"
    }
}

$appData = [Environment]::GetFolderPath([Environment+SpecialFolder]::ApplicationData)
$dataRoot = Join-Path $appData "KalapakTTS"
$modelsRoot = Join-Path $dataRoot "models"
$stagingRoot = Join-Path $dataRoot (".models-stage-" + [guid]::NewGuid().ToString("N"))
$backupRoot = Join-Path $dataRoot (".models-backup-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $stagingRoot | Out-Null
$backupCreated = $false

try {
    foreach ($entry in $verifiedFiles) {
        $destination = Join-Path $stagingRoot (
            $entry.RelativePath.Replace("/", [System.IO.Path]::DirectorySeparatorChar)
        )
        $destinationDirectory = Split-Path -Parent $destination
        New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null
        Copy-Item -LiteralPath $entry.SourcePath -Destination $destination
        $copiedItem = Get-Item -LiteralPath $destination
        if ($copiedItem.Length -ne $entry.Size -or
            (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -ne $entry.Sha256) {
            throw "Copied model file did not pass SHA-256 verification: $($entry.RelativePath)"
        }
    }

    if (Test-Path -LiteralPath $modelsRoot) {
        Move-Item -LiteralPath $modelsRoot -Destination $backupRoot
        $backupCreated = $true
    }
    Move-Item -LiteralPath $stagingRoot -Destination $modelsRoot
    if ($backupCreated) {
        try {
            Remove-Item -LiteralPath $backupRoot -Recurse -Force
        }
        catch {
            Write-Warning "New models were installed, but the previous model backup remains at $backupRoot"
        }
    }
    Write-Host "Model pack verified and installed at $modelsRoot"
}
catch {
    if (Test-Path -LiteralPath $stagingRoot) {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force
    }
    if ($backupCreated -and -not (Test-Path -LiteralPath $modelsRoot)) {
        Move-Item -LiteralPath $backupRoot -Destination $modelsRoot
    }
    throw
}
