$ErrorActionPreference = "Stop"

$desktopRoot = Split-Path -Parent $PSScriptRoot
$repositoryRoot = Split-Path -Parent $desktopRoot
$frontendRoot = Join-Path $repositoryRoot "frontend"
$buildRoot = Join-Path $desktopRoot "build"
$stagingRoot = Join-Path $buildRoot "next-export-source"
$outputRoot = Join-Path $stagingRoot "out"
$uiRoot = Join-Path $desktopRoot "ui"
$uiSourceRoot = Join-Path $desktopRoot "ui-source"
$frontendModules = Join-Path $frontendRoot "node_modules"

if (-not (Test-Path $frontendModules)) {
    throw "Install the web frontend dependencies first with npm ci in frontend\."
}

New-Item -ItemType Directory -Force -Path $buildRoot, $stagingRoot, $uiRoot | Out-Null
$null = & robocopy $frontendRoot $stagingRoot /E /XD ".git" ".next" "node_modules" "public" /XF "tsconfig.tsbuildinfo" ".env" ".env.*" "page.tsx" "layout.tsx" "globals.css" "local-model.ts"
if ($LASTEXITCODE -ge 8) {
    throw "Could not copy the frontend into the isolated desktop build directory (robocopy exit $LASTEXITCODE)."
}

$stagingAppRoot = Join-Path $stagingRoot "app"
$stagingPublicRoot = Join-Path $stagingRoot "public"
foreach ($fileName in @("page.tsx", "layout.tsx", "globals.css", "local-model.ts")) {
    $staleSource = Join-Path $stagingAppRoot $fileName
    if (Test-Path $staleSource) {
        Remove-Item -LiteralPath $staleSource -Force
    }
}
if (Test-Path $stagingPublicRoot) {
    Remove-Item -LiteralPath $stagingPublicRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stagingAppRoot, $stagingPublicRoot | Out-Null
Copy-Item -Path (Join-Path $uiSourceRoot "app\*") -Destination $stagingAppRoot -Recurse -Force
Copy-Item -Path (Join-Path $uiSourceRoot "public\*") -Destination $stagingPublicRoot -Recurse -Force

$stagingModules = Join-Path $stagingRoot "node_modules"
if (-not (Test-Path $stagingModules)) {
    New-Item -ItemType Junction -Path $stagingModules -Target $frontendModules | Out-Null
}

$configPath = Join-Path $stagingRoot "next.config.ts"
$config = [System.IO.File]::ReadAllText($configPath)
$configAnchor = "const nextConfig: NextConfig = {"
if (-not $config.Contains($configAnchor)) {
    throw "The Next.js configuration changed; review the desktop static-export patch before building."
}
$configInsert = $configAnchor + [Environment]::NewLine +
    '  output: "export",' + [Environment]::NewLine +
    '  images: { unoptimized: true },'
$config = $config.Replace($configAnchor, $configInsert)
$config = $config.Replace("  cacheComponents: true,", "  cacheComponents: false,")
$config = $config.Replace("  partialPrefetching: true,", "  partialPrefetching: false,")
[System.IO.File]::WriteAllText($configPath, $config, [System.Text.UTF8Encoding]::new($false))

Push-Location $stagingRoot
try {
    & npm run build -- --webpack
    if ($LASTEXITCODE -ne 0) {
        throw "The isolated Next.js static export failed (npm exit $LASTEXITCODE)."
    }
}
finally {
    Pop-Location
}

if (-not (Test-Path (Join-Path $outputRoot "index.html"))) {
    throw "Next.js build succeeded but did not produce out\index.html."
}

$outputFiles = @{}
Get-ChildItem $outputRoot -Recurse -File -Force | ForEach-Object {
    $relativePath = $_.FullName.Substring($outputRoot.Length + 1)
    $outputFiles[$relativePath] = $true
}
Get-ChildItem $uiRoot -Recurse -File -Force | ForEach-Object {
    $relativePath = $_.FullName.Substring($uiRoot.Length + 1)
    if (-not $outputFiles.ContainsKey($relativePath)) {
        Remove-Item -LiteralPath $_.FullName -Force
    }
}
Copy-Item -Path (Join-Path $outputRoot "*") -Destination $uiRoot -Recurse -Force
$staticFiles = Get-ChildItem $uiRoot -Recurse -File -Force
$staticBytes = ($staticFiles | Measure-Object -Property Length -Sum).Sum
Write-Output ("Static UI exported: {0} files, {1:N1} MiB" -f $staticFiles.Count, ($staticBytes / 1MB))
