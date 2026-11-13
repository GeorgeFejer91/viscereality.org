[CmdletBinding()]
param([string]$ProjectRoot = (Join-Path $PSScriptRoot '../..'), [switch]$RequireRemote)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$required = @('index.html', 'app.js', 'quest-ui.js', 'quest-rehearsal.js', 'quest-data.js', 'quest-drive.js', 'style.css', 'package.json',
    'manifest.webmanifest', 'pwa-install.js', 'assets/icons/app-192.png', 'assets/icons/app-512.png',
    'assets/icons/app-maskable-512.png', 'assets/icons/apple-touch-icon.png',
    'data-files.js', 'quest-recording-store.js', 'data-logging.js',
    'pnpm-lock.yaml', 'vendor/pretext/layout.js', 'vendor/pretext/LICENSE',
    'AGENTS.md', 'for-ai/README.md', 'for-ai/PROJECT.md',
    'for-ai/SKILLS.md', 'for-ai/VERIFICATION.md', 'for-ai/WORKFLOW.md', 'for-ai/DATA-LOGGING.md',
    'for-ai/DECISIONS.md', 'for-ai/scripts/check-context.ps1')
foreach ($relative in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $root $relative) -PathType Leaf)) {
        throw "Missing $relative"
    }
}
if ((Get-Content -Raw -LiteralPath (Join-Path $root 'AGENTS.md')) -notmatch 'for-ai/README\.md') {
    throw 'AGENTS.md must route to for-ai/README.md'
}
$html = Get-Content -Raw -LiteralPath (Join-Path $root 'index.html')
$css = Get-Content -Raw -LiteralPath (Join-Path $root 'style.css')
$app = Get-Content -Raw -LiteralPath (Join-Path $root 'app.js')
if ($html -match 'data-delivery-locked' -or $css -match 'data-delivery-locked' -or $app -match 'unlockAtStartup|requestDeliveryAccess') {
    throw 'Keep page startup independent of the encrypted delivery configuration'
}
if ($html -notmatch '<html lang="de"(?:\s|>)' -or $css -notmatch 'background:\s*#000\s*;') {
    throw 'The page must use German and retain its black background'
}
if ($html -notmatch '<section class="screen" id="quest" aria-labelledby="quest-title">' -or
    $html -notmatch 'id="quest-participant"' -or $html -notmatch 'id="quest-polar-connect"' -or
    $html -notmatch 'id="quest-export-download"' -or $html -notmatch 'id="quest-drive"') {
    throw 'The entry must expose Quest pairing, participant setup, Polar, download and Drive'
}
if ($html -match 'id="(?:h10|quest-demo|send-new-files|send-all-files|share-qr|wormhole-qr)"' -or
    $app -match 'MockECG|mountSessionData|mountSessionAudio|mountDataFiles') {
    throw 'The live controller must not mount browser recording or delivery'
}
if ($html -notmatch 'id="daten"' -or $app -notmatch 'savedBrowserExports') {
    throw 'Historical browser CSVs must remain downloadable'
}
if ($html -match '(?i)<(?:script|img|audio|video|source|iframe|link)\b[^>]*\b(?:src|href)\s*=\s*["''](?:https?:)?//') {
    throw 'Runtime assets must be local, with no external page dependency'
}
$package = Get-Content -Raw -LiteralPath (Join-Path $root 'package.json') | ConvertFrom-Json
if ($package.dependencies.'@chenglou/pretext' -ne '0.0.9') {
    throw 'Pretext must remain pinned to the validated version'
}
if ($RequireRemote) {
    $local = & git -C $root rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve local HEAD' }
    $remoteLine = & git -C $root ls-remote origin refs/heads/main
    if ($LASTEXITCODE -ne 0) { throw 'Cannot read remote main' }
    $remote = ($remoteLine -split '\s+')[0]
    if (-not $remote -or $local.Trim() -ne $remote) { throw 'Local HEAD differs from remote main' }
}
Write-Output 'PASS: Atmungsraum context, Quest-only recording, Polar/Drive controls, historical CSV recovery, locked Pretext, and requested remote gate'
