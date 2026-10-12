[CmdletBinding()]
param([string]$ProjectRoot = (Join-Path $PSScriptRoot '../..'), [switch]$RequireRemote)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$required = @('index.html', 'app.js', 'ui-logic.js', 'style.css', 'package.json',
    'pnpm-lock.yaml', 'vendor/pretext/layout.js', 'vendor/pretext/LICENSE',
    'AGENTS.md', 'for-ai/README.md', 'for-ai/PROJECT.md',
    'for-ai/SKILLS.md', 'for-ai/VERIFICATION.md', 'for-ai/WORKFLOW.md',
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
if ($html -notmatch '<html lang="de">' -or $css -notmatch 'background:\s*#000\s*;') {
    throw 'The page must use German and retain its black background'
}
foreach ($round in @('probe', 'haupt')) {
    if ($html -notmatch "href=`"#$round-runde`"" -or $html -notmatch "id=`"$round-runde`"") {
        throw "Missing $round decision branch"
    }
}
if ([regex]::Matches($html, 'class="round"').Count -ne 2 -or
    [regex]::Matches($html, 'class="round-symbol"').Count -ne 2) {
    throw 'The start must contain two round choices with simple SVG symbols'
}
if ($html -match 'class="app-header"' -or $html -match 'href="#h10"') {
    throw 'The UI must not reintroduce a global context header or an ECG return link'
}
foreach ($round in @('probe', 'haupt')) {
    $expected = if ($round -eq 'probe') { @('anleitung', '5', '10', '15') } else { @('5', '10', '15', '20') }
    $actual = @([regex]::Matches($html, "href=`"#start-$round-([^`"]+)`"") | ForEach-Object { $_.Groups[1].Value })
    if (@(Compare-Object $expected $actual).Count -ne 0) { throw "Incorrect $round options" }
}
if ($html -match '(?i)(src|href)="https?://') {
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
Write-Output 'PASS: Atmungsraum context, four choices per branch, local artwork, locked Pretext, and requested remote gate'
