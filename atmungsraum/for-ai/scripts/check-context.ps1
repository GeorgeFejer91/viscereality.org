[CmdletBinding()]
param([string]$ProjectRoot = (Join-Path $PSScriptRoot '../..'), [switch]$RequireRemote)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$required = @('index.html', 'AGENTS.md', 'for-ai/README.md', 'for-ai/PROJECT.md',
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
if ($html -notmatch '(?is)<body>\s*</body>' -or $html -notmatch 'background:\s*#000\s*;') {
    throw 'The initial page must have an empty body and a black background'
}
if ($html -match '(?i)<(script|link|img|video|audio|iframe)\b') {
    throw 'The initial page must not add scripts or external dependencies'
}
if ($RequireRemote) {
    $local = & git -C $root rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve local HEAD' }
    $remoteLine = & git -C $root ls-remote origin refs/heads/main
    if ($LASTEXITCODE -ne 0) { throw 'Cannot read remote main' }
    $remote = ($remoteLine -split '\s+')[0]
    if (-not $remote -or $local.Trim() -ne $remote) { throw 'Local HEAD differs from remote main' }
}
Write-Output 'PASS: Atmungsraum context, blank-page source, and requested remote gate'
