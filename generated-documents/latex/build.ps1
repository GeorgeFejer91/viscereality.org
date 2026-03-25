param(
    [string]$Template
)

$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$templatesDir = Join-Path $workspace "templates"
$sharedDir = Join-Path $templatesDir "shared"
$buildRoot = Join-Path $workspace "build"
$outputDir = Join-Path $workspace "output"

foreach ($dir in @($templatesDir, $sharedDir, $buildRoot, $outputDir)) {
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
    }
}

if ($Template) {
    $templatePath = Join-Path $templatesDir $Template
    if (-not (Test-Path -LiteralPath $templatePath)) {
        throw "Template not found: $Template"
    }
    $templateFiles = @(Get-Item -LiteralPath $templatePath)
}
else {
    $templateFiles = Get-ChildItem -LiteralPath $templatesDir -Filter *.tex -File | Sort-Object Name
}

if (-not $templateFiles -or $templateFiles.Count -eq 0) {
    throw "No top-level .tex templates found in $templatesDir"
}

Push-Location $templatesDir
try {
    foreach ($file in $templateFiles) {
        $baseName = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
        $buildDir = Join-Path $buildRoot $baseName

        if (-not (Test-Path -LiteralPath $buildDir)) {
            New-Item -ItemType Directory -Path $buildDir | Out-Null
        }

        Write-Host "Building $($file.Name) ..."

        & pdflatex `
            -interaction=nonstopmode `
            -halt-on-error `
            -output-directory="$buildDir" `
            "$($file.Name)" | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "pdflatex failed for $($file.Name)"
        }

        & pdflatex `
            -interaction=nonstopmode `
            -halt-on-error `
            -output-directory="$buildDir" `
            "$($file.Name)" | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "pdflatex failed on second pass for $($file.Name)"
        }

        $builtPdf = Join-Path $buildDir ($baseName + ".pdf")
        if (-not (Test-Path -LiteralPath $builtPdf)) {
            throw "PDF was not generated for $($file.Name)"
        }

        $targetPdf = Join-Path $outputDir ($baseName + ".pdf")
        Copy-Item -LiteralPath $builtPdf -Destination $targetPdf -Force

        Write-Host "Saved $targetPdf"
    }
}
finally {
    Pop-Location
}
