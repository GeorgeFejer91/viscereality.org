param(
    [Parameter(Mandatory)][string]$Output,
    [Parameter(Mandatory)][string]$Keystore,
    [string]$KeyAlias = 'androiddebugkey',
    [string]$AndroidHome = $env:ANDROID_HOME,
    [string]$JavaHome = $env:JAVA_HOME
)
$ErrorActionPreference = 'Stop'
if (!$env:ATMUNGSRAUM_DISCOVERY_STOREPASS) { throw 'Set ATMUNGSRAUM_DISCOVERY_STOREPASS for the signing keystore.' }
$env:JAVA_HOME = $JavaHome
$buildTools = Join-Path $AndroidHome 'build-tools/35.0.0'
$androidJar = Join-Path $AndroidHome 'platforms/android-35/android.jar'
$keystorePath = (Resolve-Path -LiteralPath $Keystore).Path
if (Test-Path -LiteralPath $Output) { throw 'Choose a new output directory.' }
$stage = (New-Item -ItemType Directory -Path $Output).FullName
$classes = (New-Item -ItemType Directory -Path (Join-Path $stage 'classes')).FullName
$dex = (New-Item -ItemType Directory -Path (Join-Path $stage 'dex')).FullName
$package = Join-Path $stage 'unsigned.apk'
$aligned = Join-Path $stage 'aligned.apk'
$apk = Join-Path $stage 'Atmungsraum-Bluetooth.apk'
function Invoke-BuildTool($tool, $arguments) {
    & $tool @arguments
    if ($LASTEXITCODE -ne 0) { throw "$tool failed with exit code $LASTEXITCODE" }
}
$sources = @(Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot src) -Filter '*.java' | ForEach-Object FullName)
Invoke-BuildTool "$JavaHome/bin/javac.exe" (@('--release', '11', '-classpath', $androidJar, '-d', $classes) + $sources)
Invoke-BuildTool "$JavaHome/bin/jar.exe" @('cf', "$stage/classes.jar", '-C', $classes, '.')
Invoke-BuildTool "$buildTools/d8.bat" @('--min-api', '31', '--lib', $androidJar, '--output', $dex, "$stage/classes.jar")
$compiledResources = Join-Path $stage 'resources.zip'
Invoke-BuildTool "$buildTools/aapt2.exe" @('compile', '--dir', "$PSScriptRoot/res", '-o', $compiledResources)
Invoke-BuildTool "$buildTools/aapt2.exe" @('link', '-I', $androidJar, '--manifest', "$PSScriptRoot/AndroidManifest.xml",
    '-o', $package, $compiledResources)
Push-Location -LiteralPath $dex
try { Invoke-BuildTool "$buildTools/aapt.exe" @('add', $package, 'classes.dex') } finally { Pop-Location }
Invoke-BuildTool "$buildTools/zipalign.exe" @('-p', '4', $package, $aligned)
Invoke-BuildTool "$buildTools/apksigner.bat" @('sign', '--ks', $keystorePath, '--ks-key-alias', $KeyAlias, '--ks-pass', 'env:ATMUNGSRAUM_DISCOVERY_STOREPASS', '--out', $apk, $aligned)
Invoke-BuildTool "$buildTools/apksigner.bat" @('verify', '--print-certs', $apk)
Get-FileHash -LiteralPath $apk
