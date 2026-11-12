#Requires -Version 7.6
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$QuestRepo,
    [Parameter(Mandatory)][string]$ExperimentAppSpec,
    [Parameter(Mandatory)][string]$ExperimentFeatureLock,
    [Parameter(Mandatory)][string]$PrivateAssetProviderManifest,
    [Parameter(Mandatory)][ValidatePattern('^[a-z0-9][a-z0-9-]{0,39}$')][string]$RunName,
    [string]$OpenXrLoader,
    [string]$Keystore,
    [string]$RetainedCargoTargetDirectory,
    [ValidateRange(1, 2100000000)][int]$VersionCode = 1,
    [string]$VersionName = '0.1.0',
    [switch]$ResolveOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$QuestRepo = (Resolve-Path -LiteralPath $QuestRepo).Path
$ExperimentAppSpec = (Resolve-Path -LiteralPath $ExperimentAppSpec).Path
$ExperimentFeatureLock = (Resolve-Path -LiteralPath $ExperimentFeatureLock).Path
$PrivateAssetProviderManifest = (Resolve-Path -LiteralPath $PrivateAssetProviderManifest).Path
if (!$ResolveOnly) {
    if (!$OpenXrLoader) { throw 'A build needs the ARM64 OpenXR loader.' }
    $OpenXrLoader = (Resolve-Path -LiteralPath $OpenXrLoader).Path
    if ($Keystore) { $Keystore = (Resolve-Path -LiteralPath $Keystore).Path }
}

$baseLock = Get-Content -Raw -LiteralPath $ExperimentFeatureLock | ConvertFrom-Json
if ($baseLock.schema -cne 'rusty.quest.native_app_feature_lock.v1' -or
    (Get-FileHash -LiteralPath $ExperimentAppSpec).Hash -ne $baseLock.app_spec_sha256) {
    throw 'The Experimenter feature lock must bind the exact app spec.'
}
$inherited = @($baseLock.build_inputs.env | Where-Object { $_.PSObject.Properties['source'] -and $_.source -like 'feature:*:inherited-environment' })
foreach ($entry in $inherited) {
    if ($entry.name -cnotmatch '^RUSTY_QUEST_NATIVE_RENDERER_[A-Z0-9_]+$') {
        throw 'An inherited native build input falls outside the renderer environment.'
    }
}

$app = Get-Content -Raw -LiteralPath $ExperimentAppSpec | ConvertFrom-Json -AsHashtable
if ($app.schema -cne 'rusty.quest.native_app_build.v1' -or
    'ui.experiment_session_command_gateway' -cnotin $app.requested_features -or
    !$app.Contains('private_asset_provider_request')) {
    throw 'Use the Experimenter app spec derived by the native product owner.'
}

$runRoot = Join-Path $QuestRepo "local-artifacts/atmungsraum/$RunName"
$apkOutput = Join-Path $QuestRepo "target/aq2/$RunName"
$resolved = Join-Path ([IO.Path]::GetTempPath()) "atmungsraum-native/$RunName/resolved"
if ((Test-Path -LiteralPath $runRoot) -or (Test-Path -LiteralPath $apkOutput) -or (Test-Path -LiteralPath $resolved)) {
    throw "Choose a new assembly run name: $RunName"
}
& git -C $QuestRepo check-ignore -q -- "local-artifacts/atmungsraum/$RunName/"
if ($LASTEXITCODE -ne 0) { throw 'The native assembly workspace must be ignored by Git.' }

$audio = Join-Path $runRoot 'atmungsraum-audio'
& node (Join-Path $PSScriptRoot 'stage-audio.mjs') $audio
if ($LASTEXITCODE -ne 0) { throw 'Atmungsraum scenario audio staging failed.' }
$catalogPath = Join-Path $audio 'scenario-catalog.json'
$catalog = Get-Content -Raw -LiteralPath $catalogPath | ConvertFrom-Json
$catalogHash = (Get-FileHash -LiteralPath $catalogPath).Hash.ToLowerInvariant()

$featureId = 'audio.atmungsraum_scenarios'
$featureDir = Join-Path $runRoot 'features'
Copy-Item -LiteralPath (Join-Path $QuestRepo 'fixtures/native-app-features') -Destination $featureDir -Recurse
$assetInput = [IO.Path]::GetRelativePath($QuestRepo, $audio).Replace('\','/')
$feature = [ordered]@{
    schema = 'rusty.quest.native_app_feature.v1'
    feature_id = $featureId
    module_path = 'audio/atmungsraum-scenarios'
    module_kind = 'app-owned-scenario-library'
    settings_surface = @{ authority = 'rusty.quest.native_app_settings.v1'; adapter = 'packaged-feature-selection' }
    owner_lane = 'atmungsraum'
    status = 'app-specific'
    description = 'Atmungsraum authored scenario JSON and replaceable cue files.'
    provides = @($featureId)
    depends_on = @('ui.experiment_session_command_gateway')
    incompatible_with = @()
    exclusive_groups = @{}
    android_manifest = @{ permissions = @(); uses_features = @(); activities = @(); services = @(); queries = @() }
    runtime_profile = @{ set = @{}; clear_families = @(); expected_render_modes = @() }
    build_inputs = @{
        env = @(@{ name = 'RUSTY_QUEST_NATIVE_RENDERER_SCENARIO_CATALOG_SHA256'; value = $catalogHash })
        assets = @($assetInput)
        shaders = @()
    }
    markers = @{ required = @(); forbidden = @() }
    validation = @{ quick = @(); standard = @(); device = @() }
    public_private_boundary = 'Packages the public Atmungsraum scenario library without signing or device data.'
}
$feature | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $featureDir "$featureId.feature.json") -Encoding utf8NoBOM
$scenarioSourceDir = 'apps/native-renderer-android/panel-modules/condition-audio/src/main/java/io/github/mesmerprism/rustyquest/native_renderer'
$scenarioSources = @('ScenarioAudioPackage.java', 'ScenarioAudioAndroidTracks.java') | ForEach-Object { "$scenarioSourceDir/$_" }
foreach ($source in $scenarioSources) {
    if (!(Test-Path -LiteralPath (Join-Path $QuestRepo $source) -PathType Leaf)) {
        throw 'Apply the Atmungsraum native scenario source before assembly.'
    }
}
$panelDescriptor = Join-Path $featureDir 'ui/breath-composition-panel/ui.breath_composition_control_panel.feature.json'
$panel = Get-Content -Raw -LiteralPath $panelDescriptor | ConvertFrom-Json -AsHashtable
$audioModule = @($panel.panel_composition.modules | Where-Object { $_.module_id -ceq 'condition-audio-runtime' })
if ($audioModule.Count -ne 1) { throw 'The selected panel must contain the condition audio owner.' }
$audioModule[0].source_files = @($audioModule[0].source_files) + $scenarioSources
$panel | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $panelDescriptor -Encoding utf8NoBOM

$app.app_id = 'atmungsraum_quest2'
$app.package_policy = 'app-specific-package'
$app.package_name = 'org.viscereality.atmungsraum.rehearsal'
$app.application_label = 'Atmungsraum Quest 2'
$app.experiment_gateway_origins = @('https://viscereality.org')
$app.quest_supported_devices = @('quest2')
$app.requested_features = @($app.requested_features) + @($featureId, 'ui.experiment_private_lan_data', 'ui.experiment_scene_recenter')
$specPath = Join-Path $runRoot 'atmungsraum.app.json'
$app | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $specPath -Encoding utf8NoBOM

$previousEnvironment = @{}
foreach ($entry in $inherited) {
    $previousEnvironment[$entry.name] = [Environment]::GetEnvironmentVariable($entry.name)
    [Environment]::SetEnvironmentVariable($entry.name, [string]$entry.value)
}
Push-Location $QuestRepo
try {
    $resolveArgs = @(
        '-AppSpec', $specPath, '-FeatureDir', $featureDir, '-OutputRoot', $resolved,
        '-PrivateAssetProviderManifest', $PrivateAssetProviderManifest, '-DryRun'
    )
    & pwsh -NoProfile -File ./tools/Resolve-NativeAppBuild.ps1 @resolveArgs
    if ($LASTEXITCODE -ne 0) { throw 'Native app resolution failed.' }
    $lockPath = Get-ChildItem -LiteralPath $resolved -Recurse -File -Filter feature-lock.json |
        Select-Object -First 1 -ExpandProperty FullName
    if (!$lockPath) { throw 'Native feature lock was not written.' }
    $lock = Get-Content -Raw -LiteralPath $lockPath | ConvertFrom-Json
    if ($assetInput -cnotin @($lock.build_inputs.assets)) {
        throw 'The resolved native feature lock does not include the scenario library.'
    }
    if (@($scenarioSources | Where-Object { $_ -cnotin @($lock.panel_source_closure.source_files.path) }).Count -or
        $catalogHash -cnotin @($lock.build_inputs.env | Where-Object name -CEQ 'RUSTY_QUEST_NATIVE_RENDERER_SCENARIO_CATALOG_SHA256' | ForEach-Object value)) {
        throw 'The resolved native feature lock must bind the scenario parser and catalog digest.'
    }
    $manifestPath = Join-Path (Split-Path -Parent $lockPath) 'AndroidManifest.xml'
    [xml]$manifest = Get-Content -Raw -LiteralPath $manifestPath
    $androidNamespace = 'http://schemas.android.com/apk/res/android'
    $metadata = @{}
    foreach ($entry in $manifest.SelectNodes('/manifest/application/meta-data')) {
        $metadata[$entry.GetAttribute('name', $androidNamespace)] = $entry.GetAttribute('value', $androidNamespace)
    }
    if ($metadata['com.oculus.supportedDevices'] -cne 'quest2' -or
        $metadata['rusty.quest.experiment.command_gateway.origins'] -cne 'https://viscereality.org') {
        throw 'The native manifest must target Quest 2 and admit the exact Atmungsraum origin.'
    }
    if ($metadata['rusty.quest.experiment.command_gateway.private_lan_data'] -cne 'true') {
        throw 'The native manifest must select encrypted LAN sensor and recording readback.'
    }
    if ($metadata['rusty.quest.experiment.command_gateway.scene_recenter'] -cne 'true') {
        throw 'The native manifest must select enrolled scene recentering.'
    }
    if ($ResolveOnly) {
        Get-Item -LiteralPath $lockPath
        return
    }
    $buildArgs = @(
        '-AppBuildLock', $lockPath, '-OutDir', $apkOutput, '-OpenXrLoader', $OpenXrLoader,
        '-AndroidVersionCode', $VersionCode, '-AndroidVersionName', $VersionName
    )
    if ($Keystore) { $buildArgs += @('-Keystore', $Keystore) }
    if ($RetainedCargoTargetDirectory) {
        $buildArgs += @('-RetainedCargoTargetDirectory', $RetainedCargoTargetDirectory)
    }
    & pwsh -NoProfile -File ./tools/Build-NativeRendererAndroid.ps1 @buildArgs
    if ($LASTEXITCODE -ne 0) { throw 'Native APK build failed.' }
} finally {
    Pop-Location
    foreach ($name in $previousEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name])
    }
}

$apks = @(Get-ChildItem -LiteralPath $apkOutput -Recurse -File -Filter 'rusty-quest-native-renderer.apk')
if ($apks.Count -ne 1) { throw 'Expected one signed APK from the selected output.' }
$apk = $apks[0]
$archive = [IO.Compression.ZipFile]::OpenRead($apk.FullName)
try {
    $loaderEntry = $archive.GetEntry('lib/arm64-v8a/libopenxr_loader.so')
    if (!$loaderEntry) { throw 'APK is missing the ARM64 OpenXR loader.' }
    $loaderStream = $loaderEntry.Open()
    try {
        if ((Get-FileHash -InputStream $loaderStream).Hash -ne (Get-FileHash -LiteralPath $OpenXrLoader).Hash) {
            throw 'APK ARM64 OpenXR loader differs from the selected file.'
        }
    } finally { $loaderStream.Dispose() }
    $entries = @($catalog.files) + @([pscustomobject]@{
        path = 'scenario-catalog.json'
        bytes = (Get-Item -LiteralPath $catalogPath).Length
        sha256 = (Get-FileHash -LiteralPath $catalogPath).Hash
    })
    foreach ($file in $entries) {
        $entry = $archive.GetEntry("assets/atmungsraum-audio/$($file.path)")
        if (!$entry -or $entry.Length -ne $file.bytes) { throw "APK scenario asset is absent or truncated: $($file.path)" }
        $stream = $entry.Open()
        try {
            if ((Get-FileHash -InputStream $stream).Hash -ne $file.sha256) { throw "APK scenario asset differs: $($file.path)" }
        } finally { $stream.Dispose() }
    }
} finally { $archive.Dispose() }
[pscustomobject]@{
    apk = $apk.FullName
    sha256 = (Get-FileHash -LiteralPath $apk.FullName).Hash.ToLowerInvariant()
    verified_scenario_files = $entries.Count
}
