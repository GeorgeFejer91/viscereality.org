#Requires -Version 7.6
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$QuestRepo,
    [string]$ExperimentAppSpec,
    [string]$ExperimentFeatureLock,
    [string]$PrivateAssetProviderManifest,
    [Parameter(Mandatory)][ValidatePattern('^[a-z0-9][a-z0-9-]{0,39}$')][string]$RunName,
    [string]$OpenXrLoader,
    [string]$Keystore,
    [string]$RetainedCargoTargetDirectory,
    [ValidateRange(1, 2100000000)][int]$VersionCode = 1,
    [string]$VersionName = '0.1.0',
    [switch]$Developer,
    [switch]$BootAutostart,
    [switch]$MetaPlatformUpdate,
    [string]$MetaPlatformAppId,
    [string]$MetaRequiredVersionManifestUrl,
    [switch]$MetaUpdateHelper,
    [string]$MetaPlatformSdkRoot,
    [string]$MetaPlatformHeadersSha256,
    [string]$MetaPlatformLoaderSha256,
    [string]$MetaPlatformArchiveSha256,
    [switch]$ResolveOnly,
    [switch]$BundledAssets
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$metaInputs = @($MetaPlatformAppId, $MetaPlatformSdkRoot, $MetaPlatformHeadersSha256,
    $MetaPlatformLoaderSha256, $MetaPlatformArchiveSha256, $MetaRequiredVersionManifestUrl)
if ($MetaPlatformUpdate) {
    if (@($metaInputs | Where-Object { [string]::IsNullOrWhiteSpace($_) }).Count -gt 0 -or
        $MetaPlatformAppId -cnotmatch '^[1-9][0-9]{0,19}$' -or
        @(@($MetaPlatformHeadersSha256, $MetaPlatformLoaderSha256, $MetaPlatformArchiveSha256) |
            Where-Object { $_ -cnotmatch '^[0-9a-f]{64}$' }).Count -gt 0) {
        throw 'MetaPlatformUpdate requires an explicit canonical MetaPlatformAppId, required-version manifest URL, SDK root and three SHA256 digests.'
    }
    if ($MetaRequiredVersionManifestUrl -cne 'https://viscereality.org/atmungsraum/quest-release.json') {
        throw 'MetaPlatformUpdate requires the canonical Atmungsraum required-version manifest URL.'
    }
    $MetaPlatformSdkRoot = (Resolve-Path -LiteralPath $MetaPlatformSdkRoot).Path
} elseif ($MetaUpdateHelper -or @($metaInputs | Where-Object { ![string]::IsNullOrWhiteSpace($_) }).Count -gt 0) {
    throw 'Meta SDK inputs require explicit -MetaPlatformUpdate.'
}

$QuestRepo = (Resolve-Path -LiteralPath $QuestRepo).Path
if ($Developer) {
    if (!$Keystore -or !(Test-Path -LiteralPath $Keystore -PathType Leaf)) {
        throw 'Developer requires an explicit existing -Keystore; retain the same developer signing key for later uploads.'
    }
    if (!$PSBoundParameters.ContainsKey('VersionCode')) {
        throw 'Developer requires an explicit -VersionCode greater than every previous developer upload.'
    }
    $Keystore = (Resolve-Path -LiteralPath $Keystore).Path
    $keyRepository = @(& git -C (Split-Path -Parent $Keystore) rev-parse --show-toplevel 2>$null)
    if ($LASTEXITCODE -eq 0) {
        throw 'Developer keystore must be outside Git, separate from the rehearsal/default build key.'
    }
}
if ($BundledAssets) {
    if ($ExperimentAppSpec -or $ExperimentFeatureLock -or $PrivateAssetProviderManifest) {
        throw 'Use the bundled Experimenter inputs or supply all three custom inputs.'
    }
    $bundledRoot = Join-Path $PSScriptRoot 'build-assets'
    $ExperimentAppSpec = Join-Path $bundledRoot 'experimenter.app.json'
    $PrivateAssetProviderManifest = Join-Path $bundledRoot 'provider/provider.json'
    if (!$OpenXrLoader) { $OpenXrLoader = Join-Path $bundledRoot 'arm64-v8a/libopenxr_loader.so' }
    $inherited = @(Get-Content -Raw -LiteralPath (Join-Path $bundledRoot 'experimenter.env.json') | ConvertFrom-Json)
} elseif (!$ExperimentAppSpec -or !$ExperimentFeatureLock -or !$PrivateAssetProviderManifest) {
    throw 'Supply an Experimenter app spec, feature lock and provider manifest, or use -BundledAssets.'
}

$ExperimentAppSpec = (Resolve-Path -LiteralPath $ExperimentAppSpec).Path
$PrivateAssetProviderManifest = (Resolve-Path -LiteralPath $PrivateAssetProviderManifest).Path
if (!$ResolveOnly) {
    if (!$OpenXrLoader) { throw 'A build needs the ARM64 OpenXR loader.' }
    $OpenXrLoader = (Resolve-Path -LiteralPath $OpenXrLoader).Path
    if ($Keystore) { $Keystore = (Resolve-Path -LiteralPath $Keystore).Path }
    if ($BundledAssets) {
        $lslSource = Join-Path $bundledRoot 'arm64-v8a/liblsl.so'
        $lslTarget = Join-Path $QuestRepo 'local-artifacts/liblsl-android/arm64-v8a/liblsl.so'
        if (!(Test-Path -LiteralPath $lslTarget)) {
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $lslTarget) | Out-Null
            [IO.File]::Copy($lslSource, $lslTarget, $false)
        }
        if ((Get-FileHash -LiteralPath $lslTarget).Hash -ne (Get-FileHash -LiteralPath $lslSource).Hash) {
            throw 'The Rusty Quest checkout has a different staged liblsl.so. Use a clean build checkout.'
        }
    }
}

if (!$BundledAssets) {
    $baseLock = Get-Content -Raw -LiteralPath $ExperimentFeatureLock | ConvertFrom-Json
    if ($baseLock.schema -cne 'rusty.quest.native_app_feature_lock.v1' -or
        (Get-FileHash -LiteralPath $ExperimentAppSpec).Hash -ne $baseLock.app_spec_sha256) {
        throw 'The Experimenter feature lock must bind the exact app spec.'
    }
    $inherited = @($baseLock.build_inputs.env | Where-Object { $_.PSObject.Properties['source'] -and $_.source -like 'feature:*:inherited-environment' })
}
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
if ($BundledAssets) {
    $payload = $app.payloads[0]
    $payload.data_dir = (Resolve-Path -LiteralPath (Join-Path $bundledRoot $payload.data_dir)).Path
    $payload.shader = (Resolve-Path -LiteralPath (Join-Path $bundledRoot $payload.shader)).Path
    $payload.mask_texture.path = (Resolve-Path -LiteralPath (Join-Path $bundledRoot $payload.mask_texture.path)).Path
    foreach ($entry in $payload.build_env | Where-Object { $_.value -clike 'particles/*' }) {
        $entry.value = (Resolve-Path -LiteralPath (Join-Path $bundledRoot $entry.value)).Path
    }
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
$boxLoop = $catalog.box_loop
$boxLoopPath = Join-Path $audio 'audio/box-loop.json'
if ($boxLoop.path -cne 'audio/box-loop.json' -or $boxLoop.cycle_frames -ne 705600 -or
    $boxLoop.main_seconds -ne 900 -or $boxLoop.sha256 -cnotmatch '^[0-9a-f]{64}$' -or
    $boxLoop.sha256 -cne (Get-FileHash -LiteralPath $boxLoopPath).Hash.ToLowerInvariant()) {
    throw 'The staged breathing setup loop must bind the exact 16-second cycle and 900-second main contract.'
}
$boxLoopHash = $boxLoop.sha256

$featureId = 'audio.atmungsraum_scenarios'
$featureDir = Join-Path $runRoot 'features'
Copy-Item -LiteralPath (Join-Path $QuestRepo 'fixtures/native-app-features') -Destination $featureDir -Recurse
$assetInput = [IO.Path]::GetRelativePath($QuestRepo, $audio).Replace('\','/')
$feature = [ordered]@{
    schema = 'rusty.quest.native_app_feature.v1'
    feature_id = $featureId
    module_path = 'audio/atmungsraum-scenarios'
    module_kind = 'app-owned-scenario-library'
    settings_surface = @{ authority = 'rusty.quest.native_app_settings.v1'; adapter = 'build_input' }
    owner_lane = 'atmungsraum'
    status = 'public-app-feature'
    description = 'Atmungsraum authored scenario JSON and replaceable cue files.'
    provides = @($featureId)
    depends_on = @('ui.experiment_session_command_gateway')
    incompatible_with = @()
    exclusive_groups = @{}
    android_manifest = @{ permissions = @(); uses_features = @(); activities = @(); services = @(); queries = @() }
    runtime_profile = @{ set = @{}; clear_families = @(); expected_render_modes = @() }
    build_inputs = @{
        env = @(
            @{ name = 'RUSTY_QUEST_NATIVE_RENDERER_SCENARIO_CATALOG_SHA256'; value = $catalogHash }
            @{ name = 'RUSTY_QUEST_NATIVE_RENDERER_BOX_LOOP_SHA256'; value = $boxLoopHash }
            @{ name = 'RUSTY_QUEST_NATIVE_RENDERER_EXPERIMENT_DEFAULT_REMOTE'; value = 'true' }
        )
        assets = @($assetInput)
        shaders = @()
    }
    markers = @{ required = @(); forbidden = @() }
    validation = @{ quick = @(); standard = @(); device = @() }
    public_private_boundary = 'Packages the public Atmungsraum scenario library without signing or device data.'
}
$feature | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $featureDir "$featureId.feature.json") -Encoding utf8NoBOM
$scenarioSourceDir = 'apps/native-renderer-android/panel-modules/condition-audio/src/main/java/io/github/mesmerprism/rustyquest/native_renderer'
$scenarioSources = @('ScenarioAudioPackage.java', 'ScenarioAudioAndroidTracks.java', 'BreathingSetupAudio.java', 'BreathingSetupAudioContract.java') | ForEach-Object { "$scenarioSourceDir/$_" }
foreach ($source in $scenarioSources) {
    if (!(Test-Path -LiteralPath (Join-Path $QuestRepo $source) -PathType Leaf)) {
        throw 'Apply the Atmungsraum native scenario source before assembly.'
    }
}
$panelDescriptor = Join-Path $featureDir 'ui/breath-composition-panel/ui.breath_composition_control_panel.feature.json'
$panel = Get-Content -Raw -LiteralPath $panelDescriptor | ConvertFrom-Json -AsHashtable
$audioModule = @($panel.panel_composition.modules | Where-Object { $_.module_id -ceq 'condition-audio-runtime' })
if ($audioModule.Count -ne 1) { throw 'The selected panel must contain the condition audio owner.' }
$audioModule[0].source_files = @(@($audioModule[0].source_files) + $scenarioSources | Select-Object -Unique)
$panel | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $panelDescriptor -Encoding utf8NoBOM

$app.app_id = if ($Developer) { 'atmungsraum_developer' } else { 'atmungsraum_quest2' }
$app.package_policy = 'app-specific-package'
$app.package_name = if ($Developer) { 'org.viscereality.atmungsraum.developer' } else { 'org.viscereality.atmungsraum.rehearsal' }
$app.application_label = if ($Developer) { 'Atmungsraum Developer' } else { 'Atmungsraum Quest 2' }
$app.android_debuggable = !$Developer
if ($Developer) {
    $app.android_target_sdk_version = 34
    $app.android_install_location = 'auto'
} else {
    [void]$app.Remove('android_target_sdk_version')
    [void]$app.Remove('android_install_location')
}
$app.experiment_breathing_setup_enabled = $true
$app.experiment_headset_setup_enabled = $false
$app.experiment_phone_pairing_enabled = [bool]$Developer
$app.experiment_panel_initially_closed = [bool]$Developer
$app.experiment_gateway_origins = @('https://viscereality.org')
$app.quest_supported_devices = @('quest2')
$handInputFeatures = @(
    'input.simultaneous_hands_and_controllers', 'hand_mesh_live_input',
    'hand_mesh_visual', 'input.tracked_hand_pointer', 'hand_anchor_particles'
)
$app.requested_features = @(@($app.requested_features | Where-Object {
    $_ -cnotin $handInputFeatures
}) + 'input.controllers_only' | Select-Object -Unique)
$app.denied_features = @(@($app.denied_features) + $handInputFeatures | Select-Object -Unique)
$app.declared_manifest.permissions = @($app.declared_manifest.permissions | Where-Object {
    $_ -cne 'com.oculus.permission.HAND_TRACKING'
})
$app.declared_manifest.uses_features = @($app.declared_manifest.uses_features | Where-Object {
    $_ -cne 'oculus.software.handtracking'
})
foreach ($setting in @(
    'hand_adapter.enabled', 'hand_mesh.graft_copies.enabled', 'hand_mesh.real_hands.visible',
    'hand_mesh.visual.diagnostic.enabled', 'hand_mesh.visual.wireframe.enabled',
    'simultaneous_hands_controllers.enabled'
)) {
    $app.runtime_profile.set["debug.rustyquest.native_renderer.$setting"] = 'false'
    $app.settings_assertions.required_values["native_renderer.$setting"] = 'false'
}
$app.runtime_profile.set['debug.rustyquest.native_renderer.hand_mesh.input.source'] = 'disabled'
$app.settings_assertions.required_values['native_renderer.hand_mesh.input.source'] = 'disabled'
$handInputModules = @('hand/live-input', 'hand/mesh', 'input/simultaneous-hands-and-controllers')
$app.settings_assertions.required_modules = @(@($app.settings_assertions.required_modules | Where-Object {
    $_ -cnotin $handInputModules
}) + 'input/controller-only' | Select-Object -Unique)
$app.settings_assertions.forbidden_modules = @(@($app.settings_assertions.forbidden_modules) + $handInputModules | Select-Object -Unique)
$app.expected_markers.required = @(@($app.expected_markers.required | Where-Object {
    $_ -cnotin @('handAdapterApplied=true', 'simultaneousHandsControllersSelected=true')
}) + @('nativeControllerOnlyInput=true', 'liveHandTrackingExtensionEnabled=false') | Select-Object -Unique)
$app.expected_markers.forbidden = @(@($app.expected_markers.forbidden) + 'liveHandTrackingExtensionEnabled=true' | Select-Object -Unique)
$app.declared_manifest.permissions = @($app.declared_manifest.permissions) + @(
    'android.permission.CHANGE_NETWORK_STATE',
    'android.permission.CHANGE_WIFI_STATE',
    'android.permission.NEARBY_WIFI_DEVICES'
) | Select-Object -Unique
$app.requested_features = @($app.requested_features) + @(
    $featureId,
    'ui.experiment_private_lan_data',
    'ui.experiment_scene_recenter',
    'ui.experiment_wifi_setup'
)
if ($BootAutostart) {
    $app.requested_features += 'ui.experiment_boot_autostart'
    $app.declared_manifest.receivers = @(@($app.declared_manifest.receivers) + 'NativeRendererBootReceiver' | Select-Object -Unique)
    $app.declared_manifest.permissions = @(@($app.declared_manifest.permissions) + @(
        'android.permission.RECEIVE_BOOT_COMPLETED', 'android.permission.SYSTEM_ALERT_WINDOW'
    ) | Select-Object -Unique)
}
if ('ui.native_mesh_panel' -cnotin $app.requested_features -and
    'ui.native_mesh_panel.original_product' -cnotin $app.requested_features) {
    $app.requested_features += 'ui.native_mesh_panel.original_product'
}
$specPath = Join-Path $runRoot 'atmungsraum.app.json'
$helperPermission = 'org.viscereality.atmungsraum.boot.permission.UPDATE_HANDOFF'
$app.permission_allowlist = @($app.permission_allowlist | Where-Object { $_ -cne $helperPermission })
$app.declared_manifest.permissions = @($app.declared_manifest.permissions | Where-Object { $_ -cne $helperPermission })
if ($MetaUpdateHelper) {
    $app.permission_allowlist = @(@($app.permission_allowlist) + $helperPermission | Select-Object -Unique)
    $app.declared_manifest.permissions = @(@($app.declared_manifest.permissions) + $helperPermission | Select-Object -Unique)
}
if ($MetaPlatformUpdate) {
    $app.experiment_meta_update_enabled = $true
    $app.meta_platform_app_id = $MetaPlatformAppId
    $app.meta_required_version_manifest_url = $MetaRequiredVersionManifestUrl
    $app.experiment_meta_update_helper_enabled = [bool]$MetaUpdateHelper
} else {
    $app.Remove('experiment_meta_update_enabled')
    $app.Remove('meta_platform_app_id')
    $app.Remove('meta_required_version_manifest_url')
    $app.Remove('experiment_meta_update_helper_enabled')
}
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
        $catalogHash -cnotin @($lock.build_inputs.env | Where-Object name -CEQ 'RUSTY_QUEST_NATIVE_RENDERER_SCENARIO_CATALOG_SHA256' | ForEach-Object value) -or
        $boxLoopHash -cnotin @($lock.build_inputs.env | Where-Object name -CEQ 'RUSTY_QUEST_NATIVE_RENDERER_BOX_LOOP_SHA256' | ForEach-Object value)) {
        throw 'The resolved native feature lock must bind the audio owners, catalog and breathing loop digests.'
    }
    $manifestPath = Join-Path (Split-Path -Parent $lockPath) 'AndroidManifest.xml'
    [xml]$manifest = Get-Content -Raw -LiteralPath $manifestPath
    $androidNamespace = 'http://schemas.android.com/apk/res/android'
    $metadata = @{}
    foreach ($entry in $manifest.SelectNodes('/manifest/application/meta-data')) {
        $metadata[$entry.GetAttribute('name', $androidNamespace)] = $entry.GetAttribute('value', $androidNamespace)
    }
    $bootSelected = 'ui.experiment_boot_autostart' -cin @($lock.selected_feature_ids)
    $bootReceivers = @($manifest.SelectNodes('/manifest/application/receiver') | Where-Object {
        $_.GetAttribute('name', $androidNamespace) -ceq 'io.github.mesmerprism.rustyquest.native_renderer.NativeRendererBootReceiver'
    })
    if ($bootSelected -ne [bool]$BootAutostart -or
        ('NativeRendererBootReceiver' -cin @($lock.android_manifest.receivers)) -ne [bool]$BootAutostart -or
        $bootReceivers.Count -ne [int][bool]$BootAutostart) {
        throw 'The resolved boot selection and own-app receiver must match the explicit -BootAutostart option.'
    }
    if ($BootAutostart) {
        if ($metadata['rusty.quest.experiment.boot_autostart'] -cne 'true' -or
            $bootReceivers[0].GetAttribute('exported', $androidNamespace) -cne 'false' -or
            @($bootReceivers[0].SelectNodes('intent-filter/action') | Where-Object {
                $_.GetAttribute('name', $androidNamespace) -ceq 'android.intent.action.BOOT_COMPLETED'
            }).Count -ne 1) {
            throw 'Boot autostart must bind the non-exported system boot receiver and selection metadata.'
        }
        foreach ($permission in @('android.permission.RECEIVE_BOOT_COMPLETED', 'android.permission.SYSTEM_ALERT_WINDOW')) {
            if ($permission -cnotin @($lock.android_manifest.permissions) -or
                @($manifest.SelectNodes('/manifest/uses-permission') | Where-Object {
                    $_.GetAttribute('name', $androidNamespace) -ceq $permission
                }).Count -ne 1) {
                throw "Boot autostart is missing its declared permission: $permission"
            }
        }
    } elseif ($metadata.ContainsKey('rusty.quest.experiment.boot_autostart') -or
        $manifest.OuterXml.Contains('android.intent.action.BOOT_COMPLETED')) {
        throw 'Default assembly must omit boot autostart metadata and the system boot action.'
    }
    if ($manifest.manifest.GetAttribute('package') -cne $app.package_name -or
        $manifest.manifest.application.GetAttribute('label', $androidNamespace) -cne $app.application_label -or
        $manifest.manifest.application.GetAttribute('debuggable', $androidNamespace) -cne (!$Developer).ToString().ToLowerInvariant()) {
        throw 'The native manifest must bind the selected Atmungsraum package, label and debuggability.'
    }
    $expectedTargetSdk = if ($Developer) { '34' } else { '35' }
    $installLocationMatches = if ($Developer) {
        $manifest.manifest.GetAttribute('installLocation', $androidNamespace) -ceq 'auto'
    } else {
        !$manifest.manifest.HasAttribute('installLocation', $androidNamespace)
    }
    if ($manifest.manifest.'uses-sdk'.GetAttribute('minSdkVersion', $androidNamespace) -cne '29' -or
        $manifest.manifest.'uses-sdk'.GetAttribute('targetSdkVersion', $androidNamespace) -cne $expectedTargetSdk -or
        !$installLocationMatches) {
        throw 'The native manifest must bind the selected Atmungsraum SDK and install location.'
    }
    if ($metadata['rusty.quest.experiment.breathing_setup.enabled'] -cne 'true') {
        throw 'The native manifest must explicitly select the shared breathing setup.'
    }
    if ($Developer) {
        $closed = @($lock.build_inputs.env | Where-Object name -CEQ 'RUSTY_QUEST_NATIVE_RENDERER_PANEL_INITIALLY_CLOSED')
        if ($metadata['rusty.quest.experiment.phone_pairing.enabled'] -cne 'true' -or
            $lock.android_manifest.experiment_phone_pairing_enabled -cne $true -or
            $lock.android_manifest.experiment_panel_initially_closed -cne $true -or
            $closed.Count -ne 1 -or $closed[0].value -cne 'true' -or
            $lock.android_manifest.experiment_headset_setup_enabled -cne $false -or
            @($lock.phone_pairing_sources).Count -ne 5 -or
            @($lock.headset_setup_sources).Count -ne 0 -or
            @($lock.debug_experiment_operator_sources).Count -ne 0 -or
            $manifest.OuterXml -match 'debug-experiment-control|\.headset-setup') {
            throw 'Developer must select closed startup and button-only phone pairing without machine provisioning or a debug operator.'
        }
        foreach ($source in $lock.phone_pairing_sources) {
            if ($source.sha256 -cne (Get-FileHash -LiteralPath (Join-Path $QuestRepo $source.path)).Hash.ToLowerInvariant()) {
                throw 'The selected phone pairing source digest differs from the native implementation.'
            }
        }
    }
    if ($metadata['com.oculus.supportedDevices'] -cne 'quest2' -or
        $metadata['rusty.quest.experiment.command_gateway.origins'] -cne 'https://viscereality.org') {
        throw 'The native manifest must target Quest 2 and admit the exact Atmungsraum origin.'
    }
    if ($metadata['rusty.quest.native_renderer.original_panel_native_host'] -cne 'true' -or
        $metadata['rusty.quest.experiment.command_gateway.enabled'] -cne 'true') {
        throw 'The native manifest must select the original panel host and the gateway for Remote control.'
    }
    $remoteDefault = @($lock.build_inputs.env | Where-Object name -CEQ 'RUSTY_QUEST_NATIVE_RENDERER_EXPERIMENT_DEFAULT_REMOTE')
    if ($remoteDefault.Count -ne 1 -or $remoteDefault[0].value -cne 'true') {
        throw 'The native feature lock must select Remote as the default control mode.'
    }
    if ($metadata['rusty.quest.experiment.command_gateway.private_lan_data'] -cne 'true') {
        throw 'The native manifest must select encrypted LAN sensor and recording readback.'
    }
    if ($metadata['rusty.quest.experiment.command_gateway.scene_recenter'] -cne 'true') {
        throw 'The native manifest must select enrolled scene recentering.'
    }
    if ($metadata['rusty.quest.experiment.command_gateway.wifi_setup'] -cne 'true') {
        throw 'The native manifest must select enrolled Bluetooth Wi-Fi setup.'
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
    if ($MetaPlatformUpdate) {
        $buildArgs += @('-EnableMetaPlatformUpdate', '-MetaPlatformSdkRoot', $MetaPlatformSdkRoot,
            '-MetaPlatformHeadersSha256', $MetaPlatformHeadersSha256,
            '-MetaPlatformLoaderSha256', $MetaPlatformLoaderSha256,
            '-MetaPlatformArchiveSha256', $MetaPlatformArchiveSha256)
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
$closureProbe = Join-Path $QuestRepo 'tools/checks/Test-ExperimentSessionApkClosure.ps1'
if (!(Test-Path -LiteralPath $closureProbe -PathType Leaf)) {
    throw 'Selected native source is missing the actual-APK cold-start closure probe.'
}
& pwsh -NoProfile -File $closureProbe -ApkPath $apk.FullName -OutputRoot (Join-Path $runRoot 'closure-host')
if ($LASTEXITCODE -ne 0) { throw 'Signed APK cold-start closure validation failed.' }
[pscustomobject]@{
    apk = $apk.FullName
    sha256 = (Get-FileHash -LiteralPath $apk.FullName).Hash.ToLowerInvariant()
    verified_scenario_files = $entries.Count
}
