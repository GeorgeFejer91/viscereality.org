#Requires -Version 7.6
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$AndroidSdk,
    [Parameter(Mandatory)][string]$JavaHome,
    [Parameter(Mandatory)][string]$Keystore,
    [Parameter(Mandatory)][string]$ReferenceApk,
    [Parameter(Mandatory)][string]$OutputRoot,
    [Parameter(Mandatory)][ValidatePattern('^[a-zA-Z_][a-zA-Z0-9_]*$')][string]$SigningPasswordEnvironment,
    [string]$KeyAlias = 'androiddebugkey'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
function Invoke-Checked([string]$Tool, [string[]]$Arguments) {
    & $Tool @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Tool failed: $([IO.Path]::GetFileName($Tool))" }
}
function Assert-OutsideGit([string]$Directory) {
    $null = & git -C $Directory rev-parse --show-toplevel 2>$null
    if ($LASTEXITCODE -eq 0) { throw 'Private key/output must be outside every Git checkout.' }
}
$AndroidSdk = (Resolve-Path -LiteralPath $AndroidSdk).Path
$JavaHome = (Resolve-Path -LiteralPath $JavaHome).Path
$Keystore = (Resolve-Path -LiteralPath $Keystore).Path
$ReferenceApk = (Resolve-Path -LiteralPath $ReferenceApk).Path
Assert-OutsideGit (Split-Path -Parent $Keystore)
if (![IO.Path]::IsPathFullyQualified($OutputRoot)) { throw 'OutputRoot must be absolute and private.' }
$outputParent = (Resolve-Path -LiteralPath (Split-Path -Parent $OutputRoot)).Path
Assert-OutsideGit $outputParent
if (Test-Path -LiteralPath $OutputRoot) { throw 'OutputRoot must be fresh; existing output is never overwritten.' }
if ([string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable($SigningPasswordEnvironment))) {
    throw 'Set the named signing-password environment variable privately before building.'
}
$tools = Join-Path $AndroidSdk 'build-tools/36.0.0'
$platform = Join-Path $AndroidSdk 'platforms/android-34/android.jar'
$aapt = Join-Path $tools 'aapt.exe'
$aapt2 = Join-Path $tools 'aapt2.exe'
$d8 = Join-Path $tools 'd8.bat'
$zipalign = Join-Path $tools 'zipalign.exe'
$apksigner = Join-Path $tools 'apksigner.bat'
$dexdump = Join-Path $tools 'dexdump.exe'
foreach ($tool in @($platform, $aapt, $aapt2, $d8, $zipalign, $apksigner, $dexdump, (Join-Path $JavaHome 'bin/javac.exe'))) {
    if (!(Test-Path -LiteralPath $tool -PathType Leaf)) { throw "Missing required tool: $tool" }
}
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$OutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path
$classes = Join-Path $OutputRoot 'classes'
$dex = Join-Path $OutputRoot 'dex'
New-Item -ItemType Directory -Path $classes, $dex | Out-Null
$sourceRoot = Join-Path $PSScriptRoot 'src/org/viscereality/atmungsraum/boot'
$sources = @('AndroidHost.java', 'BootPolicy.java', 'BootReceiver.java', 'SetupActivity.java', 'WatchPolicy.java', 'HandoffProvider.java', 'WatchService.java') | ForEach-Object { Join-Path $sourceRoot $_ }
$unsigned = Join-Path $OutputRoot 'unsigned.apk'
$aligned = Join-Path $OutputRoot 'aligned.apk'
$apk = Join-Path $OutputRoot 'atmungsraum-boot-helper.apk'
$previousJava = $env:JAVA_HOME
try {
    $env:JAVA_HOME = $JavaHome
    Invoke-Checked (Join-Path $JavaHome 'bin/javac.exe') (@('-encoding','UTF-8','--release','8','-classpath',$platform,'-d',$classes) + $sources)
    $classFiles = @(Get-ChildItem -LiteralPath $classes -Recurse -File -Filter '*.class' | ForEach-Object FullName)
    Invoke-Checked $d8 (@('--release','--min-api','29','--lib',$platform,'--output',$dex) + $classFiles)
    Invoke-Checked $aapt2 @('link','-I',$platform,'--manifest',(Join-Path $PSScriptRoot 'AndroidManifest.xml'),'-o',$unsigned)
    $archive = [IO.Compression.ZipFile]::Open($unsigned, [IO.Compression.ZipArchiveMode]::Update)
    try { [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive,(Join-Path $dex 'classes.dex'),'classes.dex') }
    finally { $archive.Dispose() }
    Invoke-Checked $zipalign @('-f','-P','16','4',$unsigned,$aligned)
    Invoke-Checked $apksigner @('sign','--ks',$Keystore,'--ks-key-alias',$KeyAlias,'--ks-pass',"env:$SigningPasswordEnvironment",'--key-pass',"env:$SigningPasswordEnvironment",'--out',$apk,$aligned)
    $signer = @(& $apksigner verify --verbose --print-certs $apk)
    if ($LASTEXITCODE -ne 0) { throw 'Signed helper verification failed.' }
    $referenceSigner = @(& $apksigner verify --print-certs $ReferenceApk)
    if ($LASTEXITCODE -ne 0) { throw 'Reference APK signature verification failed.' }
    $certificate = @($signer | Select-String '^Signer #1 certificate SHA-256 digest: (.+)$' | ForEach-Object { $_.Matches[0].Groups[1].Value })
    $referenceCertificate = @($referenceSigner | Select-String '^Signer #1 certificate SHA-256 digest: (.+)$' | ForEach-Object { $_.Matches[0].Groups[1].Value })
    if ($certificate.Count -ne 1 -or $referenceCertificate.Count -ne 1 -or $certificate[0] -cne $referenceCertificate[0]) { throw 'Helper must have the exact reference APK certificate.' }
    Invoke-Checked $zipalign @('-c','-P','16','4',$apk)
    $badging = @(& $aapt dump badging $apk)
    if ($LASTEXITCODE -ne 0) { throw 'Actual APK badging failed.' }
    $manifest = @(& $aapt2 dump xmltree $apk --file AndroidManifest.xml)
    if ($LASTEXITCODE -ne 0) { throw 'Actual APK manifest decoding failed.' }
    $permissions = @($badging | Select-String "^uses-permission: name='([^']+)'" | ForEach-Object { $_.Matches[0].Groups[1].Value } | Sort-Object)
    if (($permissions -join ',') -cne 'android.permission.FOREGROUND_SERVICE,android.permission.FOREGROUND_SERVICE_SPECIAL_USE,android.permission.RECEIVE_BOOT_COMPLETED,android.permission.SYSTEM_ALERT_WINDOW') { throw 'Unexpected actual APK permissions.' }
    if (!($badging -match "^package: name='org.viscereality.atmungsraum.boot' versionCode='3' versionName='3.0'") -or
        !($badging -match "^sdkVersion:'29'") -or !($badging -match "^targetSdkVersion:'34'") -or
        ($badging -match '^application-debuggable')) { throw 'Actual APK identity/SDK/debugging differs from helper contract.' }
    $actualXml = $manifest -join "`n"
    function Actual-Node([string]$Name) {
        $nodes = [regex]::Matches($actualXml, ('(?m)^([ ]*)E: ' + [regex]::Escape($Name) + '\b'))
        if ($nodes.Count -ne 1) { throw "Expected one actual manifest node: $Name" }
        $node = $nodes[0]; $indent = $node.Groups[1].Value.Length
        $remaining = $actualXml.Substring($node.Index + $node.Length)
        $next = [regex]::Match($remaining, ('(?m)^[ ]{0,' + $indent + '}E:'))
        if ($next.Success) { return $remaining.Substring(0,$next.Index) }
        return $remaining
    }
    function Assert-ActualMetadata([string]$Name,[int]$Expected) {
        $matched = 0
        foreach ($node in [regex]::Matches($actualXml, '(?m)^([ ]*)E: meta-data\b')) {
            $indent = $node.Groups[1].Value.Length
            $remaining = $actualXml.Substring($node.Index + $node.Length)
            $next = [regex]::Match($remaining, ('(?m)^[ ]{0,' + $indent + '}E:'))
            $block = if ($next.Success) { $remaining.Substring(0,$next.Index) } else { $remaining }
            if ($block -match ('android:name[^\r\n]*"' + [regex]::Escape($Name) + '"')) {
                $matched++
                $hex = '{0:x}' -f $Expected
                if ($block -notmatch ('(?m)android:value(?:\(0x[0-9a-fA-F]+\))?[ \t]*=[ \t]*(?:' + $Expected + '|(?:\(type[ \t]+0x0*10\)[ \t]*)?0x0*' + $hex + ')[ \t]*\r?$')) {
                    throw "Actual metadata capability differs: $Name"
                }
            }
        }
        if ($matched -ne 1) { throw "Expected exactly one actual metadata capability: $Name" }
    }
    $provider = Actual-Node 'provider'; $service = Actual-Node 'service'
    $permission = Actual-Node 'permission'; $receiver = Actual-Node 'receiver'
    # aapt2 emits text booleans or typed hex depending on its version. Match
    # the complete value: integer flags, quoted strings and prefix matches deny.
    $attributeId = '(?:\(0x[0-9a-fA-F]+\))?[ \t]*=[ \t]*'
    $trueValue = '(?:true|(?:\(type[ \t]+0x0*12\)[ \t]*)?0x[fF]{8})[ \t]*\r?$'
    $falseValue = '(?:false|(?:\(type[ \t]+0x0*12\)[ \t]*)?0x0+)[ \t]*\r?$'
    $enumValue = '(?:\(type[ \t]+0x0*11\)[ \t]*)?'
    $exportedTrue = '(?m)android:exported' + $attributeId + $trueValue
    $exportedFalse = '(?m)android:exported' + $attributeId + $falseValue
    if ($provider -notmatch 'android:name[^\r\n]*\.HandoffProvider' -or
        $provider -notmatch 'android:authorities[^\r\n]*"org\.viscereality\.atmungsraum\.boot\.handoff"' -or
        $provider -notmatch 'android:permission[^\r\n]*"org\.viscereality\.atmungsraum\.boot\.permission\.UPDATE_HANDOFF"' -or
        $provider -notmatch $exportedTrue -or
        $service -notmatch 'android:name[^\r\n]*\.WatchService' -or
        $service -notmatch $exportedFalse -or
        $service -notmatch ('(?m)android:foregroundServiceType' + $attributeId + $enumValue + '0x0*40000000[ \t]*\r?$') -or
        $service -notmatch 'android\.app\.PROPERTY_SPECIAL_USE_FGS_SUBTYPE' -or
        $permission -notmatch 'android:name[^\r\n]*"org\.viscereality\.atmungsraum\.boot\.permission\.UPDATE_HANDOFF"' -or
        $permission -notmatch ('(?m)android:protectionLevel' + $attributeId + $enumValue + '0x0*2[ \t]*\r?$') -or
        $receiver -notmatch $exportedFalse -or
        $actualXml -notmatch 'org\.viscereality\.atmungsraum\.boot_protocol') { throw 'Actual helper3 IPC/FGS closure differs.' }
    Assert-ActualMetadata 'org.viscereality.atmungsraum.boot_protocol' 2
    Assert-ActualMetadata 'org.viscereality.atmungsraum.monitor_protocol' 3
    $dexText = @(& $dexdump -d (Join-Path $dex 'classes.dex'))
    if ($LASTEXITCODE -ne 0) { throw 'Actual packaged DEX inspection failed.' }
    $definitions = @($dexText | Select-String "Class descriptor\s*: '([^']+)'" | ForEach-Object { $_.Matches[0].Groups[1].Value })
    foreach ($owner in @('AndroidHost','BootPolicy','BootReceiver','SetupActivity','WatchPolicy','HandoffProvider','WatchService')) {
        if (@($definitions | Where-Object { $_ -ceq "Lorg/viscereality/atmungsraum/boot/$owner;" }).Count -ne 1) { throw "Missing/duplicate DEX owner: $owner" }
    }
    foreach ($definition in $definitions) {
        if ($definition -cnotmatch '^Lorg/viscereality/atmungsraum/boot/(AndroidHost|BootPolicy|BootReceiver|SetupActivity|WatchPolicy|HandoffProvider|WatchService)(\$[^;]+)?;$') {
            throw "Unexpected actual DEX owner: $definition"
        }
    }
    $archive = [IO.Compression.ZipFile]::OpenRead($apk)
    try {
        $entries = @($archive.Entries | ForEach-Object FullName)
        if (@($entries | Where-Object { $_ -notin @('AndroidManifest.xml','classes.dex','resources.arsc') -and $_ -notlike 'META-INF/*' }).Count) { throw 'Unexpected APK payload.' }
        # aapt2 emits an empty table even for this resource-free application.
        if ($archive.GetEntry('resources.arsc').Length -ne 40) { throw 'Unexpected helper resource table.' }
        $stream = $archive.GetEntry('classes.dex').Open()
        try { $packagedDexHash = (Get-FileHash -InputStream $stream).Hash.ToLowerInvariant() } finally { $stream.Dispose() }
        if ($packagedDexHash -cne (Get-FileHash -LiteralPath (Join-Path $dex 'classes.dex')).Hash.ToLowerInvariant()) { throw 'Packaged DEX differs from inspected DEX.' }
    } finally { $archive.Dispose() }
    $badging | Set-Content -LiteralPath (Join-Path $OutputRoot 'actual-badging.txt')
    $manifest | Set-Content -LiteralPath (Join-Path $OutputRoot 'actual-manifest.txt')
    $signer | Set-Content -LiteralPath (Join-Path $OutputRoot 'signer-inspection.txt')
    $definitions | Set-Content -LiteralPath (Join-Path $OutputRoot 'dex-classes.txt')
    $receipt = [ordered]@{
        apk = $apk; sha256 = (Get-FileHash -LiteralPath $apk).Hash.ToLowerInvariant()
        certificate_sha256 = $certificate[0]; reference_sha256 = (Get-FileHash -LiteralPath $ReferenceApk).Hash.ToLowerInvariant()
        dex_sha256 = $packagedDexHash; permissions = $permissions; dex_classes = $definitions
        entries = $entries; min_sdk = 29; target_sdk = 34; compile_sdk = 34
        source_files = @(@($sources) + @(Join-Path $PSScriptRoot 'AndroidManifest.xml') + @($PSCommandPath) | ForEach-Object {
            [ordered]@{ name = [IO.Path]::GetFileName($_); sha256 = (Get-FileHash -LiteralPath $_).Hash.ToLowerInvariant() }
        })
        installed = $false; hardware_boot_confirmed = $false
    }
    $receipt | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $OutputRoot 'verification.json')
    [pscustomobject]@{ apk = $apk; sha256 = $receipt.sha256; certificate_sha256 = $certificate[0] }
} finally { $env:JAVA_HOME = $previousJava }
