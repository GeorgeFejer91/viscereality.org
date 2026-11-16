$ErrorActionPreference='Stop'
$builder=Get-Content -LiteralPath (Join-Path $PSScriptRoot '../build.ps1') -Raw
$start=$builder.IndexOf('    $actualXml =')
$end=$builder.IndexOf('    $dexText =', $start)
if($start-lt0-or$end-lt$start){throw 'Builder manifest gate missing'}
$gate=[scriptblock]::Create($builder.Substring($start,$end-$start))
$fixture=@'
E: manifest
  E: permission
    A: android:name="org.viscereality.atmungsraum.boot.permission.UPDATE_HANDOFF"
    A: android:protectionLevel=(type 0x11)0x2
  E: application
    E: meta-data
      A: android:name="org.viscereality.atmungsraum.boot_protocol"
      A: android:value=(type 0x10)0x2
    E: provider
      A: android:name=".HandoffProvider"
      A: android:authorities="org.viscereality.atmungsraum.boot.handoff"
      A: android:permission="org.viscereality.atmungsraum.boot.permission.UPDATE_HANDOFF"
      A: android:exported=(type 0x12)0xffffffff
    E: service
      A: android:name=".WatchService"
      A: android:exported=(type 0x12)0x0
      A: android:foregroundServiceType=(type 0x11)0x40000000
      E: property
        A: android:name="android.app.PROPERTY_SPECIAL_USE_FGS_SUBTYPE"
    E: receiver
      A: android:name=".BootReceiver"
      A: android:exported=(type 0x12)0x0
'@
$manifest=$fixture -split "`n"
& $gate
$checks=1
# Actual aapt2 dump from the first signed helper2 candidate (manifest-only, no private data).
$actualFixture=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'helper2-manifest-aapt2.txt') -Raw
$manifest=$actualFixture -split "`n"
& $gate
$checks++
foreach($mutation in @(
    @('android:protectionLevel=(type 0x11)0x2','android:protectionLevel=(type 0x11)0x0'),
    @('android:foregroundServiceType=(type 0x11)0x40000000','android:foregroundServiceType=(type 0x11)0x1'),
    @('android:authorities="org.viscereality.atmungsraum.boot.handoff"','android:authorities="other"'),
    @('android:permission="org.viscereality.atmungsraum.boot.permission.UPDATE_HANDOFF"','android:permission="other"'),
    @('android:exported=(type 0x12)0xffffffff','android:exported=(type 0x12)0x0'),
    @('E: receiver','E: other')
)){
    $manifest=$fixture.Replace($mutation[0],$mutation[1]) -split "`n"
    $denied=$false
    try { & $gate } catch { $denied=$true }
    if(!$denied){throw 'Builder actual-manifest gate accepted mutation'}
    $checks++
}
# Accept legacy typed output and current untyped output, including enum/false padding.
foreach($formatted in @(
    $fixture.Replace('0x2','0x00000002').Replace('(type 0x12)0x0','(type 0x12)0x00000000'),
    $actualFixture.Replace('0x00000002','0x2').Replace('=false','=(type 0x12)0x00000000').Replace('=true','=(type 0x12)0xffffffff'),
    $actualFixture.Replace('=0x40000000','=0x0000000040000000')
)){
    $manifest=$formatted -split "`n"
    & $gate
    $checks++
}
foreach($mutation in @(
    @('protectionLevel(0x01010009)=0x00000002','protectionLevel(0x01010009)=0x00000020'),
    @('protectionLevel(0x01010009)=0x00000002','protectionLevel(0x01010009)=0x00000012'),
    @('protectionLevel(0x01010009)=0x00000002','protectionLevel(0x01010009)=0x00000002garbage'),
    @('protectionLevel(0x01010009)=0x00000002','protectionLevel(0x01010009)=(type 0x03)0x00000002'),
    @('exported(0x01010010)=true','exported(0x01010010)=false'),
    @('exported(0x01010010)=false','exported(0x01010010)=true'),
    @('exported(0x01010010)=true','exported(0x01010010)="true"'),
    @('exported(0x01010010)=true','exported(0x01010010)=(type 0x11)0xffffffff'),
    @('exported(0x01010010)=false','exported(0x01010010)=(type 0x11)0x0'),
    @('exported(0x01010010)=false','exported(0x01010010)=falseextra'),
    @('foregroundServiceType(0x01010599)=0x40000000','foregroundServiceType(0x01010599)=0x40000001'),
    @('foregroundServiceType(0x01010599)=0x40000000','foregroundServiceType(0x01010599)=(type 0x03)0x40000000')
)){
    $manifest=$actualFixture.Replace($mutation[0],$mutation[1]) -split "`n"
    $denied=$false
    try { & $gate } catch { $denied=$true }
    if(!$denied){throw ('Actual decoded manifest gate accepted invalid value: ' + $mutation[1])}
    $checks++
}
Write-Output "PASS helper builder actual-manifest gate: $checks inert fixtures; no APK created"
