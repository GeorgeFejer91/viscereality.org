[CmdletBinding()]
param([Parameter(Mandatory)][string]$JavaHome)
$ErrorActionPreference = 'Stop'
$testOutput = Join-Path ([IO.Path]::GetTempPath()) ('atmungsraum-helper-policy-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testOutput | Out-Null
$sources = @('BootPolicy.java','WatchPolicy.java') | ForEach-Object { Join-Path $PSScriptRoot ('src/org/viscereality/atmungsraum/boot/' + $_) }
$test = Join-Path $PSScriptRoot '../../tests/BootHelperPolicyTest.java'
& (Join-Path $JavaHome 'bin/javac.exe') --release 8 -encoding UTF-8 -d $testOutput @sources $test (Join-Path $PSScriptRoot 'tests/WatchPolicyTest.java') (Join-Path $PSScriptRoot 'tests/MonitorPolicyTest.java')
if ($LASTEXITCODE -ne 0) { throw 'Policy compilation failed.' }
& (Join-Path $JavaHome 'bin/java.exe') -cp $testOutput org.viscereality.atmungsraum.boot.BootHelperPolicyTest
if ($LASTEXITCODE -ne 0) { throw 'Policy test failed.' }

& (Join-Path $JavaHome 'bin/java.exe') -cp $testOutput org.viscereality.atmungsraum.boot.WatchPolicyTest
if ($LASTEXITCODE -ne 0) { throw 'Update watch policy test failed.' }
& (Join-Path $JavaHome 'bin/java.exe') -cp $testOutput org.viscereality.atmungsraum.boot.MonitorPolicyTest
if ($LASTEXITCODE -ne 0) { throw 'Observed update monitor test failed.' }
