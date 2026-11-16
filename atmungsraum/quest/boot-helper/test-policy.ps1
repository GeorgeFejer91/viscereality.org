[CmdletBinding()]
param([Parameter(Mandatory)][string]$JavaHome)
$ErrorActionPreference = 'Stop'
$testOutput = Join-Path ([IO.Path]::GetTempPath()) ('atmungsraum-helper-policy-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testOutput | Out-Null
$source = Join-Path $PSScriptRoot 'src/org/viscereality/atmungsraum/boot/BootPolicy.java'
$test = Join-Path $PSScriptRoot '../../tests/BootHelperPolicyTest.java'
& (Join-Path $JavaHome 'bin/javac.exe') -encoding UTF-8 -d $testOutput $source $test
if ($LASTEXITCODE -ne 0) { throw 'Policy compilation failed.' }
& (Join-Path $JavaHome 'bin/java.exe') -cp $testOutput org.viscereality.atmungsraum.boot.BootHelperPolicyTest
if ($LASTEXITCODE -ne 0) { throw 'Policy test failed.' }
