param(
    [Parameter(Mandatory = $true)][string]$ClientBinary,
    [Parameter(Mandatory = $true)][string]$ModelDir,
    [Parameter(Mandatory = $true)][string]$Dataset,
    [string]$Config = 'configs\smoke_single_phone.json',
    [string]$DeviceRoot = '/data/local/tmp/sfl-clean',
    [string]$Serial = ''
)

$ErrorActionPreference = 'Stop'
$RepositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Config = if ([System.IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $RepositoryRoot $Config }
foreach ($Required in @($ClientBinary, $ModelDir, $Dataset, $Config)) {
    if (-not (Test-Path -LiteralPath $Required)) { throw "Missing staging input: $Required" }
}
if ($DeviceRoot -notmatch '^/data/local/tmp/[A-Za-z0-9._/-]+$') {
    throw 'DeviceRoot must be a simple path below /data/local/tmp'
}
$AdbPrefix = @()
if ($Serial) { $AdbPrefix += @('-s', $Serial) }
& adb @AdbPrefix shell "mkdir -p '$DeviceRoot/model' '$DeviceRoot/data' '$DeviceRoot/runtime/metrics'"
if ($LASTEXITCODE -ne 0) { throw 'Could not create device staging directories' }
& adb @AdbPrefix push $ClientBinary "$DeviceRoot/sfl_android_client"
if ($LASTEXITCODE -ne 0) { throw 'Could not stage client executable' }
& adb @AdbPrefix push $Config "$DeviceRoot/smoke_single_phone.json"
if ($LASTEXITCODE -ne 0) { throw 'Could not stage shared configuration' }
& adb @AdbPrefix push $Dataset "$DeviceRoot/data/wikitext.txt"
if ($LASTEXITCODE -ne 0) { throw 'Could not stage dataset' }
& adb @AdbPrefix push (Join-Path $ModelDir '.') "$DeviceRoot/model/"
if ($LASTEXITCODE -ne 0) { throw 'Could not stage external model assets' }
& adb @AdbPrefix shell "chmod 700 '$DeviceRoot/sfl_android_client'"
if ($LASTEXITCODE -ne 0) { throw 'Could not mark client executable' }
