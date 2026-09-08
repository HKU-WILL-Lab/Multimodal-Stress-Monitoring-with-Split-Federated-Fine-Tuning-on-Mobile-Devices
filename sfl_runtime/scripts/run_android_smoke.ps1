param(
    [string]$Config = 'configs\smoke_single_phone.json',
    [string]$DeviceRoot = '/data/local/tmp/sfl-clean',
    [string]$ClientId = 'phone-0',
    [int]$ClientIndex = 0,
    [int]$ClientCount = 1,
    [int]$DeviceSuffixPort = 51051,
    [int]$DeviceCoordinatorPort = 50052,
    [string]$Serial = ''
)

$ErrorActionPreference = 'Stop'
$RepositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Config = if ([System.IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $RepositoryRoot $Config }
$Settings = Get-Content -Raw -LiteralPath $Config | ConvertFrom-Json
if ($ClientId -notmatch '^[!-~]+$' -or $ClientId -match '\s') { throw 'ClientId must be printable ASCII without whitespace' }
if ($DeviceRoot -notmatch '^/data/local/tmp/[A-Za-z0-9._/-]+$') { throw 'Invalid DeviceRoot' }

function Get-Port([string]$Bind) {
    $PortText = $Bind.Substring($Bind.LastIndexOf(':') + 1)
    $Port = 0
    if (-not [int]::TryParse($PortText, [ref]$Port) -or $Port -le 0 -or $Port -gt 65535) {
        throw "Invalid bind address: $Bind"
    }
    return $Port
}
$SuffixPort = Get-Port $Settings.suffix_rpc.bind
$CoordinatorPort = Get-Port $Settings.coordinator_rpc.bind
foreach ($Port in @($DeviceSuffixPort, $DeviceCoordinatorPort)) {
    if ($Port -le 0 -or $Port -gt 65535) { throw "Invalid device port: $Port" }
}
$AdbPrefix = @()
if ($Serial) { $AdbPrefix += @('-s', $Serial) }
& adb @AdbPrefix reverse "tcp:$DeviceSuffixPort" "tcp:$SuffixPort"
if ($LASTEXITCODE -ne 0) { throw 'Could not reverse suffix port' }
& adb @AdbPrefix reverse "tcp:$DeviceCoordinatorPort" "tcp:$CoordinatorPort"
if ($LASTEXITCODE -ne 0) { throw 'Could not reverse coordinator port' }

$DeviceConfigName = [System.IO.Path]::GetFileName($Config)
$Command = "cd '$DeviceRoot' && ./sfl_android_client" +
    " --config '$DeviceRoot/$DeviceConfigName'" +
    " --client-id '$ClientId'" +
    " --model-dir '$DeviceRoot/model'" +
    " --encoder-pte '$DeviceRoot/encoder/time-series-encoder.pte'" +
    " --dataset '$DeviceRoot/data/training.sflsensor'" +
    " --suffix '127.0.0.1:$DeviceSuffixPort'" +
    " --coordinator '127.0.0.1:$DeviceCoordinatorPort'" +
    " --client-index '$ClientIndex' --client-count '$ClientCount'"
& adb @AdbPrefix shell $Command
if ($LASTEXITCODE -ne 0) { throw 'Android smoke client failed' }
