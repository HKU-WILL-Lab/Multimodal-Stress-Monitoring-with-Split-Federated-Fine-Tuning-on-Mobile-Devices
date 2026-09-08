param(
    [Parameter(Mandatory)][string]$StagedDirectory,
    [string]$Adb = 'adb',
    [string]$Serial = ''
)

$ErrorActionPreference = 'Stop'
$source = [System.IO.Path]::GetFullPath($StagedDirectory)
if (-not (Test-Path -LiteralPath (Join-Path $source 'deployment.json'))) {
    throw 'StagedDirectory must contain deployment.json'
}
$adbCommand = Get-Command $Adb -ErrorAction Stop
$arguments = @()
if ($Serial) { $arguments += @('-s', $Serial) }
$state = & $adbCommand @arguments get-state
if ($LASTEXITCODE -ne 0 -or $state.Trim() -ne 'device') {
    throw 'No authorized Android device is available'
}
$destination = '/sdcard/Android/data/org.mobihoc.wellbeing/files/inference'
& $adbCommand @arguments shell mkdir -p $destination
if ($LASTEXITCODE -ne 0) { throw 'Could not create the app inference directory' }
& $adbCommand @arguments push (Join-Path $source '.') $destination
if ($LASTEXITCODE -ne 0) { throw 'Could not copy inference assets to the phone' }
Write-Host "Inference assets deployed to $destination/deployment.json"
