param(
    [Parameter(Mandatory = $true)][string]$SourceRoot,
    [Parameter(Mandatory = $true)][string]$Destination
)

$ErrorActionPreference = 'Stop'
$PinnedCommit = 'b62d3b12a597e05489e6e8ef025527c613c94837'
$PatchPath = Join-Path $PSScriptRoot 'mobilefinetuner-hidden-span.patch'
$SourceRoot = [System.IO.Path]::GetFullPath($SourceRoot)
$Destination = [System.IO.Path]::GetFullPath($Destination)

if (-not (Test-Path -LiteralPath (Join-Path $SourceRoot '.git'))) {
    throw "SourceRoot is not a MobileFineTuner Git checkout: $SourceRoot"
}
$ActualCommit = (& git -C $SourceRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $ActualCommit -ne $PinnedCommit) {
    throw "Expected MobileFineTuner $PinnedCommit, found $ActualCommit"
}
if (Test-Path -LiteralPath $Destination) {
    throw "Destination already exists; refusing to overwrite: $Destination"
}

New-Item -ItemType Directory -Path $Destination | Out-Null
Copy-Item -LiteralPath (Join-Path $SourceRoot 'operator') -Destination $Destination -Recurse
Copy-Item -LiteralPath (Join-Path $SourceRoot 'LICENSE') -Destination $Destination

& git -C $Destination init --quiet
if ($LASTEXITCODE -ne 0) { throw 'Could not initialize generated patch workspace' }
& git -C $Destination apply --check -p2 $PatchPath
if ($LASTEXITCODE -ne 0) { throw 'MobileFineTuner patch preflight failed' }
& git -C $Destination apply -p2 $PatchPath
if ($LASTEXITCODE -ne 0) { throw 'MobileFineTuner patch application failed' }

$RequiredNotice = 'MODIFIED by the SFL Clean project, 2026.'
$ModifiedFiles = @(
    'operator/finetune_ops/graph/gemma_model.h',
    'operator/finetune_ops/graph/gemma_model.cpp',
    'operator/finetune_ops/graph/gemma_lora_injector.cpp'
)
foreach ($RelativePath in $ModifiedFiles) {
    $Content = Get-Content -Raw -LiteralPath (Join-Path $Destination $RelativePath)
    if (-not $Content.Contains($RequiredNotice)) {
        throw "Missing Apache modified-file notice in $RelativePath"
    }
}
Write-Host "Patched pristine MobileFineTuner $PinnedCommit into $Destination"
