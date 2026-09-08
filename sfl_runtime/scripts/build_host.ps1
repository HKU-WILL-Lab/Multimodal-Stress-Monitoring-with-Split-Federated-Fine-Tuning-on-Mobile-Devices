param(
    [string]$Python = 'python',
    [string]$Environment = '.venv'
)

$ErrorActionPreference = 'Stop'
$RepositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Environment = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $Environment))

& $Python -m venv $Environment
if ($LASTEXITCODE -ne 0) { throw 'Could not create Python environment' }
$EnvironmentPython = Join-Path $Environment 'Scripts/python.exe'
& $EnvironmentPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not upgrade pip' }
& $EnvironmentPython -m pip install --editable "${RepositoryRoot}[model,dev]"
if ($LASTEXITCODE -ne 0) { throw 'Could not install pinned host dependencies' }
& $EnvironmentPython -m sfl_clean.proto_runtime
if ($LASTEXITCODE -ne 0) { throw 'Could not generate Python protobuf bindings' }
& $EnvironmentPython -m compileall -q (Join-Path $RepositoryRoot 'python')
if ($LASTEXITCODE -ne 0) { throw 'Python byte compilation failed' }
& $EnvironmentPython -m pytest
if ($LASTEXITCODE -ne 0) { throw 'Host test suite failed' }
