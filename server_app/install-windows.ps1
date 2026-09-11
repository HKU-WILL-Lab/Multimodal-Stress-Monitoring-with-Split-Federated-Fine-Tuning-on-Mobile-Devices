param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$Config
)
$ErrorActionPreference = 'Stop'
$pythonPath = (Resolve-Path -LiteralPath $Python).Path
$configPath = (Resolve-Path -LiteralPath $Config).Path
$pythonWindow = Join-Path (Split-Path $pythonPath) 'pythonw.exe'
if (!(Test-Path -LiteralPath $pythonWindow)) { throw 'pythonw.exe is required beside python.exe' }
$installRoot = Join-Path $env:LOCALAPPDATA 'MobiWellbeingServer'
$appRoot = Join-Path $installRoot 'server_app'
New-Item -ItemType Directory -Force -Path $appRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'app.py') -Destination $appRoot -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'deployment.py') -Destination $appRoot -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'phone_control.cjs') -Destination $appRoot -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'static') -Destination $appRoot -Recurse -Force
$runtimeSource = Join-Path (Split-Path $PSScriptRoot) 'sfl_runtime'
$runtimeTarget = Join-Path $installRoot 'sfl_runtime'
New-Item -ItemType Directory -Force -Path $runtimeTarget | Out-Null
Copy-Item -LiteralPath (Join-Path $runtimeSource 'python') -Destination $runtimeTarget -Recurse -Force
Copy-Item -LiteralPath (Join-Path $runtimeSource 'proto') -Destination $runtimeTarget -Recurse -Force
$installedConfig = Join-Path $installRoot 'server.json'
Copy-Item -LiteralPath $configPath -Destination $installedConfig -Force
$shell = New-Object -ComObject WScript.Shell
foreach ($folder in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {
    $shortcut = $shell.CreateShortcut((Join-Path $folder 'MobiWellbeing Server.lnk'))
    $shortcut.TargetPath = $pythonWindow
    $shortcut.Arguments = '"' + (Join-Path $appRoot 'app.py') + '" --config "' + $installedConfig + '"'
    $shortcut.WorkingDirectory = $installRoot
    $shortcut.Description = 'MobiWellbeing Main Server and Federated Server'
    $shortcut.Save()
}
Write-Output "Installed in $installRoot. Open MobiWellbeing Server from the desktop or Start menu."
