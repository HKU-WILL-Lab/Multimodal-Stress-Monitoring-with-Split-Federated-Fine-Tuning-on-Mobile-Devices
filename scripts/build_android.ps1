param(
    [string]$GrpcSource = 'C:\Users\10850\grpc-src',
    [string]$NdkRoot = 'D:\functional_exe\android-ndk-r29-windows\android-ndk-r29',
    [string]$MobileFineTunerSource = 'C:\Users\10850\AppData\Local\Temp\mft-cleanroom-reference',
    [string]$BuildRoot = 'build\android-toolchain',
    [int]$ApiLevel = 28,
    [ValidateRange(1, 64)][int]$BuildJobs = 8,
    [switch]$VcEnvironmentReady
)

$ErrorActionPreference = 'Stop'
if ($VcEnvironmentReady) {
    $GrpcSource = $env:SFL_BUILD_GRPC_SOURCE
    $NdkRoot = $env:SFL_BUILD_NDK_ROOT
    $MobileFineTunerSource = $env:SFL_BUILD_MFT_SOURCE
    $BuildRoot = $env:SFL_BUILD_ROOT
    $ApiLevel = [int]$env:SFL_BUILD_API_LEVEL
    $BuildJobs = [int]$env:SFL_BUILD_JOBS
}
$RepositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$BuildRoot = if ([System.IO.Path]::IsPathRooted($BuildRoot)) {
    [System.IO.Path]::GetFullPath($BuildRoot)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $BuildRoot))
}
$GrpcSource = [System.IO.Path]::GetFullPath($GrpcSource)
$NdkRoot = [System.IO.Path]::GetFullPath($NdkRoot)
$MobileFineTunerSource = [System.IO.Path]::GetFullPath($MobileFineTunerSource)
$Toolchain = Join-Path $NdkRoot 'build/cmake/android.toolchain.cmake'
$NdkMake = Join-Path $NdkRoot 'prebuilt/windows-x86_64/bin/make.exe'

foreach ($Required in @($GrpcSource, $NdkRoot, $MobileFineTunerSource, $Toolchain, $NdkMake)) {
    if (-not (Test-Path -LiteralPath $Required)) { throw "Missing prerequisite: $Required" }
}
if ((& git -C $GrpcSource describe --tags --exact-match).Trim() -ne 'v1.83.0') {
    throw 'This reproducible build expects the public gRPC v1.83.0 tag'
}

$VsWhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $VsWhere)) { throw 'Visual Studio Build Tools were not found' }
$VisualStudio = (& $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
if (-not $VisualStudio) { throw 'Visual Studio C++ Build Tools were not found' }
$VcVars = Join-Path $VisualStudio 'VC/Auxiliary/Build/vcvars64.bat'
if (-not $VcEnvironmentReady) {
    $env:SFL_BUILD_GRPC_SOURCE = $GrpcSource
    $env:SFL_BUILD_NDK_ROOT = $NdkRoot
    $env:SFL_BUILD_MFT_SOURCE = $MobileFineTunerSource
    $env:SFL_BUILD_ROOT = $BuildRoot
    $env:SFL_BUILD_API_LEVEL = $ApiLevel.ToString()
    $env:SFL_BUILD_JOBS = $BuildJobs.ToString()
    $Self = $MyInvocation.MyCommand.Path
    $VcCommand = '"' + $VcVars + '" >nul && powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' +
        $Self + '" -VcEnvironmentReady'
    & cmd.exe /d /s /c $VcCommand
    if ($LASTEXITCODE -ne 0) { throw "Android build child process failed with exit code $LASTEXITCODE" }
    return
}
$NMakeCommand = Get-Command nmake.exe -ErrorAction SilentlyContinue
if (-not $NMakeCommand) {
    throw 'vcvars64 did not expose nmake.exe'
}
$NMake = $NMakeCommand.Source

if (-not (Test-Path -LiteralPath $BuildRoot)) {
    New-Item -ItemType Directory -Path $BuildRoot -Force | Out-Null
}
$HostBuild = Join-Path $BuildRoot 'grpc-host'
$AndroidGrpcBuild = Join-Path $BuildRoot 'grpc-arm64-make-build'
$AndroidGrpcInstall = Join-Path $BuildRoot 'grpc-arm64-make-install'
$PatchedMft = Join-Path $BuildRoot 'mft-patched'
$ClientBuild = Join-Path $BuildRoot 'client-arm64'
$ProtoDir = Join-Path $RepositoryRoot 'proto'

& cmake -S $GrpcSource -B $HostBuild -G 'NMake Makefiles' `
    "-DCMAKE_MAKE_PROGRAM=$NMake" `
    -DCMAKE_BUILD_TYPE=Release `
    -DOPENSSL_NO_ASM=ON `
    -Dprotobuf_INSTALL=OFF `
    -Dutf8_range_ENABLE_INSTALL=OFF `
    -DgRPC_BUILD_TESTS=OFF `
    -DgRPC_INSTALL=OFF
if ($LASTEXITCODE -ne 0) { throw 'Host gRPC configure failed' }
& cmake --build $HostBuild --target protoc grpc_cpp_plugin
if ($LASTEXITCODE -ne 0) { throw 'Host protobuf/gRPC codegen build failed' }
$Protoc = Get-ChildItem -LiteralPath $HostBuild -Recurse -File -Filter protoc.exe | Select-Object -First 1 -ExpandProperty FullName
$GrpcPlugin = Get-ChildItem -LiteralPath $HostBuild -Recurse -File -Filter grpc_cpp_plugin.exe | Select-Object -First 1 -ExpandProperty FullName
if (-not $Protoc -or -not $GrpcPlugin) { throw 'Host codegen executables were not found after build' }

& cmake -S $GrpcSource -B $AndroidGrpcBuild -G 'MinGW Makefiles' `
    "-DCMAKE_MAKE_PROGRAM=$NdkMake" `
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain" `
    -DANDROID_ABI=arm64-v8a `
    "-DANDROID_PLATFORM=android-$ApiLevel" `
    -DANDROID_STL=c++_static `
    -DCMAKE_BUILD_TYPE=Release `
    -DOPENSSL_NO_ASM=ON `
    -DABSL_ENABLE_INSTALL=ON `
    -DBUILD_TESTING=OFF `
    -DRE2_BUILD_TESTING=OFF `
    -DCARES_BUILD_TOOLS=OFF `
    -Dprotobuf_BUILD_TESTS=OFF `
    "-DCMAKE_INSTALL_PREFIX=$AndroidGrpcInstall" `
    -DgRPC_BUILD_TESTS=OFF `
    -DgRPC_BUILD_CODEGEN=OFF `
    -DgRPC_BUILD_GRPC_CPP_PLUGIN=OFF `
    -Dprotobuf_BUILD_PROTOC_BINARIES=OFF `
    -DgRPC_INSTALL=ON
if ($LASTEXITCODE -ne 0) { throw 'Android gRPC configure failed' }
& cmake --build $AndroidGrpcBuild --parallel $BuildJobs --target install
if ($LASTEXITCODE -ne 0) { throw 'Android gRPC install failed' }

if (Test-Path -LiteralPath $PatchedMft) {
    $ResolvedBuild = [System.IO.Path]::GetFullPath($BuildRoot)
    $ResolvedPatch = [System.IO.Path]::GetFullPath($PatchedMft)
    $BuildPrefix = $ResolvedBuild.TrimEnd([char[]]@('\', '/')) +
        [System.IO.Path]::DirectorySeparatorChar
    if (-not $ResolvedPatch.StartsWith($BuildPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'Refusing to remove a patch directory outside BuildRoot'
    }
    Remove-Item -LiteralPath $ResolvedPatch -Recurse -Force
}
& (Join-Path $RepositoryRoot 'mft_patch/apply_and_verify.ps1') `
    -SourceRoot $MobileFineTunerSource -Destination $PatchedMft

& cmake -S (Join-Path $RepositoryRoot 'android') -B $ClientBuild -G 'MinGW Makefiles' `
    "-DCMAKE_MAKE_PROGRAM=$NdkMake" `
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain" `
    -DANDROID_ABI=arm64-v8a `
    "-DANDROID_PLATFORM=android-$ApiLevel" `
    -DANDROID_STL=c++_static `
    -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_PREFIX_PATH=$AndroidGrpcInstall" `
    "-DProtobuf_DIR=$(Join-Path $AndroidGrpcInstall 'lib/cmake/protobuf')" `
    "-DgRPC_DIR=$(Join-Path $AndroidGrpcInstall 'lib/cmake/grpc')" `
    "-Dabsl_DIR=$(Join-Path $AndroidGrpcInstall 'lib/cmake/absl')" `
    "-Dutf8_range_DIR=$(Join-Path $AndroidGrpcInstall 'lib/cmake/utf8_range')" `
    "-DMFT_SOURCE_DIR=$PatchedMft" `
    "-DSFL_PROTO_DIR=$ProtoDir" `
    "-DSFL_HOST_PROTOC=$Protoc" `
    "-DSFL_HOST_GRPC_CPP_PLUGIN=$GrpcPlugin"
if ($LASTEXITCODE -ne 0) { throw 'Android client configure failed' }
& cmake --build $ClientBuild --parallel $BuildJobs --target sfl_android_client
if ($LASTEXITCODE -ne 0) { throw 'Android client build failed' }
Write-Host (Join-Path $ClientBuild 'sfl_android_client')
