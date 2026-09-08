param(
    [Parameter(Mandatory = $true)][string]$GrpcSource,
    [Parameter(Mandatory = $true)][string]$NdkRoot,
    [Parameter(Mandatory = $true)][string]$MobileFineTunerSource,
    [Parameter(Mandatory = $true)][string]$ExecuTorchSource,
    [Parameter(Mandatory = $true)][string]$PythonExecutable,
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
    $ExecuTorchSource = $env:SFL_BUILD_EXECUTORCH_SOURCE
    $PythonExecutable = $env:SFL_BUILD_PYTHON
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
$ExecuTorchSource = [System.IO.Path]::GetFullPath($ExecuTorchSource)
$PythonExecutable = [System.IO.Path]::GetFullPath($PythonExecutable)
$Toolchain = Join-Path $NdkRoot 'build/cmake/android.toolchain.cmake'
$NdkMake = Join-Path $NdkRoot 'prebuilt/windows-x86_64/bin/make.exe'

# Keep Python/CMake temporary files on the short, writable build path. Apart
# from avoiding MAX_PATH failures, this prevents Windows security software
# from intermittently denying ExecuTorch codegen access to the user TEMP tree.
if (-not (Test-Path -LiteralPath $BuildRoot)) {
    New-Item -ItemType Directory -Path $BuildRoot -Force | Out-Null
}
$NativeTemp = Join-Path $BuildRoot 'tmp'
New-Item -ItemType Directory -Path $NativeTemp -Force | Out-Null
$env:TEMP = $NativeTemp
$env:TMP = $NativeTemp

foreach ($Required in @($GrpcSource, $NdkRoot, $MobileFineTunerSource, $ExecuTorchSource, $PythonExecutable, $Toolchain, $NdkMake)) {
    if (-not (Test-Path -LiteralPath $Required)) { throw "Missing prerequisite: $Required" }
}
& $PythonExecutable -c 'import torchgen, yaml'
if ($LASTEXITCODE -ne 0) {
    throw 'ExecuTorch code generation requires a Python environment containing torchgen and PyYAML'
}
# ExecuTorch builds its schema compilers as host-side ExternalProjects using
# the Unix Makefiles generator even during an Android cross-build. Make the
# same GNU make executable visible to those child CMake processes.
$NdkMakeDirectory = Split-Path -Parent $NdkMake
if (($env:PATH -split ';') -notcontains $NdkMakeDirectory) {
    $env:PATH = $NdkMakeDirectory + ';' + $env:PATH
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
    $env:SFL_BUILD_EXECUTORCH_SOURCE = $ExecuTorchSource
    $env:SFL_BUILD_PYTHON = $PythonExecutable
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

$HostBuild = Join-Path $BuildRoot 'grpc-host'
$AndroidGrpcBuild = Join-Path $BuildRoot 'grpc-arm64-make-build'
$AndroidGrpcInstall = Join-Path $BuildRoot 'grpc-arm64-make-install'
$ClientBuild = Join-Path $BuildRoot 'client-arm64'
$InferenceBuild = Join-Path $BuildRoot 'inference-arm64'
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
    "-DMFT_SOURCE_DIR=$MobileFineTunerSource" `
    "-DSFL_PROTO_DIR=$ProtoDir" `
    "-DSFL_HOST_PROTOC=$Protoc" `
    "-DSFL_HOST_GRPC_CPP_PLUGIN=$GrpcPlugin" `
    -DSFL_ENABLE_EXECUTORCH_ENCODER=ON `
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF `
    -DEXECUTORCH_XNNPACK_ENABLE_KLEIDI=OFF `
    -DXNNPACK_ENABLE_KLEIDIAI=OFF `
    "-DPYTHON_EXECUTABLE=$PythonExecutable" `
    "-DPython3_EXECUTABLE=$PythonExecutable" `
    "-DEXECUTORCH_SOURCE_DIR=$ExecuTorchSource"
if ($LASTEXITCODE -ne 0) { throw 'Android client configure failed' }
& cmake --build $ClientBuild --parallel $BuildJobs --target sfl_android_client wellbeing_sfl
if ($LASTEXITCODE -ne 0) { throw 'Android client/JNI build failed' }

# Configure inference separately. ExecuTorch's tokenizer vendors protobuf-lite,
# RE2, and Abseil; isolating it avoids symbol and header collisions with gRPC.
& cmake -S (Join-Path $RepositoryRoot 'android/inference') -B $InferenceBuild -G 'MinGW Makefiles' `
    "-DCMAKE_MAKE_PROGRAM=$NdkMake" `
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain" `
    -DANDROID_ABI=arm64-v8a `
    "-DANDROID_PLATFORM=android-$ApiLevel" `
    -DANDROID_STL=c++_static `
    -DCMAKE_BUILD_TYPE=Release `
    "-DMFT_SOURCE_DIR=$MobileFineTunerSource" `
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF `
    -DEXECUTORCH_XNNPACK_ENABLE_KLEIDI=OFF `
    -DXNNPACK_ENABLE_KLEIDIAI=OFF `
    "-DPYTHON_EXECUTABLE=$PythonExecutable" `
    "-DPython3_EXECUTABLE=$PythonExecutable" `
    "-DEXECUTORCH_SOURCE_DIR=$ExecuTorchSource"
if ($LASTEXITCODE -ne 0) { throw 'Android local inference configure failed' }
& cmake --build $InferenceBuild --parallel $BuildJobs --target wellbeing_inference
if ($LASTEXITCODE -ne 0) { throw 'Android local inference build failed' }
$AppJniDirectory = Join-Path $BuildRoot 'app-jni/arm64-v8a'
New-Item -ItemType Directory -Path $AppJniDirectory -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $ClientBuild 'libwellbeing_sfl.so') `
    -Destination (Join-Path $AppJniDirectory 'libwellbeing_sfl.so') -Force
Copy-Item -LiteralPath (Join-Path $InferenceBuild 'libwellbeing_inference.so') `
    -Destination (Join-Path $AppJniDirectory 'libwellbeing_inference.so') -Force
Write-Host (Join-Path $ClientBuild 'sfl_android_client')
Write-Host (Join-Path $AppJniDirectory 'libwellbeing_sfl.so')
Write-Host (Join-Path $AppJniDirectory 'libwellbeing_inference.so')
