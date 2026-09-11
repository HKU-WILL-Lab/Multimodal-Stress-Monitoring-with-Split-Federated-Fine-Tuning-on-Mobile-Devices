# MobiWellbeing Server

Use the desktop application to prepare the phone and start split federated
training. **Main Server** performs suffix training; **Federated Server** aggregates
phone-side updates. The main walkthrough runs both services on one Windows
computer and uses USB or wireless ADB to communicate with one phone.

After the first-time setup, the normal sequence is:

**Select phone → choose Cutting Layer and Steps → Build & Deploy → Start Training**

## First-time setup

Run the commands below from the repository root in PowerShell. Replace the
example paths with your local paths. Keep model and output paths absolute.

### 1. Prepare the Python environment and Android toolchain

Install the prerequisites listed in the [main README](../README.md#requirements),
including Node.js 22+ and Android Platform Tools. Prepare the Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e "./sfl_runtime[model,dev]" PyYAML
```

Set `JAVA_HOME` to your Android build JDK and `ANDROID_HOME` to the Android SDK.
Build the native runtimes using the external source checkouts described in the
[native build guide](../sfl_runtime/android/README.md):

```powershell
$Repo = (Get-Location).Path
$Python = Join-Path $Repo '.venv/Scripts/python.exe'
$NativeBuild = 'C:/sfl-build'

& ./sfl_runtime/scripts/build_android.ps1 `
  -GrpcSource 'C:/src/grpc' `
  -NdkRoot 'C:/Android/android-ndk-r29' `
  -MobileFineTunerSource 'C:/src/MobileFineTuner' `
  -ExecuTorchSource 'C:/src/executorch' `
  -PythonExecutable $Python `
  -BuildRoot $NativeBuild
```

Choose a short, writable native build directory. The generated
`client-arm64` cache must use this repository's updated runtime sources. Keep
that build directory and the source checkouts available: **Build & Deploy**
reuses them. The `app-jni/arm64-v8a` output contains the training and inference
libraries that are included in the phone APK.

### 2. Prepare the model, encoder and training data

Follow [runtime asset preparation](../sfl_runtime/README.md#external-prerequisites)
to obtain the Llama 3.2 1B checkpoint/tokenizer, export the frozen 30-second
NormWear encoder, and prepare labeled `.sflsensor` records with the matching
Llama tokenizer. The desktop deployment uses the encoder with input
`[1, 6, 240]` and output `[1, 162, 768]`; prepare records with text sequence
length 128 for this walkthrough.

Set paths to those prepared files:

```powershell
$Model = 'C:/models/llama-3.2-1b-instruct'
$Encoder = 'C:/assets/normwear-30s-tokens.pte'
$Dataset = 'C:/assets/kemocon-30s-train.sflsensor'
$RunRoot = 'C:/sfl-runs'
$Adb = Join-Path $env:ANDROID_HOME 'platform-tools/adb.exe'
```

Keep the original Llama checkpoint on the computer. **Build & Deploy** exports
and transfers only the embedding and selected decoder prefix needed by the
phone. The encoder and dataset are copied once in the next step.

### 3. Install the phone application and its sensor assets

[Connect and authorize the phone](#connect-the-phone), then use its ID from
`adb devices`:

```powershell
$Device = '<device ID shown by adb devices>'
& ./apps/gradlew.bat -p apps :phone:assembleDebug `
  -PenableNativeSfl=true "-PnativeSflLibDir=$NativeBuild/app-jni"
& $Adb -s $Device install -r ./apps/phone/build/outputs/apk/debug/phone-debug.apk
& $Adb -s $Device shell am start -n org.mobihoc.wellbeing/.phone.MainActivity
& $Adb -s $Device shell mkdir -p /sdcard/Android/data/org.mobihoc.wellbeing/files/sfl
& $Adb -s $Device push $Encoder /sdcard/Android/data/org.mobihoc.wellbeing/files/sfl/normwear-30s-tokens.pte
& $Adb -s $Device push $Dataset /sdcard/Android/data/org.mobihoc.wellbeing/files/sfl/training.sflsensor
```

The desktop uses the debug APK's WebView debugging interface to apply settings
and start training. Keep the phone unlocked and its app accessible during deployment.

### 4. Install the desktop application

Create a base server configuration from the provided example. Set one round and
one participating phone for the initial five-step run:

```powershell
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$ConfigPath = Join-Path $RunRoot 'server.json'
$config = Get-Content ./sfl_runtime/configs/training.example.json -Raw | ConvertFrom-Json
$config.run_id = 'desktop-initial'
$config.model.source = $Model
$config.model.device = 'cuda'
$config.suffix_rpc.bind = '127.0.0.1:50051'
$config.coordinator_rpc.bind = '127.0.0.1:50052'
$config.suffix_rpc.deadline_seconds = 600
$config.coordinator_rpc.deadline_seconds = 600
$config.training.total_rounds = 1
$config.training.submission_quorum = 1
$config.training.cut_layer = 2
$config.training.local_steps = 5
$config.training.batch_size = 1
$config.training.sequence_length = 128
$config.metrics_path = Join-Path $RunRoot 'initial-metrics.jsonl'
$config.checkpoint_root = Join-Path $RunRoot 'checkpoints'
$Utf8 = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText($ConfigPath, ($config | ConvertTo-Json -Depth 8), $Utf8)

& ./server_app/install-windows.ps1 -Python $Python -Config $ConfigPath
```

The installer copies the application and Python server code into
`%LOCALAPPDATA%/MobiWellbeingServer` and adds **MobiWellbeing Server** shortcuts
to the desktop and Start menu. It uses your Python/CUDA environment. The offline
HTML/CSS/fonts open in an Edge application window.

### 5. Configure the desktop build and deployment paths

Before opening the desktop application, save its local `deployment.json`:

```powershell
$Deployment = @{
  repository = $Repo
  adb = $Adb
  node = (Get-Command node).Source
  cmake = (Get-Command cmake).Source
  native_build = Join-Path $NativeBuild 'client-arm64'
  jni_dir = Join-Path $NativeBuild 'app-jni'
  encoder = '/sdcard/Android/data/org.mobihoc.wellbeing/files/sfl/normwear-30s-tokens.pte'
  dataset = '/sdcard/Android/data/org.mobihoc.wellbeing/files/sfl/training.sflsensor'
  environment = @{
    JAVA_HOME = $env:JAVA_HOME
    ANDROID_HOME = $env:ANDROID_HOME
  }
}
$DeploymentPath = Join-Path $env:LOCALAPPDATA 'MobiWellbeingServer/deployment.json'
[IO.File]::WriteAllText($DeploymentPath, ($Deployment | ConvertTo-Json -Depth 8), $Utf8)
```

`encoder` and `dataset` are **phone paths**; the other tool/source paths refer to
the computer. Retain the repository, Python environment and native build cache
at these locations. If you change `deployment.json`, exit and reopen the application.

## Connect the phone

For USB, enable Developer options and USB debugging, connect the phone and accept
its authorization prompt. `adb devices` must show the phone as `device`.

For wireless ADB, enable Wireless debugging and use its two separate screens:

```powershell
& $Adb pair <IP:pairing-port>
# Enter the pairing code at the local prompt.
& $Adb connect <IP:connection-port>
& $Adb devices
```

The connection port is normally different from the pairing port. Reconnect if
wireless debugging changes its address or port. ADB authorization is needed for
this build/install/control workflow; the gRPC training protocol itself does not
require ADB pairing.

## Run training

1. Open **MobiWellbeing Server → Main Server (Training)**.
2. Under **New Training**, select **ADB Phone**. Use **Refresh Phones** when the
   list needs updating; it does not stop servers or reset training.
3. Select **Cutting Layer** and enter **Steps**. Start with **2** and **5**.
4. Click **Build & Deploy** and wait for the deployment-ready message. It builds
   the APK, exports/transfers the selected prefix weights, installs the APK,
   and prepares a fresh run with matching phone/server settings and port forwarding.
5. Click **Start Training**. It starts both local services, waits until they are
   ready, and commands the prepared phone to begin training.
6. Watch Main Server steps/PPL/logs, Federated Server uploads/aggregation, and the
   phone's Training page. Wait for training and aggregation to finish.

A cut of N runs the first N decoder blocks on the phone. **Steps** means local
steps per round; the base JSON supplies the number of rounds and upload quorum.
The sequence above uses one round and one phone. Changing input values after
preparation requires another **Build & Deploy** before **Start Training**.

For another session, repeat **Build & Deploy → Start Training**, even when using
the same settings. Previous metrics/checkpoints are retained. You do not need to
manually stop completed-run services first; deployment stops them when the new
configuration is ready. Finish or cancel any active phone training before deploying.

## Service controls and logs

**Start Main Server / Start Federated Server** starts only that page's service.
These controls are for manual operation; the normal **Start Training** flow
already starts both. **Stop Server** stops the current service and interrupts
training that depends on it. **Exit App** stops the local services and exits the
backend. Closing only the window leaves the backend running.

**Auto-Scroll** follows new Main Server log entries. Turning it off lets you read
earlier entries while new logs continue arriving. **Export** saves the displayed
log selection. The Federated Server log pause button pauses its displayed feed.

PPL comes from the actual per-step loss. The complete X axis starts at step 1;
new steps add hollow points with a translucent purple fill. The Y axis follows
the largest observed PPL. Client counts describe clients seen during the session.

The local desktop endpoint uses port 8765. Default server ports are 50051 (main),
50052 (federated), and 50053 (aggregation status). **Build & Deploy** configures
ADB reverse forwarding, including phone port 50151 to the main server port.

## Manual server operation

For separate server computers, start only the appropriate role on each and use
reachable server addresses in the phone settings. Follow the
[runtime instructions](../sfl_runtime/README.md) for matching run configurations
and manual gRPC service startup. The desktop **Build & Deploy → Start Training**
walkthrough above manages both local roles on one Windows computer.

The controller can also be launched from the source checkout:

```powershell
& $Python ./server_app/app.py --config $ConfigPath
```

For source launch, `deployment.json` belongs at the repository root beside
`server_app/`. The same Python controller can use a default browser on Linux;
this documented Android build/deployment sequence is the Windows workflow.

Continue with [inference in the same phone app](../README.md#7-configure-inference-in-the-same-phone-application)
after preparing the inference model assets.
