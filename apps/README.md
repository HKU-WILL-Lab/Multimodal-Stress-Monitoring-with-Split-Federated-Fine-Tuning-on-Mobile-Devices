# MobiWellbeing Demo Apps

This directory contains the non-UI runtime source for a vendor-neutral Android
phone companion and a Galaxy Watch8 Wear OS component. The visual interface is
intentionally withheld until its design and behavior are stable. Project-wide
licensing and provenance are documented at the repository root.

## Modules

- `shared`: sensor-window binary protocol, affect mapping, trend text, tests.
- `phone`: receives watch windows, invokes the training and inference bridges,
  maintains scores/state/trend, and sends an alert to the watch.
- `wear`: captures a 30-second window and transfers it to the paired phone.
  The `demo` flavor uses deterministic synthetic signals. The `samsung`
  flavor uses Samsung Health Sensor SDK on Galaxy Watch8.

The modules use the same application ID and must be signed with the same key
for the Wear OS Data Layer. They intentionally declare no launcher activity;
an application integrating this release must supply its own UI or service
control entry point. The phone's `MobiWellbeingApplication` owns the headless
`WellbeingController`, so incoming watch windows still reach the inference
pipeline without a launcher activity. The future UI can obtain that controller
from the application instance.

## Samsung SDK (not redistributed)

Download Samsung Health Sensor SDK 1.4.1 from Samsung Developer and copy:

```text
samsung-health-sensor-api.aar
```

to `wear/libs/`. Enable Health Sensor Service developer mode on the Watch8
for local demo builds. Public distribution requires Samsung partner approval
and registration of the package name and signing-certificate SHA-256.

## Build variants

```powershell
# Logic tests
.\gradlew.bat :shared:test

# Phone plus synthetic watch demo
.\gradlew.bat :phone:assembleDebug :wear:assembleDemoDebug

# Physical Watch8 sensor build (requires the Samsung AAR)
.\gradlew.bat :wear:assembleSamsungDebug
```

## Native SFL training build

Build the clean-room Android runtimes first. The script produces the ADB
training client plus two app-loadable libraries:
`libwellbeing_sfl.so` for training and `libwellbeing_inference.so` for local
inference.

```powershell
..\sfl_runtime\scripts\build_android.ps1 `
  -GrpcSource <grpc-source> `
  -NdkRoot <android-ndk> `
  -MobileFineTunerSource <mobilefinetuner-source> `
  -ExecuTorchSource <executorch-source> `
  -PythonExecutable <python-executable>
.\gradlew.bat :phone:assembleDebug -PenableNativeSfl=true
```

For a nondefault native build directory, pass its `app-jni` directory with
`-PnativeSflLibDir=<absolute-path>`. Without `enableNativeSfl`, the modules can
still be built for shared-logic and watch-transport integration tests.

The project uses AGP 9.2.0, Gradle 9.4.1, Kotlin 2.3.21, Play services
Wearable 20.0.1, and Android compile/target SDK 37. A JDK compatible with this
AGP version is required.

## Training and inference integration

`NativeSflTrainingEngine` starts, monitors, and cancels the native client
through JNI. It requires the training configuration, exported encoder `.pte`, Llama token
embedding asset, prepared `.sflsensor` dataset, and reachable main/federated
servers. These are runtime assets and are not packaged in the source tree.
The two server address fields accept separate `host:port` values, which is
required when the main and federated services run on different laptops.

`phone` uses `NativeSflInferenceEngine` for watch data. Its JNI boundary accepts
the compact 30-second sensor-window payload and returns:

```json
{"valence":4,"arousal":2,"assessment":"Valence remained positive while arousal decreased."}
```

The native inference path resamples and normalizes the configured sensor
channels, runs the frozen encoder, applies the trained alignment projector,
prepends the resulting sensor tokens to the Llama prompt embeddings, and runs
the local Llama decoder autoregressively. Configure
`NativeSflInferenceEngine` with a deployment JSON before processing watch
data. Its asset paths are resolved relative to the JSON file; see
`sfl_runtime/configs/inference_deployment.example.json`.
For debug deployment, copy the complete inference directory to the app's
external files directory (normally
`/sdcard/Android/data/org.mobihoc.wellbeing/files/inference`). No broad storage
permission is required because this directory belongs to the app.

The runtime deliberately does not fall back to synthetic predictions for a
real watch window. `DemoInferenceEngine` is isolated for non-model integration
tests and does not claim to execute a trained model.

## UI boundary

No Compose screen, launcher activity, visual asset, or Stitch export is part
of this source release. A later UI can consume `PhoneRuntimeState`, call
`WellbeingController`, and observe `WearRuntimeStore` without changing the
sensor, training, inference, or transport implementations.
