# Phone UI and training setup

## Included interface

- **Vitals:** actual watch windows and inference output; empty sensors show no fabricated waveform. The explicitly selected preview mode is labeled Demo.
- **Training:** native start/progress/completion, Current PPL, one point per recorded step, and the current Cutting Layer read from the training configuration.
- **Settings:** training/inference asset paths, server addresses, encoder input length, start/cancel controls and preview actions.

The original Stitch phone presentation is retained as editable HTML/CSS under
`apps/phone/src/main/assets/stitch/`, with `bridge.js` binding native state.
The committed CSS is ready to ship: rebuilding it from an external Stitch export
or running a Tailwind compiler is not required. Fonts are bundled with licenses.
The WebView serves only local app assets, blocks external page loads, disables
file/content access and exposes debugging only in debug builds.

The application owns one controller shared with the launcher activity. Rotation
or reopening the activity does not create a second native session. Runtime state
is in memory; process termination does not automatically resume a training job.

## Build and install

From `apps/`, use a JDK supported by AGP and set `ANDROID_HOME` to your Android SDK:

```powershell
.\gradlew.bat :phone:assembleDebug :phone:lintDebug :shared:test
adb -s <device> install -r phone/build/outputs/apk/debug/phone-debug.apk
adb -s <device> shell am start -n org.mobihoc.wellbeing/org.mobihoc.wellbeing.phone.MainActivity
```

This builds the UI without native model runtimes. For actual training/inference,
first follow the native build instructions in `apps/README.md`, then build with
`-PenableNativeSfl=true`. Use `-PnativeSflLibDir=<absolute-app-jni-directory>` if
the libraries are outside the default `sfl_runtime/build/android-toolchain/app-jni`.

## Training assets and configuration

Supply model assets and labeled records separately. Settings must point to:

- a client training JSON based on `sfl_runtime/configs/training-client.example.json`;
- the mobile model directory exported by `sfl_clean.export_llama_embedding`;
- the frozen NormWear `.pte` encoder;
- the matching `.sflsensor` training split.

The current native runtime requires **cut_layer=1**: the phone executes the
embedding, alignment projector and decoder block 0; the GPU main server executes
blocks 1–15 and the language-model head. The mobile export includes block-0 base
weights. An embedding-only asset from the older cut=0 runtime is insufficient.

The validated 30-second encoder uses input `[1, 6, 240]` and emits `[1, 162, 768]`.
Set Encoder Input Length to 240 only when using that matching export and dataset.
Model paths and dataset windows must match the deployed artifacts.

For a five-step smoke run, copy the server/client example configurations to
ignored runtime directories, choose one fresh matching `run_id`, and set both
training sections to `total_rounds=1`, `local_steps=5`, `submission_quorum=1`,
`cut_layer=1`, `batch_size=1`, `sequence_length=128`. Use separate metrics and
checkpoint directories for every run. Set server `model.source` to your local
model checkpoint; choose suitable RPC deadlines for model loading on your phone.

From `sfl_runtime/`, run the two services in separate terminals:

```powershell
python -m sfl_clean.cli_coordinator --config runtime/server.json --status-bind 127.0.0.1:50053
python -m sfl_clean.cli_suffix --config runtime/server.json
```

Copy the client JSON and assets to the app's external-files directory and select
their paths in Settings. No model, dataset, checkpoint, APK or private device
configuration is part of this source release.

## USB and wireless debugging

For development with both services on one computer:

```text
adb -s <device> reverse tcp:50151 tcp:50051
adb -s <device> reverse tcp:50052 tcp:50052
adb -s <device> reverse tcp:50053 tcp:50053
```

In phone Settings, use Main Server `127.0.0.1:50151`, Federated Server
`127.0.0.1:50052`, and status endpoint `http://127.0.0.1:50053/status`.
Then tap Start Training. Port 50151 on the phone forwards to server port 50051.

For wireless ADB, enable Wireless debugging on the phone, pair using the pairing
screen's address/code, then connect using the main debugging screen's connection
port (normally different from the pairing port). Use that wireless device ID in
the same three reverse commands. The computer and phone must be mutually
reachable over the network; reconnect and restore forwarding after ADB/network
changes. No pairing codes, local IP addresses or device serials are committed.

ADB is a development transport, not a training protocol requirement. Direct
network training can use reachable server `host:port` addresses without ADB,
with services bound appropriately. The bundled HTTP status policy allows only
loopback cleartext; use HTTPS for a remote status endpoint. The local WebView
itself stays offline while the native runtime exchanges training data.

## Chart and status behavior

An unknown total shows an unnumbered baseline. Once known, the complete session
axis spans step **1** through the final step, with up to five intervals and
integer-rounded ticks. New points use fixed step coordinates, including a
single-step run. The Y maximum is the largest observed PPL, with no extra
headroom; a new maximum rescales the curve, lower values do not reduce the range.
Every JSONL metrics record is consumed, including steps between UI polls.

Client LoRA Update and Gradients Exchange show steady **Idle** before training,
both pulse as **Pending** during startup/waiting, and stop pulsing in other phases.
The four system-status rows have circular icons, equal heights and aligned status
columns. **Disconnected** fits one line.

**Federated Server** reads the coordinator's independent status endpoint and
matches the active run ID. No active session or an unreachable endpoint displays
Disconnected; an old completed run is not presented as the current session.
Aggregation follows Waiting → Aggregating → Complete; the endpoint retains events
because very short aggregation phases can occur between UI polls. Client-update,
gradient and main-server labels describe session progress, not network health probes.

## Verification

From `apps/`, install the optional browser-test dependencies:

```text
npm install
npx playwright install chromium
npm run test:ui
```

To use installed Microsoft Edge instead, set `PLAYWRIGHT_CHANNEL=msedge`.
The check covers full/unknown/single-step axes, incremental dots, dynamic Y maximum,
Cutting Layer, state labels/pulses and aligned status rows. It writes its screenshot
to ignored `apps/build/`. Backend checks live in
`sfl_runtime/tests/test_aggregation_status.py` and `test_coordinator.py`.

Development smoke tests on a physical Android phone completed five cut=1 steps
over USB and wireless ADB. The wireless run took approximately 152 seconds,
finished at PPL 15.27, and saved 18 prefix tensors (4 alignment + 14 block-0 LoRA)
and 210 suffix tensors. Client/server losses matched. This is a functional smoke
test, not a model-quality benchmark or a controlled USB/Wi-Fi speed comparison.
Private logs, screenshots containing device information and trained tensors remain
outside this repository.
