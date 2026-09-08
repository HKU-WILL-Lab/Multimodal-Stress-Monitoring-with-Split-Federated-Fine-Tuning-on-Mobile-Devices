# Android arm64 client

The native client consumes the same strict JSON file as both Python host
services. Device-specific values that do not belong in the shared research
configuration remain explicit CLI inputs: client ID, local model directory,
private dataset, and deterministic partition index/count. Host bind addresses
from the shared file are converted to loopback targets for `adb reverse`.

The current Llama client splits after decoder block 0. It loads the frozen
Llama 3.2 1B token embedding and first-block weights, runs a frozen 30-second
NormWear encoder with ExecuTorch, and applies the OpenTSLM-SP token-wise
projector (`LayerNorm -> Linear -> GELU`). Decoder block 0 and its LoRA adapters
run on the phone; the server receives its boundary activations, never the raw
sensor window. Dataset records are pretokenized with the exact Llama tokenizer
so Python and C++ cannot silently disagree about token IDs.

Build with `SFL_ENABLE_EXECUTORCH_ENCODER=ON` and set
`EXECUTORCH_SOURCE_DIR` to the same ExecuTorch checkout used during `.pte`
export. The configured encoder contract is
`[batch, 6, 240] -> [batch, 162, 768]`, corresponding to a 30-second window
resampled to 8 Hz. Both shape and element count are checked when the `.pte`
program is executed.

## Exact Windows build and smoke commands

From the clean-room repository root, build and test the host environment:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_host.ps1
```

Build host protobuf tools, cross-compile/install gRPC v1.83.0 for arm64, and
link the client to the pristine pinned MobileFineTuner checkout:

```powershell
$shortRoot = [System.IO.Path]::GetFullPath((Join-Path $PWD 'build\android-short'))
New-Item -ItemType Directory -Force $shortRoot | Out-Null
subst S: $shortRoot
powershell -ExecutionPolicy Bypass -File scripts/build_android.ps1 `
  -GrpcSource C:/external/grpc-src `
  -NdkRoot D:/functional_exe/android-ndk-r29-windows/android-ndk-r29 `
  -MobileFineTunerSource C:/external/MobileFineTuner `
  -ExecuTorchSource C:/external/executorch `
  -BuildRoot S:/ `
  -BuildJobs 8
```

The short drive is intentional: MSVC can otherwise exceed its object-file path
limit while building host protobuf. It maps to `build/android-short`, so all
generated files remain inside the clean-room tree. Host tools use `nmake.exe`
from VS 2022 Build Tools. Android targets use the NDK r29-bundled GNU Make 4.3
with bounded parallelism. Ninja is not required. After all build and staging
commands, remove the temporary mapping with `subst S: /d`.

The default `-ApiLevel 28` is the native binary's minimum supported Android
API and is expected to run on the confirmed API 36 phone. To require API 36
symbols instead, add `-ApiLevel 36`; this is optional and is not the setting
used for the default artifact documented above.

The same build also produces two JNI libraries under
`S:/app-jni/arm64-v8a`: `libwellbeing_sfl.so` for the Training page and
`libwellbeing_inference.so` for the Inference page. They are separate because
gRPC and the ExecuTorch tokenizer vendor incompatible copies of several
support libraries when linked into one shared object.

Start the services in two host terminals:

```powershell
.venv/Scripts/python.exe -m sfl_clean.cli_coordinator --config configs/smoke_single_phone.json
.venv/Scripts/python.exe -m sfl_clean.cli_suffix --config configs/smoke_single_phone.json
```

Export the minimal Llama client asset from a separately obtained checkpoint:

```powershell
.venv/Scripts/python.exe -m sfl_clean.export_llama_embedding `
  --model-dir C:/external/Llama-3.2-1B-Instruct `
  --output-dir C:/external/Llama-3.2-1B-mobile-embedding
```

Export the full-token NormWear program from a separately obtained NormWear
checkpoint (the earlier mean-pooled 15-second `.pte` is incompatible):

```powershell
.venv/Scripts/python.exe -m sfl_clean.export_normwear_tokens `
  --opentslm-source ../Figure3/OpenTSLM/src `
  --checkpoint C:/external/normwear-checkpoint.pt `
  --output C:/external/normwear-30s-tokens.pte
```

Stage the binary and separately obtained runtime assets, then run two rounds:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/stage_android.ps1 `
  -ClientBinary build/android-short/client-arm64/sfl_android_client `
  -ModelDir C:/external/Llama-3.2-1B-mobile-embedding `
  -Dataset C:/external/training.sflsensor

powershell -ExecutionPolicy Bypass -File scripts/run_android_smoke.ps1
```

Use `-Serial DEVICE_SERIAL` with both device scripts when multiple devices are
attached. Model/tokenizer files, dataset content, generated bindings, metrics,
and checkpoints are runtime artifacts and must not be committed.

For local inference, stage the embedding-input decoder `.pte`, tokenizer,
encoder, embedding table, and aggregated alignment checkpoint as one directory:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/stage_inference.ps1 `
  -DecoderPte C:/external/llama-sensor-decoder.pte `
  -Tokenizer C:/external/Llama-3.2-1B-Instruct/tokenizer.json `
  -EncoderPte C:/external/sensor-encoder.pte `
  -EmbeddingDirectory C:/external/Llama-3.2-1B-mobile-embedding `
  -AlignmentCheckpoint C:/external/prefix-round-0002 `
  -OutputDirectory C:/external/mobihoc-inference
```

Copy that directory intact to the app-specific path shown on the phone's
Inference page, then press **Load local model**. An ordinary token-ID Llama
`.pte` is not compatible with the sensor-token interface and is rejected at
load time.

With an authorized USB device, `scripts/deploy_inference.ps1 -StagedDirectory
C:/external/mobihoc-inference` performs this copy. Add `-Serial` when more than
one Android device is connected.
