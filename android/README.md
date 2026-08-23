# Android arm64 client

The native client consumes the same strict JSON file as both Python host
services. Device-specific values that do not belong in the shared research
configuration remain explicit CLI inputs: client ID, local model directory,
private dataset, and deterministic partition index/count. Host bind addresses
from the shared file are converted to loopback targets for `adb reverse`.

The client loads only the Gemma token embedding and decoder layers below the
positive cut. It attaches LoRA to q/k/v/o and gate/up/down projections, exports
them by stable fully qualified names, and applies only coordinator states with
an exact name/shape match. Raw text is tokenized lazily on the phone and never
enters an RPC.

## Exact Windows build and smoke commands

From the clean-room repository root, build and test the host environment:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_host.ps1
```

Build host protobuf tools, cross-compile/install gRPC v1.83.0 for arm64, apply
the pinned MobileFineTuner patch to a generated copy, and link the client:

```powershell
$shortRoot = [System.IO.Path]::GetFullPath((Join-Path $PWD 'build\android-short'))
New-Item -ItemType Directory -Force $shortRoot | Out-Null
subst S: $shortRoot
powershell -ExecutionPolicy Bypass -File scripts/build_android.ps1 `
  -GrpcSource C:/Users/10850/grpc-src `
  -NdkRoot D:/functional_exe/android-ndk-r29-windows/android-ndk-r29 `
  -MobileFineTunerSource C:/Users/10850/AppData/Local/Temp/mft-cleanroom-reference `
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

Start the services in two host terminals:

```powershell
.venv/Scripts/python.exe -m sfl_clean.cli_coordinator --config configs/smoke_single_phone.json
.venv/Scripts/python.exe -m sfl_clean.cli_suffix --config configs/smoke_single_phone.json
```

Stage the binary and separately obtained runtime assets, then run two rounds:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/stage_android.ps1 `
  -ClientBinary build/android-short/client-arm64/sfl_android_client `
  -ModelDir C:/external/gemma-3-270m `
  -Dataset C:/external/wikitext-2-raw/wiki.train.raw

powershell -ExecutionPolicy Bypass -File scripts/run_android_smoke.ps1
```

Use `-Serial DEVICE_SERIAL` with both device scripts when multiple devices are
attached. Model/tokenizer files, dataset content, generated bindings, metrics,
and checkpoints are runtime artifacts and must not be committed.
