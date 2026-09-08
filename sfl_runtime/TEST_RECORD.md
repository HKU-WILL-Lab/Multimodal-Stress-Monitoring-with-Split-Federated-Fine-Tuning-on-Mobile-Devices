# Verification record

Last updated: 2026-09-07 (Asia/Shanghai)

## Verified on this branch

- Python protocol, objective, alignment, dataset, coordinator, suffix, and
  embedding-export tests: 24 passed, including a rerun from the sanitized
  source-release copy.
- Android application shared-module tests and headless phone/wear Kotlin
  compilation: passed with the demo watch-sensor flavor after the unfinished
  Compose UI was removed from the public source boundary.
- Android arm64 native command-line client: compiled with NDK r29.
- Android arm64 JNI library (`libwellbeing_sfl.so`): compiled and linked with
  gRPC, Protocol Buffers, MobileFineTuner operators, ExecuTorch, and XNNPACK.
- Android arm64 local-inference JNI library (`libwellbeing_inference.so`):
  compiled and linked separately with MobileFineTuner operators, ExecuTorch,
  XNNPACK, and the ExecuTorch tokenizer runtime.
- Headless phone APK: compiled with both native libraries packaged under
  `lib/arm64-v8a`; synthetic Watch demo APK and shared application tests passed.
- Architecture: Llama 3.2 1B with decoder block 0 on the phone
  (`cut_layer=1`) and decoder blocks 1--15 plus the output head on the main
  server.
- Mobile input: 30-second, six-channel windows resampled to 240 time points;
  a frozen ExecuTorch NormWear encoder exports 162 tokens of width 768.
- Alignment: the OpenTSLM-SP token-wise projector applies LayerNorm(768),
  Linear(768, 2048), and GELU before inserting sensor tokens into the LLM
  input sequence.
- Client training: the projector and decoder-block-0 LoRA adapters receive
  gradients through the split boundary. Token embeddings, encoder parameters,
  and base Llama weights remain frozen.
- Full prepared K-EmoCon sensor asset: 560 labeled 30-second records.

## Real-device result

The complete one-client, one-round training path was executed successfully on
a connected OnePlus PLK110 phone (Android API 36). The phone ran the frozen
30-second encoder, OpenTSLM-SP projector, Llama embeddings and decoder block 0,
exchanged the boundary activation and gradient with the Python main server,
updated its local trainable parameters, and completed aggregation with the
federated server. The initial smoke step reported loss 7.8593 and token
accuracy 0.0370; these values verify execution only and are not model-quality
results.

On 2026-09-07, a subsequent five-step real-phone run used five distinct
records from the participant-disjoint training split. It completed the round
and checkpointed both sides successfully. The client-reported split-RPC times
were 1616, 651, 518, 495, and 759 ms; these exclude encoder and client-side
Llama execution and are not end-to-end step latencies.

The prepared dataset contains 440 training records (22 participants), 60
validation records (3 participants), and 60 test records (3 participants),
with no participant overlap. Exact participant IDs are recorded in
`configs/kemocon_participant_split.json`. Generated-asset hashes remain local
and are intentionally not distributed in this source release.

## Not yet claimed

This verification does not record a completed real-device inference run.
The local inference runtime and app integration compile, but executing Llama
still requires a decoder `.pte` whose
public input is float32 embeddings (rather than an ordinary token-ID model), a
tokenizer, the trained alignment checkpoint, and model/encoder assets. Those
inference assets have not yet been exported as a matched deployment bundle.
The proprietary Samsung Health Sensor integration likewise remains pending its
separately licensed SDK archive.
