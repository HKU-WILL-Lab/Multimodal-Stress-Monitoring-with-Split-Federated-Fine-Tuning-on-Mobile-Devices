# Clean Split-LoRA for Android

This repository contains a narrowly scoped research implementation of
time-series-conditioned split fine-tuning for Llama 3.2 1B. A mobile C++
client executes a frozen `.pte` encoder through ExecuTorch, trains a pluggable
sensor-to-LLM projector, and exchanges boundary activations/gradients with a
Python/PyTorch main server. A separate federated server performs
sample-weighted aggregation of the client-side trainable state.

The SFL implementation was independently authored. External dependencies and
assets are listed in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## Supported workload

- Llama 3.2 1B/Instruct only.
- Android arm64 client and a host PC with a compatible GPU.
- Pretokenized labeled sensor JSONL generated with the matching Llama tokenizer.
- Frozen 30-second ExecuTorch NormWear encoder that emits 162 sensor tokens.
- OpenTSLM-SP alignment (`LayerNorm(768) -> Linear(768, 2048) -> GELU`).
- LoRA adapters in the client-side first decoder block and main-server suffix.
- Protocol Buffer messages over gRPC; raw sensor windows remain on the client.

The integrated native path uses `cut_layer=1`: the phone runs the embedding,
OpenTSLM-SP alignment, and Llama 3.2 1B decoder block 0, while the main server
runs blocks 1--15 and the language-model head. The projector and block-0 LoRA
state participate in client-side aggregation; the frozen base block weights
remain local and are never uploaded.

This is research software, not a claim of production privacy or security.
Boundary activations and gradients can reveal information and should be
protected in transit and handled according to the deployment's threat model.

## External prerequisites

- Python and the pinned dependencies declared by the Python package metadata.
- A CUDA-capable host environment suitable for the selected PyTorch build.
- Android NDK r29 and an arm64 Android device.
- gRPC v1.83.0 and its compatible Protocol Buffers dependency.
- A pristine external MobileFineTuner checkout at commit
  `b62d3b12a597e05489e6e8ef025527c613c94837`.
- A compatible ExecuTorch checkout and separately exported time-series `.pte`.
- Separately obtained Llama 3.2 1B weights/tokenizer and labeled sensor data.

Export the client-only token embedding and decoder-block-0 asset with
`python -m sfl_clean.export_llama_embedding --model-dir <checkpoint> --output-dir <mobile-asset>`.
Only the remaining decoder blocks and output head stay on the main server.

For post-training inference, the phone loads a full decoder exported for
floating-point input embeddings, the same frozen encoder, the aggregated
alignment checkpoint, and the client-only embedding table. The deployment
contract is documented by `configs/inference_deployment.example.json`.
Standard token-ID Llama `.pte` files are intentionally rejected: the decoder
must expose `[1, sequence, 2048]` input embeddings so projected sensor tokens
enter the model without being approximated as vocabulary tokens.

Once these separately licensed/generated assets exist, create one deployable
directory with `scripts/stage_inference.ps1`. The generated `deployment.json`
uses relative paths, so the directory can be copied intact to the phone.

Prepare labeled sensor records with the same tokenizer using
`python -m sfl_clean.sensor_dataset --input <records.jsonl> --output <training.sflsensor> --model <checkpoint> --sequence-length <N>`.

For K-EmoCon experiments, export `train`, `validation`, and `test` separately
with `python -m sfl_clean.kemocon_dataset ... --split <name>`. The fixed
participant-disjoint protocol is recorded in
`configs/kemocon_participant_split.json`; no participant occurs in more than
one split. The publication-safe example runs eight rounds with 440 local steps
per round and saves the main-server LoRA state once per round
(`configs/training.example.json`). Adjust those values to the actual prepared
training split rather than treating them as dataset-independent defaults.

MobileFineTuner is not vendored here. Use the unmodified Apache-2.0 checkout
at the exact commit above.
Model weights, tokenizer assets, datasets, generated bindings, device-specific
configuration, logs, and checkpoints are intentionally absent.

## Layout

| Path | Purpose |
| --- | --- |
| `proto/` | Original wire contract; generated bindings are build artifacts. |
| `python/sfl_clean/` | GPU suffix service, coordinator, codecs, metrics, and CLIs. |
| `android/` | Android arm64 C++ client and build files. |
| `configs/` | Publication-safe server, client, split, and inference examples. |
| `tests/` | Protocol, aggregation, round-state, objective, and smoke tests. |
| `scripts/` | Build and deployment helpers. |

## Research references

- [OpenTSLM: time-series/LLM alignment](https://arxiv.org/abs/2510.02410)
- [NormWear: pretrained wearable-signal encoder](https://doi.org/10.1145/3803808)
- [K-EmoCon: experimental dataset](https://doi.org/10.1038/s41597-020-00630-y)
- [MobileFineTuner: client-side training engine](https://arxiv.org/abs/2512.08211)
- [SplitLoRA, arXiv:2407.00952](https://arxiv.org/abs/2407.00952)
- [SplitFed, arXiv:2004.12088](https://arxiv.org/abs/2004.12088)
- [MobileFineTuner pinned source](https://github.com/Edge-Intelligence-Lab/MobileFineTuner/tree/b62d3b12a597e05489e6e8ef025527c613c94837)

Author credits, official project links, and dependency notices are recorded in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

No reuse license is currently granted for newly authored project code.
External software, model assets, and datasets retain their own terms; see
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
