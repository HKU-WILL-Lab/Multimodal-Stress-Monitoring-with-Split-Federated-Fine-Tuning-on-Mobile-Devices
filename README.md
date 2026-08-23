# Clean Split-LoRA for Android

This repository contains a narrowly scoped research implementation of split
causal-language-model fine-tuning for Gemma 3 270M. An Android arm64 client
runs embeddings and decoder layers below a configurable cut; a Python GPU
service runs the remaining layers and language-model objective. A separate
coordinator aggregates named client-side LoRA tensors at round boundaries by
processed-sequence count.

The implementation is clean-room work. Its source boundary and reproducible
inputs are recorded in [PROVENANCE.md](PROVENANCE.md).

## Supported workload

- Gemma 3 270M only.
- Android arm64 client and a host PC with a compatible GPU.
- Raw WikiText-style UTF-8 input, streamed on the phone.
- Causal next-token training with a positive decoder cut layer.
- LoRA adapters on both sides of the cut.
- Sample-count-weighted aggregation of explicitly named prefix tensors.
- Protocol Buffer messages over gRPC; raw text remains on the client.

This is research software, not a claim of production privacy or security.
Boundary activations and gradients can reveal information and should be
protected in transit and handled according to the deployment's threat model.

## External prerequisites

- Python and the pinned dependencies declared by the Python package metadata.
- A CUDA-capable host environment suitable for the selected PyTorch build.
- Android NDK r29 and an arm64 Android device.
- gRPC v1.83 or a newer compatible stable release and Protocol Buffers.
- A pristine external MobileFineTuner checkout at commit
  `b62d3b12a597e05489e6e8ef025527c613c94837`.
- Separately obtained Gemma 3 270M weights/tokenizer and private UTF-8 text.

MobileFineTuner is patched during setup but is not vendored here. Apply and
verify the documented patch in `mft_patch/` against the exact commit above.
Model weights, tokenizer assets, datasets, generated bindings, device-specific
configuration, logs, and checkpoints are intentionally absent.

## Layout

| Path | Purpose |
| --- | --- |
| `proto/` | Original wire contract; generated bindings are build artifacts. |
| `python/sfl_clean/` | GPU suffix service, coordinator, codecs, metrics, and CLIs. |
| `android/` | Android arm64 C++ client and build files. |
| `mft_patch/` | External dependency patch and verification tooling. |
| `configs/` | Single-phone, two-round smoke configuration. |
| `tests/` | Protocol, aggregation, round-state, objective, and smoke tests. |
| `scripts/` | Build/run support and release-integrity tooling. |

## Release integrity

Release manifest tooling has no third-party Python dependency:

```console
python scripts/release_manifest.py --write
python scripts/release_manifest.py --check
python scripts/integrity_check.py
```

Before publishing, replace the placeholder author values in `CITATION.cff`,
then run the stricter gate:

```console
python scripts/integrity_check.py --release
```

The release manifest excludes the clean-room input specification, external
assets, generated code, build/cache output, local credentials, device
configuration, logs, and checkpoints. Regenerate it only from a clean source
tree after all source changes are complete.

## Research references

- [SplitLoRA, arXiv:2407.00952](https://arxiv.org/abs/2407.00952)
- [SplitFed, arXiv:2004.12088](https://arxiv.org/abs/2004.12088)
- [MobileFineTuner pinned source](https://github.com/Edge-Intelligence-Lab/MobileFineTuner/tree/b62d3b12a597e05489e6e8ef025527c613c94837)

Newly authored project code is licensed under Apache License 2.0. External
software, model assets, and datasets retain their own terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
