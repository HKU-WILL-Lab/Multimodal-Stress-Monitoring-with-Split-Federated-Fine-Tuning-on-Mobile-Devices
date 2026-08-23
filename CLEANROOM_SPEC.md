# Clean-room SFL/SplitLoRA functional specification

Status: implementation input, 2026-08-23

## Source boundary

Implementers must not inspect, copy, diff, import, or adapt any source file
outside this directory in the surrounding project. In particular, the current
`src/`, `clients/`, `scripts/`, and `configs/` trees and every EdgeFlowerTune
repository are forbidden implementation sources.

The only permitted code reference is the pristine official MobileFineTuner
checkout at:

`C:/Users/10850/AppData/Local/Temp/mft-cleanroom-reference`

Its pinned commit is `b62d3b12a597e05489e6e8ef025527c613c94837`
and its license is Apache-2.0. Public API/library documentation for Python,
PyTorch, Transformers, PEFT, protobuf, gRPC, CMake, Android NDK, and the
following public papers may also be used:

- SplitLoRA, arXiv:2407.00952
- SplitFed, arXiv:2004.12088

Do not use EdgeFlowerTune as documentation, a test oracle, or a naming source.

## Supported scope

Build one narrowly scoped system:

- Android arm64 C++ client.
- Python GPU suffix service and Python round coordinator on a host PC.
- Gemma 3 270M only.
- Raw WikiText-style UTF-8 text only.
- Causal next-token language-model training only.
- Split learning with a configurable positive decoder cut layer.
- LoRA on both sides of the cut.
- Weighted averaging of client-side prefix LoRA tensors at round boundaries.
- No Flower, `flwr`, EdgeFlowerTune, generic FL baselines, MMLU, multiple
  choice, evaluation suites, quantization, mock backend, or unrelated dataset
  converters.

Model weights and datasets are external runtime assets and must never be
committed or packaged.

## Training behavior

For each client and global round:

1. Fetch the current global prefix-LoRA tensors from the round coordinator.
2. Load those tensors into the client prefix.
3. For each local step, stream and tokenize one private WikiText batch.
4. Execute embeddings and decoder blocks `[0, cut_layer)` on the client.
5. Send the boundary activation, attention mask, and token IDs to the suffix
   service. Raw text must not leave the client.
6. Execute decoder blocks `[cut_layer, layer_count)`, final normalization, and
   LM head on the server.
7. Compute standard shifted-token cross entropy:
   `logits[:, :-1]` predicts `token_ids[:, 1:]`; padded targets are ignored.
8. Backpropagate on the server, update server-side suffix LoRA, and return only
   the float32 gradient with respect to the boundary activation plus scalar
   metrics.
9. Backpropagate that gradient through the client prefix and update only
   client-side prefix LoRA.
10. Submit the resulting prefix LoRA and the number of processed sequences to
    the coordinator.
11. Once the configured client quorum submits, compute a sample-count-weighted
    mean independently for every named tensor and advance the round.

Base-model parameters and token embeddings remain frozen. Server suffix LoRA
persists across split steps. Prefix tensors must be identified by stable,
explicit names; positional tensor ordering alone is not acceptable.

## New wire contract

Define a new protobuf schema under `proto/`. Do not use handwritten magic-byte
messages, NPY payloads, Flower protobufs, or pickle.

At minimum, define:

- A tensor message containing name, shape, dtype, and raw little-endian data.
- A split-step request containing protocol version, client ID, round, local
  step, cut layer, activation tensor, token-ID tensor, and attention-mask
  tensor.
- A split-step response containing the activation-gradient tensor, loss,
  token accuracy, and server step.
- Coordinator RPCs for fetching a round, bootstrapping the first prefix state,
  and submitting one client update.
- Explicit done/wait/train states and actionable gRPC status errors.

All RPCs must use configurable deadlines and message-size limits. Validate
tensor dtype, rank, dimensions, byte length, finite floating-point values,
round number, and tensor-name/shape consistency before use.

## MobileFineTuner integration

Depend on MobileFineTuner as an external Apache-2.0 source/package; do not
vendor it into this repository. Design a minimal, generally named hidden-state
execution API that lets a Gemma model execute only a requested prefix span and
avoid allocating/loading the unused final norm, LM head, and suffix blocks on
the client. Supply this as a small documented patch against the pinned pristine
commit, with prominent modified-file notices required by Apache-2.0.

The Android client must use MobileFineTuner tensors, autograd, tokenizer, LoRA,
and optimizer facilities through that permitted checkout. It must load only
the embedding and layers below the cut.

## Data behavior

Read UTF-8 text lazily rather than tokenizing an entire dataset at startup.
Assign source lines to clients by deterministic round-robin line index. Join
tokens into fixed-length sequences, preserve a small carry buffer between
reads, pad only the final short sequence when necessary, and produce an
attention mask. Seeded sequence shuffling is optional; behavior must be
documented and deterministic when enabled.

## Metrics and artifacts

Write append-only JSON Lines records using a newly defined schema. Each record
must include run ID, client ID, global round, local step, batch size, sequence
length, loss, token accuracy, split RPC bytes, duration, and available host GPU
memory. Checkpoints must contain only LoRA tensors and metadata, never base
weights or raw samples.

## Required repository contents

- `proto/`: original protobuf schema.
- `python/sfl_clean/`: suffix service, coordinator, configuration, tensor
  codec, metrics, and CLI modules.
- `android/`: original C++ client and CMake build.
- `mft_patch/`: patch plus application/verification script for the pinned
  MobileFineTuner commit.
- `configs/`: one single-phone two-round smoke configuration.
- `scripts/`: only build, stage, run, and integrity-check scripts required by
  this system.
- `tests/`: protocol validation, weighted aggregation, round-state behavior,
  shifted-token objective, and an end-to-end random-small-model smoke test.
- `LICENSE`: Apache License 2.0 for newly authored project code.
- `THIRD_PARTY_NOTICES.md`, `CITATION.cff`, `PROVENANCE.md`, and a concise
  `README.md`.

Generated protobuf files, build output, virtual environments, Python caches,
models, tokenizers, datasets, logs, checkpoints, and device identifiers must be
ignored and absent from the release manifest.

## Acceptance checks

1. Repository-wide case-insensitive scans find no `flower`, `flwr`, or
   `edgeflowertune` in code, config, or build files. The provenance document may
   state the forbidden-source boundary without naming a repository if needed
   for a strict zero-string release scan.
2. Python supports the newest stable version compatible with pinned runtime
   dependencies and passes import, compile, and test checks.
3. The C++ client builds for Android arm64 using NDK r29 and gRPC v1.83 or a
   newer compatible stable release.
4. A real phone completes two rounds with cut layer 1, batch size 1, sequence
   length 16, and one local step per round.
5. Metrics prove two suffix forward/backward steps and two coordinator updates.
6. A fresh checkout plus documented external prerequisites reproduces the
   build without any file from the surrounding legacy project.

