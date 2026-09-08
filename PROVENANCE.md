# Source provenance

## Scope

This record applies only to source files distributed in this repository. Model
weights, tokenizers, datasets, exported programs, checkpoints, locally built
binaries, and SDK packages are outside its scope and are not distributed.

## SFL implementation boundary

The SFL protocol, tensor serialization, suffix service, federated coordinator,
client training control flow, metrics, and tests under `sfl_runtime/` were
implemented from a functional specification and public papers. They do not
copy, vendor, import, or require EdgeFlowerTune or Flower source code. The
implementation uses a dedicated gRPC protocol and Python services.

MobileFineTuner is an external build dependency for mobile-native LLM
execution, backpropagation, and LoRA optimization. Development used a pristine
official checkout pinned to commit
`b62d3b12a597e05489e6e8ef025527c613c94837`. It is not vendored or modified in
this repository.

## Time-series integration

The encoder bridge's `.pte` loading and tensor I/O contract was informed by a
collaborator-provided CppTorchEncoder prototype. That prototype is not included.
The exported encoder remains frozen during SFL training.

The repository implements a general sensor-to-LLM alignment interface. Its
included LayerNorm--Linear--GELU configuration follows the published
OpenTSLM-SP architecture. The implementation in this repository is newly
written; OpenTSLM source and model files are not included. The optional encoder
export helper loads a separately obtained NormWear/OpenTSLM implementation
supplied by the user at runtime.

## Application source

The Android and Wear OS application source under `apps/` was written for this
demo. Google Stitch was used as a visual design handoff only; no generated HTML
or Stitch runtime is embedded. Samsung Health Sensor SDK is an optional,
external binary dependency and is not included.

## Authorship requirement

Before public release, every person or institution that owns copyright in
contributed source must approve its public distribution. A future reuse license
will require an additional licensing decision by the relevant copyright
owners. This provenance record documents technical origin; it is not a
substitute for that approval or for legal review by the authors' institutions.
