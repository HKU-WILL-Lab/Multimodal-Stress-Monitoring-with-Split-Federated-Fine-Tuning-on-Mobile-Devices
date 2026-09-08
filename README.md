# MobiWellbeing

MobiWellbeing is a research prototype for proactive emotional-state estimation
from wearable signals. It combines a smartwatch and phone application with a
time-series-conditioned split federated learning (SFL) system for LoRA
fine-tuning of Llama 3.2 1B.

During training, a phone executes a frozen time-series encoder, a trainable
modality-alignment module, and the first decoder block. A main server executes
the remaining decoder blocks and language-model head, while a logically
separate federated server coordinates rounds and aggregates client-side
trainable state. Raw sensor windows remain on the phone; intermediate
activations and gradients cross the split boundary.

The demo application maps predicted valence and arousal to five user-facing
emotional states, presents a recent trend and short assessment, and sends a
watch vibration for negative states. It is a non-clinical research demo, not a
medical device or diagnostic tool.

## Repository layout

| Path | Contents |
| --- | --- |
| `sfl_runtime/` | C++ mobile client, Python servers, protocol, exporters, and tests |
| `apps/` | Headless phone/Watch8 runtime, sensing, transport, and shared logic |
| `configs/` | Publication-safe configuration templates |
| `docs/` | Reproduction and release documentation |
| `scripts/` | Source-release audit and deterministic manifest tools |

## What is included

- Independently implemented SFL protocol, suffix service, federated
  coordinator, serialization, metrics, and checkpoint logic.
- Android arm64 C++ training path using external MobileFineTuner operators.
- ExecuTorch encoder bridge and extensible sensor-to-LLM alignment interface.
- Llama 3.2 1B split-training support with the first decoder block on-device.
- Headless phone and Watch8 application modules, including training/inference
  bridges, sensor transport, vibration alerts, and a synthetic sensor flavor.
- Unit and integration tests that do not require private model or dataset files.

## What is not included

This source repository intentionally excludes the unfinished phone/watch UI,
model weights, tokenizers,
K-EmoCon data, exported `.pte` programs, `.sflsensor` files, checkpoints,
Samsung SDK binaries, native libraries, APKs, credentials, device identifiers,
and training logs. Obtain each external dependency or asset under its own
terms and generate local runtime files using the documented scripts.

In particular, Llama 3.2 materials remain subject to Meta's separate terms. Do
not add them to the Git history.

## Current verification status

- Python protocol/server tests: verified locally.
- Native Android SFL training smoke path: verified on a connected phone with
  locally supplied assets and servers.
- Headless phone and synthetic-watch application builds: verified locally.
- Full post-training phone inference: implementation is present, but a final
  merged LoRA decoder exported for embedding input and the matching trained
  alignment checkpoint are still required for an end-to-end model run.
- Physical Samsung sensor capture: requires the separately obtained Samsung
  Health Sensor SDK and the applicable Samsung approval/configuration.

See [sfl_runtime/README.md](sfl_runtime/README.md) and
[apps/README.md](apps/README.md) for build details. The recorded checks are in
[sfl_runtime/TEST_RECORD.md](sfl_runtime/TEST_RECORD.md).

## Relationship to related projects

The client-side LLM training engine builds on the separately obtained
[MobileFineTuner](https://github.com/Edge-Intelligence-Lab/MobileFineTuner).
[EdgeFlowerTune](https://github.com/Edge-Intelligence-Lab/EdgeFlowerTune)
provides a broader Flower-based platform for heterogeneous federated LLM
fine-tuning and could also support this class of application. This repository
uses its own focused SFL protocol and does not require EdgeFlowerTune or Flower
at build time or runtime.

## Releasing safely

Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md),
[PROVENANCE.md](PROVENANCE.md), and
[RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) before publishing or distributing
binaries. Then run:

```powershell
py -3 scripts/release_manifest.py --write
py -3 scripts/release_check.py
```

## Usage rights

This repository is currently a public source snapshot, not an open-source
release. No license is granted for the newly authored project source. Unless
applicable law provides otherwise, permission is required before copying,
modifying, redistributing, or incorporating it into another project. External
dependencies, datasets, model materials, and generated artifacts remain under
their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
