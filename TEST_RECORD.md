# Verification record

Date: 2026-08-25 (Asia/Shanghai)

Source revision: `51c529b` (`fix: bound checkpoint paths for Windows workspaces`)

## Environment

- Host: Windows, Python 3.14.4, CUDA-enabled PyTorch 2.13.0.
- Host GPU: NVIDIA GeForce RTX 4070 Laptop GPU.
- Device: Android arm64-v8a, API 36. No device identifier is recorded.
- Native build: Android NDK r29, arm64-v8a, minimum API 28.
- Model architecture: Gemma 3 270M, obtained separately under its own terms.
- Input: separately supplied WikiText-style UTF-8 smoke text.

## Static and host checks

- Full Python suite: 15 passed.
- Python byte compilation: passed.
- Deterministic release manifest: matched.
- Development integrity gate: passed.
- Android client SHA-256 on host and device:
  `2ec53e3ee02173aa3e979df5e4faf5685bd81dfefa9b12813479684a0037ffb4`.

## Real-device two-round smoke

Configuration: one client, two global rounds, one local step per round,
batch size 1, sequence length 16, cut layer 1, prefix and suffix LoRA rank 8.

The Android client completed rounds 1 and 2 and then received the coordinator
`DONE` state. The phone and host recorded two matching split-step metrics:

| Round | Loss | Token accuracy | RPC bytes | Host suffix duration | Device RPC duration |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 11.8944578 | 0.06666667 | 82,337 | 4.6638 s | 4,692 ms |
| 2 | 15.2203417 | 0.0 | 82,332 | 1.8240 s | 1,837 ms |

Artifacts produced locally but excluded from the release include two prefix
checkpoints with 14 LoRA tensors each and two suffix checkpoints with 238 LoRA
tensors each. Base-model weights and raw data were not placed in checkpoints.

## Issue found during the run

The first attempt completed suffix optimization but failed while writing a
checkpoint because a hash-based tensor filename exceeded the Windows path
limit in the deeply nested workspace. Revision `51c529b` replaced checkpoint
blob names with deterministic bounded ordinals while preserving full tensor
identity in the JSON manifest. A regression test was added, all 15 tests
passed, and the clean two-round run above then completed successfully.
