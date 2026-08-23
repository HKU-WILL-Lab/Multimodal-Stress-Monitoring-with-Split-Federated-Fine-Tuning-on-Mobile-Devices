# Provenance and clean-room boundary

## Scope of this record

This record applies to the source release rooted in this directory. It records
the information boundary used to create the implementation; it is not a
statement about separately obtained model weights, datasets, generated files,
or locally built binaries.

## Boundary

The implementation was authored from a written functional specification
without inspecting, copying, diffing, importing, adapting, or testing against
source code from the surrounding legacy project or from unapproved related
implementations. Those sources were not used as documentation, naming input,
or a behavioral oracle.

The written functional specification is a development input, not a release
artifact. `scripts/release_manifest.py` excludes it from distributable source
manifests so that the released tree contains only implementation and release
materials.

## Permitted inputs

The permitted inputs were limited to:

1. The clean-room functional specification dated 2026-08-23.
2. A pristine official
   [MobileFineTuner](https://github.com/Edge-Intelligence-Lab/MobileFineTuner)
   checkout pinned to commit
   `b62d3b12a597e05489e6e8ef025527c613c94837`, licensed under Apache-2.0.
3. Public, official documentation for Python, PyTorch, Transformers, PEFT,
   Protocol Buffers, gRPC, CMake, and the Android NDK.
4. The public papers [SplitLoRA, arXiv:2407.00952](https://arxiv.org/abs/2407.00952)
   and [SplitFed, arXiv:2004.12088](https://arxiv.org/abs/2004.12088).

MobileFineTuner remains an external dependency. It is not vendored into this
release. The separately authored patch targets only the pinned pristine commit
and must be applied with the verification procedure supplied in `mft_patch/`.

## Authorship and artifact policy

All project source, protocol definitions, tests, configuration examples,
scripts, and documentation in the release manifest are newly authored for
this implementation unless a file explicitly says otherwise. Generated
Protocol Buffer bindings and build products are not source artifacts and are
not included.

Model weights, tokenizer assets, datasets, raw samples, logs, metrics,
checkpoints, secrets, and device identifiers are external or local runtime
state. They must not be committed, packaged, or recorded in the release
manifest. Checkpoints produced by the system are expected to contain only LoRA
tensors and non-sensitive metadata, but remain excluded as runtime output.

## Reproduction and audit

A release audit should:

1. obtain the MobileFineTuner checkout directly from its official upstream;
2. verify that its checkout equals the pinned commit above;
3. apply and verify the documented patch;
4. obtain model and dataset assets separately under their applicable terms;
5. generate Protocol Buffer bindings and build outputs locally; and
6. run `python scripts/integrity_check.py --release` before packaging.

The generated `RELEASE_MANIFEST.json` records the relative path, byte length,
and SHA-256 digest of each distributable source file. It intentionally omits
the manifest itself so that it can be regenerated deterministically.
