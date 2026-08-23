# Third-party notices

This project is original clean-room work distributed under Apache License 2.0.
It does not bundle the external software or runtime assets listed below.

## MobileFineTuner

The Android client depends on an externally obtained pristine checkout of
[MobileFineTuner](https://github.com/Edge-Intelligence-Lab/MobileFineTuner) at
commit `b62d3b12a597e05489e6e8ef025527c613c94837`.

MobileFineTuner is licensed under Apache License 2.0 and carries the notice:

> Copyright 2024 Mobile LLM Fine-Tuning Project Contributors

The patch in `mft_patch/` is distributed separately from the upstream source.
When applied, changed upstream files must retain applicable notices and carry
prominent modification notices as required by Apache License 2.0. No upstream
checkout or built binary is bundled. A textual patch can necessarily contain
limited upstream context; that material remains covered by the upstream
Apache-2.0 license and notices.

## Research references

The design cites, but does not copy code or text from:

- [SplitLoRA, arXiv:2407.00952](https://arxiv.org/abs/2407.00952)
- [SplitFed, arXiv:2004.12088](https://arxiv.org/abs/2004.12088)

The papers and their associated artifacts remain subject to their respective
authors' and publishers' terms.

## Runtime assets

Gemma model weights and tokenizer files are external runtime assets. They are
not covered by this project's Apache-2.0 license and must be obtained and used
under the model provider's applicable terms.

Input datasets are supplied by the user and are never redistributed by this
project. Users are responsible for confirming that their data license and
handling practices permit the intended use.

Python, C++, Android, gRPC, Protocol Buffers, PyTorch, Transformers, PEFT, and
other runtime/build dependencies retain their own licenses. The resolved lock
files or dependency metadata used for a particular build are authoritative for
the exact dependency set; this notice does not replace those licenses.
