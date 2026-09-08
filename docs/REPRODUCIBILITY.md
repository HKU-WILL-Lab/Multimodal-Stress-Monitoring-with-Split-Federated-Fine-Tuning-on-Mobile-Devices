# Reproducibility boundary

The repository is a source release, not a model or dataset archive. A complete
reproduction requires separately obtaining:

1. the pinned MobileFineTuner source and compatible ExecuTorch/gRPC toolchains;
2. Llama 3.2 1B weights and tokenizer under Meta's applicable terms;
3. a time-series encoder implementation/checkpoint that the user is authorized
   to use and export;
4. K-EmoCon or another lawfully obtained labeled sensor dataset; and
5. Samsung Health Sensor SDK for the optional physical Watch8 flavor.

Generate all model, dataset, protocol-binding, checkpoint, and native-library
artifacts locally. They belong in ignored `runtime/` or build directories and
must not be added to commits.

The included K-EmoCon split file records a participant-disjoint experimental
partition but contains no samples. The test record reports the checks performed
on the development machine; it should not be interpreted as a guarantee that a
different hardware/software combination will produce identical performance.
