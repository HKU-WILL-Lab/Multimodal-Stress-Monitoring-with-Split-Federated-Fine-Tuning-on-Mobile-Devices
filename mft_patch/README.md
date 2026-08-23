# MobileFineTuner hidden-span patch

This directory contains a minimal patch for pristine MobileFineTuner commit
`b62d3b12a597e05489e6e8ef025527c613c94837` (Apache-2.0).

The patch adds `GemmaHiddenSpanConfig` and
`GemmaModel::forward_hidden_states`. A prefix consumer can allocate the token
embedding and only decoder layers in `[0, cut_layer)`, without allocating the
final normalization, language-model head, or suffix blocks. The historical
constructor and full causal-LM `forward` behavior remain the default. Default
Gemma LoRA injection is limited to layers owned by the configured span.

Every modified upstream file carries a prominent modified-file notice. The
patch does not copy MobileFineTuner into this repository; build scripts apply it
to a generated working copy of the pinned pristine checkout.

Apply and verify on Windows:

```powershell
./mft_patch/apply_and_verify.ps1 `
  -SourceRoot C:/path/to/mft-cleanroom-reference `
  -Destination ./build/mft-patched
```

The script refuses a source checkout at any other commit and never modifies the
source checkout itself.
