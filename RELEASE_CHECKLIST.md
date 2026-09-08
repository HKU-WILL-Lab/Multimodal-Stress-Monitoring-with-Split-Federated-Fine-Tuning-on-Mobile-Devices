# Public release checklist

Complete every blocking item before making the GitHub repository public.

## Human approval

- [ ] Revoke the OpenRouter credential found in the private development tree;
      never copy it or the containing `Figure3` directory here.
- [ ] Confirm that all relevant copyright owners authorize public distribution
      of this no-license source snapshot.
- [ ] Decide on a reuse license and citation metadata before describing a future
      release as open source.
- [ ] Confirm any university or sponsor disclosure requirements.

## Source and asset boundary

- [ ] Confirm that unfinished UI screens and visual assets remain outside this
      release while phone/watch runtime code stays included.
- [ ] Keep model weights, tokenizers, `.pte`, `.sflsensor`, checkpoints,
      datasets, Samsung SDK packages, APKs, native libraries, logs, and device
      identifiers outside Git.
- [ ] Review `THIRD_PARTY_NOTICES.md` against the exact dependencies resolved by
      the release build.
- [ ] If distributing an APK or model separately, prepare a separate binary
      notice bundle and review all model/vendor terms; source-only approval is
      not sufficient.
- [ ] Confirm that example configuration contains no personal absolute paths,
      hostnames, IP addresses, account names, or credentials.

## Verification

- [ ] Run the Python tests under `sfl_runtime/tests`.
- [ ] Run `apps/gradlew.bat :shared:test` and build the phone plus demo-watch
      variants.
- [ ] Generate and verify `RELEASE_MANIFEST.json` with the commands in the root
      README.
- [ ] Inspect `git status` and `git diff --cached` before the first push.
- [ ] Enable GitHub secret scanning/push protection where available.

The source may be published while full model inference remains experimental,
provided the README's verification-status section remains accurate.

On Windows, clone to a short path or pass pytest a short `--basetemp` path;
checkpoint tests can exceed the legacy path-length limit in deeply nested
directories.
