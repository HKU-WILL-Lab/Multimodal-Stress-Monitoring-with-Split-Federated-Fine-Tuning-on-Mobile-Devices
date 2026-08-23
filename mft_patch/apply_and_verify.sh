#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SOURCE_ROOT DESTINATION" >&2
  exit 2
fi

source_root=$(cd "$1" && pwd)
destination=$2
pinned=b62d3b12a597e05489e6e8ef025527c613c94837
patch_dir=$(cd "$(dirname "$0")" && pwd)

actual=$(git -C "$source_root" rev-parse HEAD)
[[ "$actual" == "$pinned" ]] || {
  echo "expected MobileFineTuner $pinned, found $actual" >&2
  exit 1
}
[[ ! -e "$destination" ]] || {
  echo "destination already exists; refusing to overwrite: $destination" >&2
  exit 1
}

mkdir -p "$destination"
cp -R "$source_root/operator" "$destination/operator"
cp "$source_root/LICENSE" "$destination/LICENSE"
git -C "$destination" init --quiet
git -C "$destination" apply --check -p2 "$patch_dir/mobilefinetuner-hidden-span.patch"
git -C "$destination" apply -p2 "$patch_dir/mobilefinetuner-hidden-span.patch"

for file in \
  operator/finetune_ops/graph/gemma_model.h \
  operator/finetune_ops/graph/gemma_model.cpp \
  operator/finetune_ops/graph/gemma_lora_injector.cpp; do
  grep -Fq 'MODIFIED by the SFL Clean project, 2026.' "$destination/$file"
done
echo "patched pristine MobileFineTuner $pinned into $destination"
