"""LoRA-only checkpoints with an inspectable JSON manifest."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Mapping
from uuid import uuid4

import numpy as np
import torch

from .aggregation import validate_lora_state


def save_lora_checkpoint(
    directory: Path,
    state: Mapping[str, torch.Tensor],
    metadata: Mapping[str, object],
) -> Path:
    """Atomically create a directory containing only raw LoRA tensors and metadata."""

    validate_lora_state(state)
    directory = directory.resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.parent / f".{directory.name}.tmp-{uuid4().hex}"
    staging.mkdir()
    tensors: list[dict[str, object]] = []
    try:
        for index, name in enumerate(sorted(state)):
            value = state[name].detach().to(device="cpu", dtype=torch.float32).contiguous()
            # Tensor identity lives in the inspectable manifest. A bounded,
            # deterministic ordinal avoids both hash collisions and Windows
            # path-limit failures in deeply nested research workspaces.
            filename = f"tensor-{index:04d}.f32le"
            array = np.asarray(value).astype(np.dtype("<f4"), copy=False)
            (staging / filename).write_bytes(array.tobytes(order="C"))
            tensors.append(
                {
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": "float32",
                    "file": filename,
                }
            )
        manifest = {"format_version": 1, "metadata": dict(metadata), "tensors": tensors}
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        if directory.exists():
            raise FileExistsError(f"checkpoint already exists: {directory}")
        os.replace(staging, directory)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return directory
