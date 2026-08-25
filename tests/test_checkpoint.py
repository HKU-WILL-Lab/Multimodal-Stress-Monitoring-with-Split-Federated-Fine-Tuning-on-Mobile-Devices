from __future__ import annotations

import json

import torch

from sfl_clean.checkpoint import save_lora_checkpoint


def test_checkpoint_uses_bounded_unique_tensor_filenames(tmp_path) -> None:
    state = {
        "model.layers.0.self_attn.q_proj.lora_A." + "x" * 300: torch.ones(2),
        "model.layers.0.self_attn.q_proj.lora_B." + "y" * 300: torch.zeros(2),
    }

    output = save_lora_checkpoint(tmp_path / "checkpoint", state, {"kind": "test"})
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert [item["name"] for item in manifest["tensors"]] == sorted(state)
    assert [item["file"] for item in manifest["tensors"]] == [
        "tensor-0000.f32le",
        "tensor-0001.f32le",
    ]
    assert all((output / item["file"]).stat().st_size == 8 for item in manifest["tensors"])
