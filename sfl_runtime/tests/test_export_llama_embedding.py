from __future__ import annotations

import json

import pytest
import torch

from sfl_clean import export_llama_embedding


def test_export_embedding_writes_minimal_prefix_asset(tmp_path, monkeypatch) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    expected = {
        "model.embed_tokens.weight": (8, 4),
        "model.layers.0.input_layernorm.weight": (4,),
        "model.layers.0.self_attn.q_proj.weight": (4, 4),
    }
    monkeypatch.setattr(export_llama_embedding, "EXPECTED_TENSORS", expected)
    source = tmp_path / "source"
    destination = tmp_path / "mobile"
    source.mkdir()
    values = {name: torch.zeros(shape, dtype=torch.float16) for name, shape in expected.items()}
    safetensors.save_file(values, source / "model.safetensors")

    exported = export_llama_embedding.export_embedding(source, destination)

    loaded = safetensors.load_file(exported)
    assert set(loaded) == set(expected)
    assert tuple(loaded["model.embed_tokens.weight"].shape) == (8, 4)
    manifest = json.loads((destination / "embedding_manifest.json").read_text())
    assert manifest["architecture"] == "Llama-3.2-1B"
    assert manifest["cut_layer"] == 1


def test_export_rejects_wrong_architecture_shape(tmp_path) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    source = tmp_path / "source"
    source.mkdir()
    safetensors.save_file(
        {"model.embed_tokens.weight": torch.zeros((8, 4))},
        source / "model.safetensors",
    )
    with pytest.raises(ValueError, match="checkpoint is missing required tensors"):
        export_llama_embedding.export_embedding(source, tmp_path / "mobile")
