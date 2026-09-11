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


@pytest.mark.parametrize('cut', [2, 3, 4])
def test_export_includes_exact_prefix_from_sharded_source(tmp_path, monkeypatch, cut):
    safetensors = pytest.importorskip('safetensors.torch')
    expected = {'model.embed_tokens.weight': (8, 4),
                'model.layers.0.input_layernorm.weight': (4,)}
    monkeypatch.setattr(export_llama_embedding, 'EXPECTED_TENSORS', expected)
    source = tmp_path / 'source'
    source.mkdir()
    tensors = {'model.embed_tokens.weight': torch.zeros(8, 4)}
    tensors.update({f'model.layers.{i}.input_layernorm.weight': torch.full((4,), float(i+1))
                    for i in range(4)})
    weight_map = {}
    for index, (name, tensor) in enumerate(tensors.items()):
        shard = f'shard-{index}.safetensors'
        safetensors.save_file({name: tensor}, source / shard)
        weight_map[name] = shard
    (source / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weight_map}))
    result = export_llama_embedding.export_embedding(source, tmp_path / 'out', cut)
    loaded = safetensors.load_file(result)
    assert len(loaded) == cut+1
    for i in range(cut):
        assert torch.equal(loaded[f'model.layers.{i}.input_layernorm.weight'], torch.full((4,), float(i+1)))
    assert f'model.layers.{cut}.input_layernorm.weight' not in loaded
    manifest = json.loads((result.parent / 'embedding_manifest.json').read_text())
    assert manifest['cut_layer'] == cut


@pytest.mark.parametrize('cut', [0, 5, -1, True])
def test_export_rejects_unsupported_cut(cut):
    with pytest.raises(ValueError, match='cut_layer'):
        export_llama_embedding.expected_tensors(cut)
