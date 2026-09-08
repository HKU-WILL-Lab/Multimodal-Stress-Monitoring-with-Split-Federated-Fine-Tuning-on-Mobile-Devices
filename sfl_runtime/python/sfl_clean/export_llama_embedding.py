"""Export the minimal Llama assets required by the cut-after-block-0 client."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


TENSOR_NAME = "model.embed_tokens.weight"
EXPECTED_SHAPE = (128256, 2048)
EXPECTED_TENSORS = {
    TENSOR_NAME: EXPECTED_SHAPE,
    "model.layers.0.input_layernorm.weight": (2048,),
    "model.layers.0.post_attention_layernorm.weight": (2048,),
    "model.layers.0.self_attn.q_proj.weight": (2048, 2048),
    "model.layers.0.self_attn.k_proj.weight": (512, 2048),
    "model.layers.0.self_attn.v_proj.weight": (512, 2048),
    "model.layers.0.self_attn.o_proj.weight": (2048, 2048),
    "model.layers.0.mlp.gate_proj.weight": (8192, 2048),
    "model.layers.0.mlp.up_proj.weight": (8192, 2048),
    "model.layers.0.mlp.down_proj.weight": (2048, 8192),
}


def locate_tensor_shards(model_dir: Path) -> dict[str, Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        result: dict[str, Path] = {}
        for name in EXPECTED_TENSORS:
            relative = weight_map.get(name)
            if not isinstance(relative, str) or not relative:
                raise ValueError(f"{name} is absent from {index.name}")
            shard = (model_dir / relative).resolve()
            if shard.parent != model_dir.resolve() or not shard.is_file():
                raise ValueError(f"checkpoint shard path is invalid for {name}")
            result[name] = shard
        return result

    candidates = sorted(model_dir.glob("*.safetensors"))
    if not candidates:
        raise ValueError("no safetensors checkpoint was found")
    from safetensors import safe_open

    located: dict[str, Path] = {}
    for candidate in candidates:
        with safe_open(candidate, framework="pt", device="cpu") as source:
            keys = set(source.keys())
            for name in EXPECTED_TENSORS:
                if name in keys:
                    if name in located:
                        raise ValueError(f"multiple shards contain {name}")
                    located[name] = candidate
    missing = sorted(set(EXPECTED_TENSORS) - set(located))
    if missing:
        raise ValueError(f"checkpoint is missing required tensors: {missing}")
    return located


def export_embedding(model_dir: Path, output_dir: Path) -> Path:
    from safetensors import safe_open
    from safetensors.torch import save_file

    model_dir = model_dir.resolve(strict=True)
    if not model_dir.is_dir():
        raise ValueError("model source must be a directory")
    shards = locate_tensor_shards(model_dir)
    tensors: dict[str, torch.Tensor] = {}
    for name, expected_shape in EXPECTED_TENSORS.items():
        with safe_open(shards[name], framework="pt", device="cpu") as source:
            tensor = source.get_tensor(name)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"unexpected {name} shape {tuple(tensor.shape)}; expected {expected_shape}"
            )
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"unsupported dtype for {name}: {tensor.dtype}")
        tensors[name] = tensor.contiguous()

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "model.safetensors"
    save_file(tensors, destination)
    manifest = {
        "architecture": "Llama-3.2-1B",
        "cut_layer": 1,
        "tensors": {name: list(shape) for name, shape in EXPECTED_TENSORS.items()},
        "source_asset_not_redistributable": True,
    }
    (output_dir / "embedding_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(export_embedding(args.model_dir, args.output_dir))


if __name__ == "__main__":
    main()
