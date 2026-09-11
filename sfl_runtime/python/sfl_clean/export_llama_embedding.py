"""Export the minimal Llama assets required by the configurable mobile prefix."""

from __future__ import annotations

import argparse
import json
import struct
import tempfile
import os
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


def expected_tensors(cut_layer: int):
    if isinstance(cut_layer, bool) or cut_layer not in range(1, 5):
        raise ValueError("cut_layer must be in 1..4")
    tensors = {TENSOR_NAME: EXPECTED_TENSORS[TENSOR_NAME]}
    for layer in range(cut_layer):
        tensors.update({name.replace("model.layers.0.", f"model.layers.{layer}."): shape
                        for name, shape in EXPECTED_TENSORS.items() if name != TENSOR_NAME})
    return tensors


def locate_tensor_shards(model_dir: Path, cut_layer: int = 1) -> dict[str, Path]:
    required = expected_tensors(cut_layer)
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        result: dict[str, Path] = {}
        for name in required:
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
            for name in required:
                if name in keys:
                    if name in located:
                        raise ValueError(f"multiple shards contain {name}")
                    located[name] = candidate
    missing = sorted(set(required) - set(located))
    if missing:
        raise ValueError(f"checkpoint is missing required tensors: {missing}")
    return located


def export_embedding(model_dir: Path, output_dir: Path, cut_layer: int = 1) -> Path:
    model_dir = model_dir.resolve(strict=True)
    if not model_dir.is_dir():
        raise ValueError("model source must be a directory")
    required = expected_tensors(cut_layer)
    shards = locate_tensor_shards(model_dir, cut_layer)
    headers = {}
    spans = []
    output_header = {}
    offset = 0
    for name, shape in required.items():
        shard = shards[name]
        if shard not in headers:
            with shard.open('rb') as source:
                size = struct.unpack('<Q', source.read(8))[0]
                if size > 100_000_000:
                    raise ValueError('Checkpoint header is too large')
                headers[shard] = (8+size, json.loads(source.read(size)))
        base, header = headers[shard]
        meta = header[name]
        if tuple(meta['shape']) != shape:
            raise ValueError(f"unexpected {name} shape {meta['shape']}; expected {shape}")
        element_size = {'F16': 2, 'BF16': 2, 'F32': 4}.get(meta['dtype'])
        if element_size is None:
            raise ValueError(f"unsupported dtype for {name}: {meta['dtype']}")
        begin, end = meta['data_offsets']
        count = 1
        for dimension in shape:
            count *= dimension
        length = count * element_size
        if begin < 0 or end-begin != length or base+end > shard.stat().st_size:
            raise ValueError(f"invalid tensor span for {name}")
        output_header[name] = dict(dtype=meta['dtype'], shape=list(shape), data_offsets=[offset,offset+length])
        spans.append((shard,base+begin,length))
        offset += length
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "model.safetensors"
    encoded = json.dumps(output_header, separators=(',', ':')).encode('utf-8')
    encoded += b' ' * ((-len(encoded)) % 8)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix='.partial', delete=False) as target:
            temporary = Path(target.name)
            target.write(struct.pack('<Q',len(encoded)))
            target.write(encoded)
            # Frozen tensors retain their exact bytes/dtype. Bounded copying avoids
            # holding the embedding plus several blocks and a serialization copy in RAM.
            for shard, begin, length in spans:
                with shard.open('rb') as source:
                    source.seek(begin)
                    while length:
                        chunk = source.read(min(length, 8*1024*1024))
                        if not chunk:
                            raise ValueError('Checkpoint ended during tensor export')
                        target.write(chunk)
                        length -= len(chunk)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    manifest = {
        "architecture": "Llama-3.2-1B",
        "cut_layer": cut_layer,
        "tensors": {name: list(shape) for name, shape in required.items()},
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
    parser.add_argument("--cut-layer", type=int, default=1, choices=range(1, 5))
    args = parser.parse_args()
    print(export_embedding(args.model_dir, args.output_dir, args.cut_layer))


if __name__ == "__main__":
    main()
