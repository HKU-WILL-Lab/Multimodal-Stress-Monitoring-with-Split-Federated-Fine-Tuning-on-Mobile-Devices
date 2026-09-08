"""Export the frozen 30-second NormWear token encoder for ExecuTorch.

The source checkout is supplied explicitly and remains a build-time input; the
resulting program returns [batch, 162, 768] instead of mean-pooling the encoder.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch


def _supply_source_checkout_schemas() -> None:
    """Let an uninstalled ExecuTorch checkout find its packaged schemas."""
    import executorch
    import executorch.exir._serialize._flatbuffer as flatbuffer

    package_root = Path(next(iter(executorch.__path__)))
    packaged = package_root / "exir" / "_serialize" / "program.fbs"
    if packaged.is_file():
        return
    schema_root = package_root / "schema"
    if not (schema_root / "program.fbs").is_file():
        raise RuntimeError("ExecuTorch program schemas are unavailable")
    original = flatbuffer.importlib.resources.read_binary

    def read_binary(package: str, name: str) -> bytes:
        candidate = schema_root / name
        if name in {"program.fbs", "scalar_type.fbs"} and candidate.is_file():
            return candidate.read_bytes()
        return original(package, name)

    flatbuffer.importlib.resources.read_binary = read_binary


def export_encoder(source: Path, checkpoint: Path, output: Path) -> tuple[int, ...]:
    module_path = source.resolve() / "opentslm" / "model" / "encoder" / "normwear_inference.py"
    if not module_path.is_file():
        raise RuntimeError("--opentslm-source must contain the opentslm package")
    spec = importlib.util.spec_from_file_location("_sfl_normwear_inference", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the supplied NormWear implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    NormWearInference = module.NormWearInference

    model = NormWearInference(checkpoint).eval()
    example = torch.zeros((1, 6, 240), dtype=torch.float32)
    with torch.inference_mode():
        eager = model(example)
    if tuple(eager.shape) != (1, 162, 768):
        raise RuntimeError(f"unexpected 30-second NormWear output: {tuple(eager.shape)}")

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    _supply_source_checkout_schemas()

    exported = torch.export.export(model, (example,), strict=True)
    edge = to_edge_transform_and_lower(
        exported,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        partitioner=[XnnpackPartitioner()],
    )
    program = edge.to_executorch()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(program.buffer)
    return tuple(eager.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opentslm-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flatc", type=Path)
    args = parser.parse_args()
    if args.flatc is not None:
        flatc = args.flatc.resolve(strict=True)
        os.environ["FLATC_EXECUTABLE"] = str(flatc)
    shape = export_encoder(args.opentslm_source, args.checkpoint, args.output)
    print(f"exported {args.output} with output shape {shape}")


if __name__ == "__main__":
    main()
