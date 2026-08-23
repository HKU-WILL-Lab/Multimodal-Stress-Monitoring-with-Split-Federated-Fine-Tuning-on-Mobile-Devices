"""Generate and import Python bindings without committing generated files."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
from types import ModuleType


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def generate_bindings(output: Path | None = None) -> Path:
    try:
        from grpc_tools import protoc
    except ImportError as error:
        raise RuntimeError(
            "grpcio-tools is required to generate bindings; install the 'dev' extra"
        ) from error
    root = repository_root()
    proto_directory = root / "proto"
    proto_file = proto_directory / "sfl_clean.proto"
    output = (output or root / ".generated" / "python").resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{proto_directory}",
            f"--python_out={output}",
            f"--grpc_python_out={output}",
            str(proto_file),
        ]
    )
    if result != 0:
        raise RuntimeError(f"protoc failed with exit code {result}")
    return output


def load_bindings(*, generate_if_missing: bool = True) -> tuple[ModuleType, ModuleType]:
    output = repository_root() / ".generated" / "python"
    if generate_if_missing and not (output / "sfl_clean_pb2.py").is_file():
        generate_bindings(output)
    path = str(output.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        pb2 = importlib.import_module("sfl_clean_pb2")
        pb2_grpc = importlib.import_module("sfl_clean_pb2_grpc")
    except ImportError as error:
        raise RuntimeError(
            "protobuf bindings are missing; run `python -m sfl_clean.proto_runtime`"
        ) from error
    return pb2, pb2_grpc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    print(generate_bindings(arguments.output))


if __name__ == "__main__":
    main()

