#!/usr/bin/env python3
"""Create or verify a deterministic manifest of distributable source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


MANIFEST_FORMAT = "sfl-clean-release-manifest-v1"
DEFAULT_MANIFEST_NAME = "RELEASE_MANIFEST.json"

# The functional specification is a process input rather than a distributable
# artifact. Everything else below is generated, external, local, or sensitive.
EXCLUDED_EXACT_PATHS = frozenset(
    {
        "cleanroom_spec.md",
        DEFAULT_MANIFEST_NAME.casefold(),
        (DEFAULT_MANIFEST_NAME + ".tmp").casefold(),
    }
)

EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".cache",
        ".cxx",
        ".deps",
        ".generated",
        ".eggs",
        ".externalnativebuild",
        ".git",
        ".gradle",
        ".hypothesis",
        ".idea",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".python_deps",
        ".ruff_cache",
        ".tox",
        ".test-deps",
        ".venv",
        ".vscode",
        "__pycache__",
        "artifacts",
        "assets",
        "build",
        "checkpoints",
        "cmakefiles",
        "credentials",
        "data",
        "datasets",
        "deps",
        "dist",
        "env",
        "generated",
        "htmlcov",
        "logs",
        "metrics",
        "model",
        "models",
        "obj",
        "out",
        "outputs",
        "pip-wheel-metadata",
        "runs",
        "runtime",
        "secrets",
        "tokenizer",
        "tokenizers",
        "third_party",
        "vendor",
        "venv",
        "_deps",
    }
)

EXCLUDED_SUFFIXES = (
    ".a",
    ".aab",
    ".aar",
    ".apk",
    ".arrow",
    ".bin",
    ".ckpt",
    ".desc",
    ".dll",
    ".dylib",
    ".gguf",
    ".jks",
    ".jsonl",
    ".key",
    ".keystore",
    ".lib",
    ".log",
    ".model",
    ".npy",
    ".npz",
    ".o",
    ".obj",
    ".onnx",
    ".p12",
    ".parquet",
    ".pb.cc",
    ".pb.h",
    ".pem",
    ".pfx",
    ".pte",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
    ".safetensors",
    ".serial",
    ".sflsensor",
    ".so",
    ".tar",
    ".tar.gz",
    ".tflite",
    ".tgz",
    ".zip",
)

EXCLUDED_FILE_NAMES = frozenset(
    {
        ".coverage",
        ".ds_store",
        ".env",
        ".python",
        "cmakecache.txt",
        "cmake_install.cmake",
        "compile_commands.json",
        "coverage.xml",
        "local.properties",
        "merges.txt",
        "special_tokens_map.json",
        "thumbs.db",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    }
)


def exclusion_reason(relative_path: PurePosixPath) -> str | None:
    """Return the release-policy reason for excluding a relative path."""

    normalized = relative_path.as_posix().casefold()
    parts = tuple(part.casefold() for part in relative_path.parts)
    name = parts[-1]

    if normalized in EXCLUDED_EXACT_PATHS:
        return "clean-room process input or generated manifest"
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in parts[:-1]):
        return "generated, external, runtime, or local directory"
    if name in EXCLUDED_DIRECTORY_NAMES:
        return "generated, external, runtime, or local directory"
    if name in EXCLUDED_FILE_NAMES or name.startswith(".coverage."):
        return "generated, external, runtime, or local file"
    if name.startswith(".env.") or name.startswith("cmake-build-"):
        return "local configuration or build output"
    if name.endswith(".egg-info"):
        return "Python build metadata"
    if name.startswith("device") and name.endswith(
        (".ini", ".json", ".properties", ".toml", ".yaml", ".yml")
    ):
        return "device-specific configuration"
    if name.startswith("adb") and name.endswith((".json", ".toml", ".yaml", ".yml")):
        return "device-specific configuration"
    if name.endswith(("_pb2.py", "_pb2.pyi", "_pb2_grpc.py")):
        return "generated protocol binding"
    if name.endswith((".grpc.pb.cc", ".grpc.pb.h")):
        return "generated protocol binding"
    if name.endswith(EXCLUDED_SUFFIXES):
        return "binary, model, dataset, secret, runtime, or archive artifact"
    if name.endswith((".egg", ".whl")):
        return "Python build artifact"
    if name.startswith(".ninja") or name.endswith(".ninja"):
        return "build output"
    return None


def _relative_posix(root: Path, path: Path) -> PurePosixPath:
    return PurePosixPath(path.relative_to(root).as_posix())


def iter_release_files(root: Path, manifest_path: Path) -> Iterator[tuple[PurePosixPath, Path]]:
    """Yield release files in stable path order without following symlinks."""

    selected: list[tuple[PurePosixPath, Path]] = []
    manifest_path = manifest_path.resolve()

    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        kept_directories: list[str] = []
        for directory_name in sorted(directory_names, key=str.casefold):
            directory_path = current_path / directory_name
            relative = _relative_posix(root, directory_path)
            if exclusion_reason(relative) is not None:
                continue
            if directory_path.is_symlink():
                raise ValueError(f"release tree contains a directory symlink: {relative}")
            kept_directories.append(directory_name)
        directory_names[:] = kept_directories

        for file_name in sorted(file_names, key=str.casefold):
            file_path = current_path / file_name
            relative = _relative_posix(root, file_path)
            if exclusion_reason(relative) is not None:
                continue
            if file_path.is_symlink():
                raise ValueError(f"release tree contains a file symlink: {relative}")
            if file_path.resolve() == manifest_path:
                continue
            if not file_path.is_file():
                raise ValueError(f"release entry is not a regular file: {relative}")
            selected.append((relative, file_path))

    yield from sorted(selected, key=lambda item: item[0].as_posix().casefold())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    manifest_path = manifest_path.resolve()
    try:
        manifest_path.relative_to(root)
    except ValueError as error:
        raise ValueError("manifest output must be inside the source root") from error

    files = [
        {
            "path": relative.as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for relative, path in iter_release_files(root, manifest_path)
    ]
    return {
        "format": MANIFEST_FORMAT,
        "hash_algorithm": "sha256",
        "files": files,
    }


def serialized_manifest(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def write_manifest(root: Path, manifest_path: Path) -> None:
    content = serialized_manifest(build_manifest(root, manifest_path))
    temporary_path = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_path.write_text(content, encoding="utf-8", newline="\n")
    temporary_path.replace(manifest_path)


def verify_manifest(root: Path, manifest_path: Path) -> tuple[bool, str]:
    if not manifest_path.is_file():
        return False, f"manifest does not exist: {manifest_path}"
    try:
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return False, f"cannot read manifest: {error}"

    expected = build_manifest(root, manifest_path)
    if recorded == expected:
        return True, "release manifest matches the source tree"

    recorded_files = {item.get("path"): item for item in recorded.get("files", [])}
    expected_files = {item["path"]: item for item in expected["files"]}
    added = sorted(expected_files.keys() - recorded_files.keys())
    removed = sorted(recorded_files.keys() - expected_files.keys())
    changed = sorted(
        path
        for path in expected_files.keys() & recorded_files.keys()
        if expected_files[path] != recorded_files[path]
    )
    details = []
    if added:
        details.append("unrecorded: " + ", ".join(added))
    if removed:
        details.append("missing: " + ", ".join(removed))
    if changed:
        details.append("changed: " + ", ".join(changed))
    if not details:
        details.append("manifest metadata is invalid")
    return False, "; ".join(details)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="source tree root (defaults to the parent of scripts/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_MANIFEST_NAME),
        help="manifest path, absolute or relative to --root",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="write a fresh manifest")
    action.add_argument("--check", action="store_true", help="verify the existing manifest")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.root.resolve(strict=True)
    manifest_path = args.output if args.output.is_absolute() else root / args.output
    manifest_path = manifest_path.resolve()

    try:
        manifest_path.relative_to(root)
    except ValueError:
        print("error: manifest output must be inside the source root", file=sys.stderr)
        return 2

    try:
        if args.write:
            write_manifest(root, manifest_path)
            print(f"wrote {manifest_path}")
            return 0
        valid, message = verify_manifest(root, manifest_path)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(message, file=sys.stdout if valid else sys.stderr)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
