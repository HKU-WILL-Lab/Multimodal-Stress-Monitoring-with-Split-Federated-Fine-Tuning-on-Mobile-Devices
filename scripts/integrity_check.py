#!/usr/bin/env python3
"""Validate release policy, provenance markers, and manifest integrity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import release_manifest


PINNED_MFT_COMMIT = "b62d3b12a597e05489e6e8ef025527c613c94837"
PAPER_IDENTIFIERS = ("2407.00952", "2004.12088")
BASE_REQUIRED_FILES = (
    ".gitignore",
    "CITATION.cff",
    "LICENSE",
    "PROVENANCE.md",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "requirements-release.txt",
    "scripts/integrity_check.py",
    "scripts/release_manifest.py",
)
RELEASE_REQUIRED_DIRECTORIES = (
    "android",
    "configs",
    "mft_patch",
    "proto",
    "python/sfl_clean",
    "tests",
)


def forbidden_source_terms() -> tuple[str, ...]:
    # Construct the policy terms so the integrity checker does not itself make
    # a literal-source scan fail. The process specification is not released.
    return (
        "flo" + "wer",
        "fl" + "wr",
        "edge" + "flo" + "wer" + "tune",
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="source tree root (defaults to the parent of scripts/)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(release_manifest.DEFAULT_MANIFEST_NAME),
        help="manifest path, absolute or relative to --root",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="also enforce complete-tree and no-placeholder publication gates",
    )
    return parser.parse_args(argv)


def check_required_paths(root: Path, release_mode: bool) -> list[str]:
    errors = [
        f"required release file is missing: {relative}"
        for relative in BASE_REQUIRED_FILES
        if not (root / relative).is_file()
    ]
    if release_mode:
        errors.extend(
            f"required implementation directory is missing: {relative}"
            for relative in RELEASE_REQUIRED_DIRECTORIES
            if not (root / relative).is_dir()
        )
    return errors


def check_document_markers(root: Path) -> list[str]:
    errors: list[str] = []
    paths = (
        root / "README.md",
        root / "THIRD_PARTY_NOTICES.md",
        root / "PROVENANCE.md",
        root / "CITATION.cff",
    )
    readable: dict[Path, str] = {}
    for path in paths:
        try:
            readable[path] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            errors.append(f"cannot read {path.relative_to(root)} as UTF-8: {error}")

    for path, content in readable.items():
        if PINNED_MFT_COMMIT not in content:
            errors.append(f"pinned MobileFineTuner commit is missing from {path.relative_to(root)}")

    for path in (
        root / "README.md",
        root / "THIRD_PARTY_NOTICES.md",
        root / "PROVENANCE.md",
        root / "CITATION.cff",
    ):
        content = readable.get(path)
        if content is None:
            continue
        for identifier in PAPER_IDENTIFIERS:
            if identifier not in content:
                errors.append(f"paper identifier {identifier} is missing from {path.relative_to(root)}")

    license_path = root / "LICENSE"
    try:
        license_text = license_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        errors.append(f"cannot read LICENSE as UTF-8: {error}")
    else:
        for marker in ("Apache License", "Version 2.0, January 2004", "END OF TERMS AND CONDITIONS"):
            if marker not in license_text:
                errors.append(f"LICENSE is missing expected marker: {marker}")
    return errors


def check_source_terms(root: Path, manifest_path: Path) -> list[str]:
    errors: list[str] = []
    encoded_terms = tuple(term.encode("ascii") for term in forbidden_source_terms())
    for relative, path in release_manifest.iter_release_files(root, manifest_path):
        try:
            lowered = path.read_bytes().lower()
        except OSError as error:
            errors.append(f"cannot scan {relative}: {error}")
            continue
        for term in encoded_terms:
            if term in lowered:
                errors.append(f"forbidden source-boundary term occurs in {relative}")
    return errors


def check_placeholders(root: Path, manifest_path: Path) -> list[str]:
    errors: list[str] = []
    markers = (
        b"todo" + b"-replace",
        b"todo" + b" before release",
        b"example" + b".invalid",
    )
    for relative, path in release_manifest.iter_release_files(root, manifest_path):
        try:
            lowered = path.read_bytes().lower()
        except OSError as error:
            errors.append(f"cannot scan {relative}: {error}")
            continue
        if any(marker in lowered for marker in markers):
            errors.append(f"release-blocking placeholder occurs in {relative}")
    return errors


def check_manifest(root: Path, manifest_path: Path) -> list[str]:
    try:
        valid, message = release_manifest.verify_manifest(root, manifest_path)
    except (OSError, ValueError) as error:
        return [f"cannot verify release manifest: {error}"]
    return [] if valid else [message]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        root = args.root.resolve(strict=True)
    except OSError as error:
        print(f"error: cannot resolve source root: {error}", file=sys.stderr)
        return 2

    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    manifest_path = manifest_path.resolve()
    try:
        manifest_path.relative_to(root)
    except ValueError:
        print("error: manifest must be inside the source root", file=sys.stderr)
        return 2

    errors: list[str] = []
    try:
        errors.extend(check_required_paths(root, args.release))
        errors.extend(check_document_markers(root))
        errors.extend(check_source_terms(root, manifest_path))
        if args.release:
            errors.extend(check_placeholders(root, manifest_path))
        errors.extend(check_manifest(root, manifest_path))
    except ValueError as error:
        errors.append(str(error))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    mode = "release" if args.release else "development"
    print(f"integrity check passed ({mode} mode)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
