#!/usr/bin/env python3
"""Fail closed when a source tree is unsafe or incomplete for publication."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import release_manifest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / release_manifest.DEFAULT_MANIFEST_NAME
REQUIRED_FILES = (
    ".gitignore",
    "CONTRIBUTING.md",
    "PROVENANCE.md",
    "README.md",
    "RELEASE_CHECKLIST.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
)
REQUIRED_DIRECTORIES = ("apps", "configs", "docs", "sfl_runtime", "scripts")
REQUIRED_IMPLEMENTATION_FILES = (
    "sfl_runtime/android/src/executorch_encoder.cpp",
    "sfl_runtime/android/src/wellbeing_sfl_jni.cpp",
    "sfl_runtime/android/src/local_inference.cpp",
    "sfl_runtime/android/src/local_inference_jni.cpp",
    "apps/phone/src/main/kotlin/org/mobihoc/wellbeing/phone/inference/InferenceEngine.kt",
    "apps/phone/src/main/kotlin/org/mobihoc/wellbeing/phone/training/TrainingEngine.kt",
    "apps/wear/src/main/kotlin/org/mobihoc/wellbeing/wear/capture/SensorCaptureService.kt",
    "apps/wear/src/main/kotlin/org/mobihoc/wellbeing/wear/transport/WearDataLayerService.kt",
)
MAX_SOURCE_FILE_BYTES = 50 * 1024 * 1024
TEXT_SUFFIXES = {
    "", ".bat", ".cff", ".cmake", ".cpp", ".gradle", ".h", ".json",
    ".kts", ".kt", ".md", ".properties", ".proto", ".ps1", ".py",
    ".sh", ".toml", ".txt", ".xml",
}
SECRET_PATTERNS = (
    re.compile(rb"sk-(?:or-)?[A-Za-z0-9_-]{16,}"),
    re.compile(rb"(?i)(?:api[_-]?key|client[_-]?secret|password)\s*[:=]\s*['\"]?[^\s'\"]{8,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
PERSONAL_PATH_PATTERNS = (
    re.compile(rb"(?i)[A-Z]:\\Users\\[^\\\s]+"),
    re.compile(rb"(?i)[A-Z]:/Users/[^/\s]+"),
    re.compile(rb"/home/[^/\s]+"),
)
FORBIDDEN_CODE_PATTERNS = (
    re.compile(rb"(?i)\b(?:from|import)\s+flwr\b"),
    re.compile(rb"(?i)edgeflowertune"),
)
FORBIDDEN_UI_PATHS = (
    re.compile(r"(?i)(?:^|/)MainActivity\.kt$"),
    re.compile(r"(?i)(?:^|/)ui(?:/|$)"),
    re.compile(r"(?i)(?:^|/)res/(?:layout|drawable|mipmap)(?:/|$)"),
)


def main() -> int:
    errors: list[str] = []
    for name in REQUIRED_FILES:
        if not (ROOT / name).is_file():
            errors.append(f"missing required file: {name}")
    for name in REQUIRED_DIRECTORIES:
        if not (ROOT / name).is_dir():
            errors.append(f"missing required directory: {name}")
    for name in REQUIRED_IMPLEMENTATION_FILES:
        if not (ROOT / name).is_file():
            errors.append(f"missing training/inference/device implementation: {name}")

    files = list(release_manifest.iter_release_files(ROOT, MANIFEST))
    for relative, path in files:
        normalized = relative.as_posix()
        if any(pattern.search(normalized) for pattern in FORBIDDEN_UI_PATHS):
            errors.append(f"unfinished UI file included in source release: {relative}")
        if path.stat().st_size > MAX_SOURCE_FILE_BYTES:
            errors.append(f"source file exceeds 50 MiB: {relative}")
        if path.suffix.casefold() not in TEXT_SUFFIXES and relative.as_posix() != "apps/gradle/wrapper/gradle-wrapper.jar":
            errors.append(f"unexpected binary or unknown source type: {relative}")
            continue
        if relative.as_posix() == "apps/gradle/wrapper/gradle-wrapper.jar":
            continue
        if relative.as_posix() == "scripts/release_check.py":
            continue
        try:
            content = path.read_bytes()
        except OSError as error:
            errors.append(f"cannot read {relative}: {error}")
            continue
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            errors.append(f"possible credential in: {relative}")
        if any(pattern.search(content) for pattern in PERSONAL_PATH_PATTERNS):
            errors.append(f"personal absolute path in: {relative}")
        if path.suffix.casefold() in {".py", ".cpp", ".h", ".kt", ".kts", ".proto"}:
            if any(pattern.search(content) for pattern in FORBIDDEN_CODE_PATTERNS):
                errors.append(f"forbidden EFT/Flower code dependency in: {relative}")
            if b"androidx.compose" in content:
                errors.append(f"unfinished Compose UI dependency in: {relative}")

    if not MANIFEST.is_file():
        errors.append("RELEASE_MANIFEST.json is missing; generate it first")
    else:
        valid, message = release_manifest.verify_manifest(ROOT, MANIFEST)
        if not valid:
            errors.append(message)

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"release check passed: {len(files)} source files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
