"""Validated JSON configuration shared by host services."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .errors import InvalidRequest


@dataclass(frozen=True)
class RpcSettings:
    bind: str
    max_message_bytes: int
    deadline_seconds: float
    worker_threads: int


@dataclass(frozen=True)
class ModelSettings:
    source: str
    local_files_only: bool
    device: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float


@dataclass(frozen=True)
class TrainingSettings:
    total_rounds: int
    submission_quorum: int
    cut_layer: int
    batch_size: int
    sequence_length: int
    local_steps: int
    gradient_clip_norm: float | None


@dataclass(frozen=True)
class AppConfig:
    protocol_version: int
    run_id: str
    suffix_rpc: RpcSettings
    coordinator_rpc: RpcSettings
    model: ModelSettings
    training: TrainingSettings
    metrics_path: Path
    checkpoint_root: Path


def load_config(path: Path) -> AppConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidRequest(f"cannot load configuration {path}: {error}") from error
    if not isinstance(raw, dict):
        raise InvalidRequest("configuration root must be a JSON object")
    _keys(
        raw,
        {
            "protocol_version",
            "run_id",
            "suffix_rpc",
            "coordinator_rpc",
            "model",
            "training",
            "metrics_path",
            "checkpoint_root",
        },
        "configuration",
    )
    config = AppConfig(
        protocol_version=_integer(raw, "protocol_version"),
        run_id=_string(raw, "run_id"),
        suffix_rpc=_rpc(_mapping(raw, "suffix_rpc"), "suffix_rpc"),
        coordinator_rpc=_rpc(_mapping(raw, "coordinator_rpc"), "coordinator_rpc"),
        model=_model(_mapping(raw, "model")),
        training=_training(_mapping(raw, "training")),
        metrics_path=Path(_string(raw, "metrics_path")),
        checkpoint_root=Path(_string(raw, "checkpoint_root")),
    )
    _validate(config)
    return config


def _rpc(raw: Mapping[str, Any], label: str) -> RpcSettings:
    _keys(raw, {"bind", "max_message_bytes", "deadline_seconds", "worker_threads"}, label)
    return RpcSettings(
        bind=_string(raw, "bind"),
        max_message_bytes=_integer(raw, "max_message_bytes"),
        deadline_seconds=_number(raw, "deadline_seconds"),
        worker_threads=_integer(raw, "worker_threads"),
    )


def _model(raw: Mapping[str, Any]) -> ModelSettings:
    _keys(
        raw,
        {
            "source",
            "local_files_only",
            "device",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "learning_rate",
        },
        "model",
    )
    local = raw.get("local_files_only")
    if not isinstance(local, bool):
        raise InvalidRequest("model.local_files_only must be boolean")
    return ModelSettings(
        source=_string(raw, "source"),
        local_files_only=local,
        device=_string(raw, "device"),
        lora_rank=_integer(raw, "lora_rank"),
        lora_alpha=_integer(raw, "lora_alpha"),
        lora_dropout=_number(raw, "lora_dropout"),
        learning_rate=_number(raw, "learning_rate"),
    )


def _training(raw: Mapping[str, Any]) -> TrainingSettings:
    _keys(
        raw,
        {
            "total_rounds",
            "submission_quorum",
            "cut_layer",
            "batch_size",
            "sequence_length",
            "local_steps",
            "gradient_clip_norm",
        },
        "training",
    )
    clip = raw.get("gradient_clip_norm")
    if clip is not None and (isinstance(clip, bool) or not isinstance(clip, (int, float))):
        raise InvalidRequest("training.gradient_clip_norm must be numeric or null")
    return TrainingSettings(
        total_rounds=_integer(raw, "total_rounds"),
        submission_quorum=_integer(raw, "submission_quorum"),
        cut_layer=_integer(raw, "cut_layer"),
        batch_size=_integer(raw, "batch_size"),
        sequence_length=_integer(raw, "sequence_length"),
        local_steps=_integer(raw, "local_steps"),
        gradient_clip_norm=float(clip) if clip is not None else None,
    )


def _validate(config: AppConfig) -> None:
    if config.protocol_version != 1:
        raise InvalidRequest("only protocol_version 1 is supported")
    if not config.run_id or len(config.run_id) > 128:
        raise InvalidRequest("run_id must contain 1 to 128 characters")
    for label, rpc in (("suffix_rpc", config.suffix_rpc), ("coordinator_rpc", config.coordinator_rpc)):
        if not rpc.bind or rpc.max_message_bytes <= 0 or rpc.deadline_seconds <= 0 or rpc.worker_threads <= 0:
            raise InvalidRequest(f"{label} values must be nonempty and positive")
    model = config.model
    if model.device != "cpu" and model.device != "cuda" and not model.device.startswith("cuda:"):
        raise InvalidRequest("model.device must be cpu, cuda, or cuda:<index>")
    if model.lora_rank <= 0 or model.lora_alpha <= 0 or model.learning_rate <= 0:
        raise InvalidRequest("LoRA sizes and learning rate must be positive")
    if not 0.0 <= model.lora_dropout < 1.0:
        raise InvalidRequest("LoRA dropout must be in [0, 1)")
    training = config.training
    positive = (
        training.total_rounds,
        training.submission_quorum,
        training.cut_layer,
        training.batch_size,
        training.sequence_length,
        training.local_steps,
    )
    if any(value <= 0 for value in positive):
        raise InvalidRequest("all integer training settings must be positive")
    if training.cut_layer >= 18:
        raise InvalidRequest("cut_layer must be below the 18-layer model depth")
    if training.sequence_length < 2:
        raise InvalidRequest("sequence_length must be at least two")
    if training.gradient_clip_norm is not None and training.gradient_clip_norm <= 0:
        raise InvalidRequest("gradient_clip_norm must be positive when enabled")


def _keys(raw: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(raw)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise InvalidRequest(f"{label} keys differ; missing={missing}, extra={extra}")


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise InvalidRequest(f"{key} must be a JSON object")
    return value


def _string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise InvalidRequest(f"{key} must be a nonempty string")
    return value


def _integer(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidRequest(f"{key} must be an integer")
    return value


def _number(raw: Mapping[str, Any], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidRequest(f"{key} must be numeric")
    return float(value)

