"""Append-only JSON Lines training metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from threading import Lock

import torch

from .errors import InvalidRequest


@dataclass(frozen=True)
class MetricRecord:
    schema_version: int
    run_id: str
    client_id: str
    global_round: int
    local_step: int
    batch_size: int
    sequence_length: int
    loss: float
    token_accuracy: float
    split_rpc_bytes: int
    duration_seconds: float
    host_gpu_memory_available_bytes: int | None

    def validate(self) -> None:
        if self.schema_version != 1:
            raise InvalidRequest("metric schema_version must be 1")
        if not self.run_id or not self.client_id:
            raise InvalidRequest("metric run_id and client_id cannot be empty")
        if self.global_round <= 0 or self.local_step < 0:
            raise InvalidRequest("metric round must be positive and local step nonnegative")
        if self.batch_size <= 0 or self.sequence_length <= 0:
            raise InvalidRequest("metric batch and sequence dimensions must be positive")
        if not math.isfinite(self.loss) or not math.isfinite(self.token_accuracy):
            raise InvalidRequest("metric loss and accuracy must be finite")
        if not 0.0 <= self.token_accuracy <= 1.0:
            raise InvalidRequest("metric token accuracy must be in [0, 1]")
        if self.split_rpc_bytes < 0 or self.duration_seconds < 0:
            raise InvalidRequest("metric byte count and duration cannot be negative")
        if (
            self.host_gpu_memory_available_bytes is not None
            and self.host_gpu_memory_available_bytes < 0
        ):
            raise InvalidRequest("metric available GPU memory cannot be negative")


class JsonlMetricWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def append(self, record: MetricRecord) -> None:
        record.validate()
        payload = json.dumps(
            asdict(record), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        with self._lock, self._path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(payload + "\n")
            stream.flush()


def available_gpu_memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    free_bytes, _ = torch.cuda.mem_get_info(device)
    return int(free_bytes)
