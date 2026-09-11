"""Thread-safe round coordinator with weighted prefix aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Mapping

import torch

from .aggregation import WeightedState, validate_lora_state, weighted_average
from .aggregation_status import AggregationStatus
from .checkpoint import save_lora_checkpoint
from .errors import DuplicateUpdate, InvalidRequest, NotBootstrapped, StateConflict


class RoundState(Enum):
    TRAIN = "train"
    WAIT = "wait"
    DONE = "done"


@dataclass(frozen=True)
class RoundSnapshot:
    state: RoundState
    global_round: int
    tensors: Mapping[str, torch.Tensor]
    submissions_received: int
    submission_quorum: int


def _copy_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().to(device="cpu", dtype=torch.float32).clone()
        for name, tensor in state.items()
    }


class RoundCoordinator:
    """A one-process, lock-protected coordinator using one-based round numbers."""

    def __init__(
        self,
        *,
        total_rounds: int,
        submission_quorum: int,
        checkpoint_root: Path | None = None,
    ) -> None:
        if total_rounds <= 0:
            raise InvalidRequest("total_rounds must be positive")
        if submission_quorum <= 0:
            raise InvalidRequest("submission_quorum must be positive")
        self._total_rounds = total_rounds
        self._quorum = submission_quorum
        self._checkpoint_root = checkpoint_root
        self._lock = RLock()
        self._global_round = 1
        self._completed_round = 0
        self._global_state: dict[str, torch.Tensor] | None = None
        self._submissions: dict[str, WeightedState] = {}
        self.status = AggregationStatus(submission_quorum)

    def bootstrap(self, client_id: str, tensors: Mapping[str, torch.Tensor]) -> RoundSnapshot:
        _validate_client_id(client_id)
        validate_lora_state(tensors)
        proposed = _copy_state(tensors)
        with self._lock:
            if self._global_state is None:
                self._global_state = proposed
                self.status.publish('WAITING', self._global_round, 0, self._quorum)
            else:
                if set(proposed) != set(self._global_state) or any(
                    proposed[name].shape != self._global_state[name].shape
                    or not torch.equal(proposed[name], self._global_state[name])
                    for name in proposed
                ):
                    raise DuplicateUpdate(
                        "coordinator is already bootstrapped with a different prefix state"
                    )
            self.status.client(client_id, state='Training', round=self._global_round)
            return self._snapshot(RoundState.TRAIN)

    def fetch(self, client_id: str, last_completed_round: int) -> RoundSnapshot:
        _validate_client_id(client_id)
        if last_completed_round < 0:
            raise InvalidRequest("last_completed_round cannot be negative")
        with self._lock:
            self._require_state()
            if self._completed_round >= self._total_rounds:
                return self._snapshot(RoundState.DONE)
            if last_completed_round > self._completed_round:
                raise StateConflict(
                    f"client reports completed round {last_completed_round}, but coordinator "
                    f"has completed only {self._completed_round}"
                )
            state = RoundState.WAIT if client_id in self._submissions else RoundState.TRAIN
            return self._snapshot(state)

    def submit(
        self,
        client_id: str,
        global_round: int,
        processed_sequences: int,
        tensors: Mapping[str, torch.Tensor],
    ) -> RoundSnapshot:
        _validate_client_id(client_id)
        if global_round <= 0:
            raise InvalidRequest("global_round must be positive")
        if processed_sequences <= 0:
            raise InvalidRequest("processed_sequences must be positive")
        validate_lora_state(tensors)
        update = WeightedState(_copy_state(tensors), processed_sequences)
        with self._lock:
            current = self._require_state()
            if self._completed_round >= self._total_rounds:
                raise StateConflict("training is already complete")
            if global_round != self._global_round:
                raise StateConflict(
                    f"update is for round {global_round}; current round is {self._global_round}"
                )
            if client_id in self._submissions:
                raise DuplicateUpdate(
                    f"client {client_id!r} already submitted round {global_round}"
                )
            if set(update.tensors) != set(current):
                raise InvalidRequest("submitted tensor names do not match global prefix state")
            for name, tensor in update.tensors.items():
                if tensor.shape != current[name].shape:
                    raise InvalidRequest(
                        f"submitted tensor {name!r} has shape {tuple(tensor.shape)}; "
                        f"expected {tuple(current[name].shape)}"
                    )
            self._submissions[client_id] = update
            self.status.client(client_id, state='Uploaded', round=global_round,
                               sequences=processed_sequences,
                               bytes=sum(t.numel() * t.element_size() for t in update.tensors.values()))
            if len(self._submissions) < self._quorum:
                self.status.publish('WAITING', self._global_round, len(self._submissions), self._quorum)
                return self._snapshot(RoundState.WAIT)

            finished_round = self._global_round
            self.status.publish('AGGREGATING', finished_round, len(self._submissions), self._quorum)
            try:
                aggregated = weighted_average(list(self._submissions.values()))
                self._save_checkpoint(finished_round, aggregated)
            except BaseException:
                self.status.publish('FAILED', finished_round, len(self._submissions), self._quorum)
                raise
            self._global_state = aggregated
            self._completed_round = finished_round
            self._submissions.clear()
            if self._completed_round >= self._total_rounds:
                self.status.publish('COMPLETE', finished_round, self._quorum, self._quorum)
                return self._snapshot(RoundState.DONE)
            self._global_round += 1
            self.status.publish('WAITING', self._global_round, 0, self._quorum)
            return self._snapshot(RoundState.TRAIN)

    def _require_state(self) -> dict[str, torch.Tensor]:
        if self._global_state is None:
            raise NotBootstrapped(
                "prefix state is not initialized; call Bootstrap exactly once first"
            )
        return self._global_state

    def _snapshot(self, state: RoundState) -> RoundSnapshot:
        tensors = self._require_state()
        round_number = (
            self._completed_round if state is RoundState.DONE else self._global_round
        )
        return RoundSnapshot(
            state=state,
            global_round=round_number,
            tensors=_copy_state(tensors),
            submissions_received=len(self._submissions),
            submission_quorum=self._quorum,
        )

    def _save_checkpoint(
        self, round_number: int, state: Mapping[str, torch.Tensor]
    ) -> None:
        if self._checkpoint_root is None:
            return
        save_lora_checkpoint(
            self._checkpoint_root / f"prefix-round-{round_number:04d}",
            state,
            {
                "kind": "prefix_lora",
                "global_round": round_number,
                "submission_quorum": self._quorum,
            },
        )


def _validate_client_id(client_id: str) -> None:
    if not client_id or len(client_id) > 128:
        raise InvalidRequest("client_id must contain 1 to 128 characters")
    if any(character.isspace() or ord(character) < 33 or ord(character) > 126 for character in client_id):
        raise InvalidRequest("client_id must contain printable ASCII without whitespace")
