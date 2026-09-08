"""Name-stable, sample-weighted prefix aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .errors import InvalidRequest


@dataclass(frozen=True)
class WeightedState:
    tensors: Mapping[str, torch.Tensor]
    processed_sequences: int


def validate_lora_state(state: Mapping[str, torch.Tensor]) -> None:
    if not state:
        raise InvalidRequest("LoRA state must contain at least one tensor")
    for name, tensor in state.items():
        if not name or len(name) > 512:
            raise InvalidRequest("LoRA tensor names must contain 1 to 512 characters")
        if tensor.dtype != torch.float32:
            raise InvalidRequest(f"LoRA tensor {name!r} must be float32")
        if tensor.ndim < 1:
            raise InvalidRequest(f"LoRA tensor {name!r} must have positive rank")
        if any(dimension <= 0 for dimension in tensor.shape):
            raise InvalidRequest(f"LoRA tensor {name!r} has an empty dimension")
        if not torch.isfinite(tensor).all():
            raise InvalidRequest(f"LoRA tensor {name!r} contains NaN or infinity")


def weighted_average(updates: Sequence[WeightedState]) -> dict[str, torch.Tensor]:
    """Average each explicitly named tensor using sequence counts as weights."""

    if not updates:
        raise InvalidRequest("at least one update is required")
    reference_names = set(updates[0].tensors)
    validate_lora_state(updates[0].tensors)
    reference = updates[0].tensors
    total_weight = 0
    accumulators = {
        name: torch.zeros_like(tensor, dtype=torch.float64, device="cpu")
        for name, tensor in reference.items()
    }
    for update in updates:
        if isinstance(update.processed_sequences, bool) or update.processed_sequences <= 0:
            raise InvalidRequest("processed sequence count must be positive")
        validate_lora_state(update.tensors)
        if set(update.tensors) != reference_names:
            raise InvalidRequest("all updates must contain exactly the same tensor names")
        total_weight += update.processed_sequences
        for name, tensor in update.tensors.items():
            if tensor.shape != reference[name].shape:
                raise InvalidRequest(
                    f"shape mismatch for tensor {name!r}: "
                    f"{tuple(tensor.shape)} != {tuple(reference[name].shape)}"
                )
            accumulators[name].add_(
                tensor.detach().to(device="cpu", dtype=torch.float64),
                alpha=update.processed_sequences,
            )
    return {
        name: (accumulator / total_weight).to(dtype=torch.float32)
        for name, accumulator in accumulators.items()
    }

