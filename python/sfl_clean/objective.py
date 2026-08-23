"""Causal next-token objective used by the suffix trainer."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .errors import InvalidRequest


def shifted_token_objective(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean shifted cross entropy and accuracy over non-padding targets."""

    if logits.ndim != 3:
        raise InvalidRequest("logits must have shape [batch, sequence, vocabulary]")
    if token_ids.ndim != 2 or attention_mask.ndim != 2:
        raise InvalidRequest("token IDs and attention mask must have rank two")
    if logits.shape[:2] != token_ids.shape or token_ids.shape != attention_mask.shape:
        raise InvalidRequest("logits, token IDs, and attention mask batch/sequence shapes differ")
    if logits.shape[1] < 2:
        raise InvalidRequest("sequence length must be at least two")
    if logits.shape[2] < 2:
        raise InvalidRequest("vocabulary dimension must be at least two")
    if token_ids.dtype != torch.int64 or attention_mask.dtype != torch.int64:
        raise InvalidRequest("token IDs and attention mask must be int64")
    if (token_ids < 0).any() or (token_ids >= logits.shape[2]).any():
        raise InvalidRequest("token IDs are outside the model vocabulary")
    if not torch.logical_or(attention_mask == 0, attention_mask == 1).all():
        raise InvalidRequest("attention mask values must be zero or one")
    target_mask = attention_mask[:, 1:].to(dtype=torch.bool)
    if not target_mask.any():
        raise InvalidRequest("batch contains no non-padding prediction target")
    shifted_logits = logits[:, :-1, :]
    shifted_targets = token_ids[:, 1:]
    loss = F.cross_entropy(shifted_logits[target_mask], shifted_targets[target_mask])
    predictions = shifted_logits.argmax(dim=-1)
    accuracy = (predictions[target_mask] == shifted_targets[target_mask]).float().mean()
    return loss, accuracy

