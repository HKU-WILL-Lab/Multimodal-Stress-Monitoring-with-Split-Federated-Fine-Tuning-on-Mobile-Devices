from __future__ import annotations

import torch
from torch.nn import functional as F

from sfl_clean.objective import shifted_token_objective


def test_shifted_objective_ignores_padded_targets() -> None:
    logits = torch.tensor(
        [[[0.0, 3.0, 0.0], [9.0, 0.0, 0.0], [0.0, 0.0, 9.0]]],
        requires_grad=True,
    )
    token_ids = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.int64)
    loss, accuracy = shifted_token_objective(logits, token_ids, attention_mask)
    expected = F.cross_entropy(logits[:, 0, :], torch.tensor([1]))
    assert torch.allclose(loss, expected)
    assert accuracy.item() == 1.0
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, 1:, :]) == 0


def test_loss_mask_excludes_sensor_and_prompt_tokens() -> None:
    logits = torch.zeros(1, 4, 5, requires_grad=True)
    token_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    attention_mask = torch.ones(1, 4, dtype=torch.int64)
    loss_mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.int64)
    loss, _ = shifted_token_objective(logits, token_ids, attention_mask, loss_mask)
    loss.backward()
    assert loss.isfinite()
    assert torch.count_nonzero(logits.grad[:, 0, :]) == 0
    assert torch.count_nonzero(logits.grad[:, 1:3, :]) > 0
