from __future__ import annotations

import torch

from sfl_clean.alignment import (
    SensorTokenProjector,
    conditioned_masks,
    condition_text_embeddings,
    named_alignment_state,
)


def test_projector_prepends_trainable_sensor_tokens() -> None:
    projector = SensorTokenProjector(encoder_width=6, llm_width=8, sensor_tokens=2)
    features = torch.randn(3, 2, 6)
    text = torch.randn(3, 4, 8)
    conditioned = condition_text_embeddings(text, features, projector)
    assert conditioned.shape == (3, 6, 8)
    assert torch.equal(conditioned[:, :4], text)
    conditioned.square().mean().backward()
    assert all(parameter.grad is not None for parameter in projector.parameters())
    assert set(named_alignment_state(projector)) == {
        "alignment.layers.0.bias",
        "alignment.layers.0.weight",
        "alignment.layers.1.bias",
        "alignment.layers.1.weight",
    }


def test_conditioned_masks_exclude_sensor_tokens_from_loss() -> None:
    attention, loss = conditioned_masks(
        torch.tensor([[1, 1, 0]], dtype=torch.int64),
        torch.tensor([[0, 1, 0]], dtype=torch.int64),
        sensor_tokens=2,
    )
    assert attention.tolist() == [[1, 1, 1, 1, 0]]
    assert loss.tolist() == [[0, 0, 0, 1, 0]]
