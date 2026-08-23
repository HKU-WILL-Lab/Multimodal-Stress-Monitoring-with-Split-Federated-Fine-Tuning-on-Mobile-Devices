from __future__ import annotations

import pytest
import torch

from sfl_clean.aggregation import WeightedState, weighted_average
from sfl_clean.errors import InvalidRequest


def test_weighted_average_uses_sequence_counts_per_named_tensor() -> None:
    result = weighted_average(
        [
            WeightedState(
                {"layer.a": torch.tensor([1.0, 3.0]), "layer.b": torch.tensor([[2.0]])},
                1,
            ),
            WeightedState(
                {"layer.b": torch.tensor([[8.0]]), "layer.a": torch.tensor([7.0, 9.0])},
                3,
            ),
        ]
    )
    assert torch.equal(result["layer.a"], torch.tensor([5.5, 7.5]))
    assert torch.equal(result["layer.b"], torch.tensor([[6.5]]))


def test_weighted_average_rejects_name_and_shape_drift() -> None:
    with pytest.raises(InvalidRequest, match="same tensor names"):
        weighted_average(
            [
                WeightedState({"a": torch.ones(1)}, 1),
                WeightedState({"b": torch.ones(1)}, 1),
            ]
        )
    with pytest.raises(InvalidRequest, match="shape mismatch"):
        weighted_average(
            [
                WeightedState({"a": torch.ones(1)}, 1),
                WeightedState({"a": torch.ones(2)}, 1),
            ]
        )

