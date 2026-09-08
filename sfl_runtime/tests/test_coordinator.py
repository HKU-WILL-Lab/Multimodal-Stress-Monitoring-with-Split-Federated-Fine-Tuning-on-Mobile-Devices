from __future__ import annotations

import json

import pytest
import torch

from sfl_clean.coordinator import RoundCoordinator, RoundState
from sfl_clean.errors import DuplicateUpdate, NotBootstrapped, StateConflict


def test_two_round_single_client_state_machine(tmp_path) -> None:
    coordinator = RoundCoordinator(
        total_rounds=2, submission_quorum=1, checkpoint_root=tmp_path
    )
    with pytest.raises(NotBootstrapped):
        coordinator.fetch("phone-1", 0)
    first = coordinator.bootstrap("phone-1", {"prefix.lora_A": torch.tensor([0.0])})
    assert first.state is RoundState.TRAIN
    assert first.global_round == 1

    second = coordinator.submit(
        "phone-1", 1, 2, {"prefix.lora_A": torch.tensor([2.0])}
    )
    assert second.state is RoundState.TRAIN
    assert second.global_round == 2
    done = coordinator.submit(
        "phone-1", 2, 1, {"prefix.lora_A": torch.tensor([4.0])}
    )
    assert done.state is RoundState.DONE
    assert done.global_round == 2
    assert torch.equal(done.tensors["prefix.lora_A"], torch.tensor([4.0]))

    manifest = json.loads(
        (tmp_path / "prefix-round-0002" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["metadata"]["kind"] == "prefix_lora"
    assert {tensor["name"] for tensor in manifest["tensors"]} == {"prefix.lora_A"}


def test_quorum_wait_duplicate_and_weighted_transition() -> None:
    coordinator = RoundCoordinator(total_rounds=1, submission_quorum=2)
    coordinator.bootstrap("phone-a", {"p.lora_A": torch.zeros(2)})
    waiting = coordinator.submit(
        "phone-a", 1, 1, {"p.lora_A": torch.tensor([1.0, 3.0])}
    )
    assert waiting.state is RoundState.WAIT
    assert coordinator.fetch("phone-a", 0).state is RoundState.WAIT
    assert coordinator.fetch("phone-b", 0).state is RoundState.TRAIN
    with pytest.raises(DuplicateUpdate):
        coordinator.submit("phone-a", 1, 1, {"p.lora_A": torch.ones(2)})
    done = coordinator.submit(
        "phone-b", 1, 3, {"p.lora_A": torch.tensor([5.0, 7.0])}
    )
    assert done.state is RoundState.DONE
    assert torch.equal(done.tensors["p.lora_A"], torch.tensor([4.0, 6.0]))
    with pytest.raises(StateConflict):
        coordinator.submit("phone-a", 1, 1, {"p.lora_A": torch.ones(2)})

