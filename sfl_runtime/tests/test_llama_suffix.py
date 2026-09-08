"""Loader-contract tests for the Llama 3.2 suffix."""

from __future__ import annotations

from types import ModuleType
import sys

import pytest
import torch

from sfl_clean.llama_suffix import load_llama32_1b_suffix


class LoadStopped(Exception):
    pass


def test_loader_uses_transformers_dtype_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(source: str, **kwargs: object) -> object:
            captured["source"] = source
            captured.update(kwargs)
            raise LoadStopped

    fake_peft = ModuleType("peft")
    fake_peft.LoraConfig = object
    fake_peft.get_peft_model = object
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(LoadStopped):
        load_llama32_1b_suffix(
            "local-model",
            cut_layer=1,
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.0,
            device=torch.device("cpu"),
            local_files_only=True,
        )

    assert captured == {
        "source": "local-model",
        "dtype": torch.float32,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }


def test_embedding_boundary_reaches_model_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            raise LoadStopped

    fake_peft = ModuleType("peft")
    fake_peft.LoraConfig = object
    fake_peft.get_peft_model = object
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(LoadStopped):
        load_llama32_1b_suffix(
            "local-model",
            cut_layer=0,
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.0,
            device=torch.device("cpu"),
            local_files_only=True,
        )
