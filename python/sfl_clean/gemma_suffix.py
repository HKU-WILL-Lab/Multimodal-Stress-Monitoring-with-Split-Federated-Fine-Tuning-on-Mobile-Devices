"""Gemma 3 270M decoder suffix with LoRA adapters."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .errors import InvalidRequest


class Gemma3Suffix(nn.Module):
    """Runs only decoder blocks at or above a positive split cut."""

    def __init__(self, causal_model: nn.Module, cut_layer: int) -> None:
        super().__init__()
        self.cut_layer = cut_layer
        self.text_model = causal_model.model
        self.lm_head = causal_model.lm_head
        self.config = causal_model.config

    def forward(self, boundary: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )

        if boundary.ndim != 3:
            raise InvalidRequest("Gemma boundary activation must have rank three")
        if boundary.shape[-1] != self.config.hidden_size:
            raise InvalidRequest(
                f"Gemma boundary hidden size is {boundary.shape[-1]}; "
                f"expected {self.config.hidden_size}"
            )
        compute_dtype = next(self.text_model.layers.parameters()).dtype
        hidden_states = boundary.to(dtype=compute_dtype)
        position_ids = torch.arange(
            hidden_states.shape[1], device=hidden_states.device, dtype=torch.long
        ).unsqueeze(0)
        mask_arguments: dict[str, Any] = {
            "config": self.config,
            "inputs_embeds": hidden_states,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        causal_masks = {
            "full_attention": create_causal_mask(**mask_arguments),
            "sliding_attention": create_sliding_window_causal_mask(**mask_arguments),
        }
        position_embeddings = {
            layer_type: self.text_model.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in set(self.config.layer_types)
        }
        for decoder_layer in self.text_model.layers:
            layer_index = int(decoder_layer.self_attn.layer_idx)
            layer_type = self.config.layer_types[layer_index]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_masks[layer_type],
                position_embeddings=position_embeddings[layer_type],
                position_ids=position_ids,
                past_key_values=None,
            )
        hidden_states = self.text_model.norm(hidden_states)
        return self.lm_head(hidden_states).float()


def load_gemma3_270m_suffix(
    source: str,
    *,
    cut_layer: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    device: torch.device,
    local_files_only: bool,
) -> Gemma3Suffix:
    """Load the supported architecture, discard prefix blocks, and inject suffix LoRA."""

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    if cut_layer <= 0:
        raise InvalidRequest("cut_layer must be positive")
    if lora_rank <= 0 or lora_alpha <= 0:
        raise InvalidRequest("LoRA rank and alpha must be positive")
    if not 0.0 <= lora_dropout < 1.0:
        raise InvalidRequest("LoRA dropout must be in [0, 1)")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        source, torch_dtype=dtype, local_files_only=local_files_only
    )
    config = base.config
    if (
        config.model_type != "gemma3_text"
        or int(config.num_hidden_layers) != 18
        or int(config.hidden_size) != 640
    ):
        raise InvalidRequest("only the Gemma 3 270M text architecture is supported")
    if cut_layer >= int(config.num_hidden_layers):
        raise InvalidRequest(
            f"cut_layer must be below layer count {config.num_hidden_layers}"
        )
    base.model.layers = nn.ModuleList(list(base.model.layers[cut_layer:]))
    lora = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    adapted = get_peft_model(base, lora)
    causal_model = adapted.get_base_model()
    suffix = Gemma3Suffix(causal_model, cut_layer)
    suffix.to(device)
    for name, parameter in suffix.named_parameters():
        if parameter.requires_grad and "lora_" not in name:
            raise RuntimeError(f"PEFT left non-LoRA parameter trainable: {name}")
    return suffix

