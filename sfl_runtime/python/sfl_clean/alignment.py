"""Encoder-independent sensor-to-LLM alignment primitives.

The frozen encoder is deliberately outside this module: on phones it is an
ExecuTorch program, while tests and server-side export use ordinary tensors.
Only the projector parameters are trainable and federated.
"""

from __future__ import annotations

import torch
from torch import nn


class SensorTokenProjector(nn.Module):
    """OpenTSLM-SP token-wise LayerNorm--Linear--GELU alignment."""

    def __init__(
        self,
        encoder_width: int,
        llm_width: int,
        *,
        sensor_tokens: int | None = None,
        hidden_width: int | None = None,
    ) -> None:
        super().__init__()
        if encoder_width <= 0 or llm_width <= 0:
            raise ValueError("alignment widths must be positive")
        if sensor_tokens is not None and sensor_tokens <= 0:
            raise ValueError("sensor_tokens must be positive when supplied")
        if hidden_width not in (None, llm_width):
            raise ValueError("OpenTSLM-SP has no hidden projector layer")
        self.encoder_width = encoder_width
        self.llm_width = llm_width
        self.sensor_tokens = sensor_tokens
        self.layers = nn.Sequential(
            nn.LayerNorm(encoder_width),
            nn.Linear(encoder_width, llm_width),
            nn.GELU(),
        )

    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        if encoder_features.ndim != 3 or encoder_features.shape[-1] != self.encoder_width:
            raise ValueError(
                f"encoder_features must have shape [batch, tokens, {self.encoder_width}]"
            )
        if self.sensor_tokens is not None and encoder_features.shape[1] != self.sensor_tokens:
            raise ValueError("encoder token count does not match the deployment manifest")
        return self.layers(encoder_features)


def condition_text_embeddings(
    text_embeddings: torch.Tensor,
    encoder_features: torch.Tensor,
    projector: SensorTokenProjector,
) -> torch.Tensor:
    """Append projected tokens after a prompt embedding (OpenTSLM-SP order)."""
    if text_embeddings.ndim != 3 or text_embeddings.shape[-1] != projector.llm_width:
        raise ValueError("text_embeddings must be [batch, text_tokens, llm_width]")
    if text_embeddings.shape[0] != encoder_features.shape[0]:
        raise ValueError("sensor and text batch sizes differ")
    return torch.cat((text_embeddings, projector(encoder_features)), dim=1)


def conditioned_masks(
    text_attention_mask: torch.Tensor,
    text_loss_mask: torch.Tensor,
    sensor_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keeps sensor tokens visible to attention while excluding them from LM loss."""
    if text_attention_mask.dtype != torch.int64 or text_loss_mask.dtype != torch.int64:
        raise TypeError("text masks must use torch.int64")
    if text_attention_mask.ndim != 2 or text_attention_mask.shape != text_loss_mask.shape:
        raise ValueError("text masks must be rank-two and have identical shapes")
    if sensor_tokens <= 0:
        raise ValueError("sensor_tokens must be positive")
    shape = (text_attention_mask.shape[0], sensor_tokens)
    visible = torch.ones(shape, dtype=torch.int64, device=text_attention_mask.device)
    excluded = torch.zeros(shape, dtype=torch.int64, device=text_loss_mask.device)
    return (
        torch.cat((visible, text_attention_mask), dim=1),
        torch.cat((excluded, text_loss_mask), dim=1),
    )


def named_alignment_state(projector: SensorTokenProjector) -> dict[str, torch.Tensor]:
    """Stable coordinator names for sample-weighted projector aggregation."""
    return {
        f"alignment.{name}": parameter
        for name, parameter in sorted(projector.named_parameters())
    }
