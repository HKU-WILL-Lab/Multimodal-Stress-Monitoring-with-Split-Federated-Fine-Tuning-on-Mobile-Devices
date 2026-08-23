"""Thread-safe suffix optimization at the split boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Mapping

import torch
from torch import nn

from .checkpoint import save_lora_checkpoint
from .errors import InvalidRequest
from .objective import shifted_token_objective


@dataclass(frozen=True)
class StepResult:
    activation_gradient: torch.Tensor
    loss: float
    token_accuracy: float
    server_step: int


class SuffixTrainer:
    """Serializes updates so persistent suffix LoRA state cannot race."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        device: torch.device,
        gradient_clip_norm: float | None = 1.0,
        checkpoint_root: Path | None = None,
    ) -> None:
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise InvalidRequest("gradient_clip_norm must be positive when enabled")
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.gradient_clip_norm = gradient_clip_norm
        self.checkpoint_root = checkpoint_root
        self._lock = RLock()
        self._server_step = 0

    def train_step(
        self,
        activation: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> StepResult:
        if activation.dtype != torch.float32 or activation.ndim != 3:
            raise InvalidRequest("boundary activation must be rank-three float32")
        if not torch.isfinite(activation).all():
            raise InvalidRequest("boundary activation contains NaN or infinity")
        if token_ids.dtype != torch.int64 or attention_mask.dtype != torch.int64:
            raise InvalidRequest("token IDs and attention mask must be int64")
        if token_ids.ndim != 2 or attention_mask.ndim != 2:
            raise InvalidRequest("token IDs and attention mask must have rank two")
        if activation.shape[:2] != token_ids.shape or token_ids.shape != attention_mask.shape:
            raise InvalidRequest("split tensors have inconsistent batch or sequence dimensions")

        with self._lock:
            self.model.train()
            boundary = activation.detach().to(self.device).requires_grad_(True)
            ids = token_ids.to(self.device)
            mask = attention_mask.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(boundary, mask)
            if not isinstance(logits, torch.Tensor):
                raise InvalidRequest("suffix model must return one logits tensor")
            if not torch.isfinite(logits).all():
                raise InvalidRequest("suffix produced non-finite logits")
            loss, accuracy = shifted_token_objective(logits, ids, mask)
            loss.backward()
            if boundary.grad is None:
                raise RuntimeError("suffix graph did not produce a boundary gradient")
            gradient = boundary.grad.detach().to(device="cpu", dtype=torch.float32).clone()
            if not torch.isfinite(gradient).all():
                raise RuntimeError("suffix produced a non-finite boundary gradient")
            if self.gradient_clip_norm is not None:
                trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
                if trainable:
                    torch.nn.utils.clip_grad_norm_(trainable, self.gradient_clip_norm)
            self.optimizer.step()
            self._server_step += 1
            self._save_checkpoint()
            return StepResult(
                activation_gradient=gradient,
                loss=float(loss.detach().cpu()),
                token_accuracy=float(accuracy.detach().cpu()),
                server_step=self._server_step,
            )

    def _save_checkpoint(self) -> None:
        if self.checkpoint_root is None:
            return
        state = trainable_lora_state(self.model)
        save_lora_checkpoint(
            self.checkpoint_root / f"suffix-step-{self._server_step:08d}",
            state,
            {"kind": "suffix_lora", "server_step": self._server_step},
        )


def trainable_lora_state(model: nn.Module) -> Mapping[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" not in name:
            raise InvalidRequest(
                f"trainable non-LoRA parameter {name!r} would violate checkpoint policy"
            )
        state[name] = parameter.detach().to(device="cpu", dtype=torch.float32).clone()
    if not state:
        raise InvalidRequest("suffix model has no trainable LoRA parameters")
    return state

