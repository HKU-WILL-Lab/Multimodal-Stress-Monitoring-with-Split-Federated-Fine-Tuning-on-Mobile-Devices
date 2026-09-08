"""Run the persistent GPU suffix training service."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import load_config
from .llama_suffix import load_llama32_1b_suffix
from .grpc_services import SuffixRpcService, make_server
from .metrics import JsonlMetricWriter
from .proto_runtime import load_bindings
from .suffix import SuffixTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    device = torch.device(config.model.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configuration requests CUDA, but PyTorch cannot access a CUDA device")
    model = load_llama32_1b_suffix(
        config.model.source,
        cut_layer=config.training.cut_layer,
        lora_rank=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        device=device,
        local_files_only=config.model.local_files_only,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.model.learning_rate,
    )
    trainer = SuffixTrainer(
        model,
        optimizer,
        device=device,
        gradient_clip_norm=config.training.gradient_clip_norm,
        checkpoint_root=config.checkpoint_root / config.run_id,
        checkpoint_interval_steps=config.training.local_steps,
    )
    pb2, pb2_grpc = load_bindings()
    service = SuffixRpcService(
        pb2,
        trainer,
        run_id=config.run_id,
        cut_layer=config.training.cut_layer,
        metrics=JsonlMetricWriter(config.metrics_path),
    )
    server = make_server(
        config.suffix_rpc.bind,
        config.suffix_rpc.max_message_bytes,
        config.suffix_rpc.worker_threads,
    )
    pb2_grpc.add_SuffixServiceServicer_to_server(service, server)
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=5.0).wait()


if __name__ == "__main__":
    main()
