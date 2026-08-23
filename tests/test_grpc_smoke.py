from __future__ import annotations

from concurrent import futures
import json

import grpc
import pytest
import torch
from torch import nn

from sfl_clean.coordinator import RoundCoordinator
from sfl_clean.grpc_services import CoordinatorRpcService, SuffixRpcService
from sfl_clean.metrics import JsonlMetricWriter
from sfl_clean.proto_runtime import load_bindings
from sfl_clean.suffix import SuffixTrainer
from sfl_clean.tensor_codec import decode_tensor, encode_tensor


pb2, pb2_grpc = load_bindings()


class TinyRandomSuffix(nn.Module):
    def __init__(self, hidden: int, vocabulary: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        self.register_buffer("base_weight", torch.randn(hidden, vocabulary, generator=generator))
        self.lora_delta = nn.Parameter(torch.zeros(hidden, vocabulary))

    def forward(self, activation: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        return activation @ (self.base_weight + self.lora_delta)


def test_end_to_end_random_model_rpc_and_round(tmp_path) -> None:
    model = TinyRandomSuffix(hidden=4, vocabulary=7)
    trainer = SuffixTrainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.05),
        device=torch.device("cpu"),
        checkpoint_root=tmp_path / "checkpoints",
    )
    suffix = SuffixRpcService(
        pb2,
        trainer,
        run_id="test-run",
        cut_layer=1,
        metrics=JsonlMetricWriter(tmp_path / "metrics.jsonl"),
    )
    coordinator = RoundCoordinator(total_rounds=1, submission_quorum=1)
    rounds = CoordinatorRpcService(pb2, coordinator)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_SuffixServiceServicer_to_server(suffix, server)
    pb2_grpc.add_RoundCoordinatorServicer_to_server(rounds, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(
        f"127.0.0.1:{port}",
        options=(
            ("grpc.max_receive_message_length", 1_048_576),
            ("grpc.max_send_message_length", 1_048_576),
        ),
    )
    try:
        suffix_stub = pb2_grpc.SuffixServiceStub(channel)
        round_stub = pb2_grpc.RoundCoordinatorStub(channel)
        with pytest.raises(grpc.RpcError) as not_ready:
            round_stub.FetchRound(
                pb2.FetchRoundRequest(
                    protocol_version=1, client_id="phone-1", last_completed_round=0
                ),
                timeout=3.0,
            )
        assert not_ready.value.code() is grpc.StatusCode.FAILED_PRECONDITION

        prefix = encode_tensor(torch.zeros(2, 2), "prefix.lora_A", pb2.Tensor)
        fetched = round_stub.Bootstrap(
            pb2.BootstrapRequest(
                protocol_version=1, client_id="phone-1", prefix_tensors=[prefix]
            ),
            timeout=3.0,
        )
        assert fetched.state == pb2.TRAIN

        activation = torch.randn(1, 4, 4, generator=torch.Generator().manual_seed(3))
        request = pb2.SplitStepRequest(
            protocol_version=1,
            client_id="phone-1",
            global_round=1,
            local_step=0,
            cut_layer=1,
            activation=encode_tensor(activation, "boundary_activation", pb2.Tensor),
            token_ids=encode_tensor(
                torch.tensor([[1, 2, 3, 4]], dtype=torch.int64), "token_ids", pb2.Tensor
            ),
            attention_mask=encode_tensor(
                torch.ones(1, 4, dtype=torch.int64), "attention_mask", pb2.Tensor
            ),
        )
        reply = suffix_stub.TrainSplitStep(request, timeout=3.0)
        gradient = decode_tensor(reply.activation_gradient)
        assert gradient.shape == activation.shape
        assert reply.server_step == 1
        assert 0.0 <= reply.token_accuracy <= 1.0

        replay = suffix_stub.TrainSplitStep(request, timeout=3.0)
        assert replay.server_step == 1
        assert replay.SerializeToString() == reply.SerializeToString()

        finished = round_stub.SubmitUpdate(
            pb2.SubmitUpdateRequest(
                protocol_version=1,
                client_id="phone-1",
                global_round=1,
                processed_sequences=1,
                prefix_tensors=[
                    encode_tensor(torch.ones(2, 2), "prefix.lora_A", pb2.Tensor)
                ],
            ),
            timeout=3.0,
        )
        assert finished.state == pb2.DONE
    finally:
        channel.close()
        server.stop(grace=None).wait()

    metrics = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(metrics) == 1
    assert metrics[0]["split_rpc_bytes"] > 0
    assert metrics[0]["batch_size"] == 1
    assert metrics[0]["sequence_length"] == 4


def test_protocol_version_error_is_actionable(tmp_path) -> None:
    coordinator = RoundCoordinator(total_rounds=1, submission_quorum=1)
    service = CoordinatorRpcService(pb2, coordinator)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    pb2_grpc.add_RoundCoordinatorServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        stub = pb2_grpc.RoundCoordinatorStub(channel)
        with pytest.raises(grpc.RpcError) as failure:
            stub.FetchRound(
                pb2.FetchRoundRequest(
                    protocol_version=99, client_id="phone-1", last_completed_round=0
                ),
                timeout=3.0,
            )
        assert failure.value.code() is grpc.StatusCode.INVALID_ARGUMENT
        assert "expected 1" in failure.value.details()
    finally:
        channel.close()
        server.stop(grace=None).wait()

