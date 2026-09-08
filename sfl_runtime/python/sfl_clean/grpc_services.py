"""gRPC adapters for suffix training and round coordination."""

from __future__ import annotations

from concurrent import futures
import hashlib
import logging
from threading import RLock
import time
from types import ModuleType

import grpc
import torch

from . import PROTOCOL_VERSION
from .coordinator import RoundCoordinator, RoundSnapshot, RoundState
from .errors import DuplicateUpdate, InvalidRequest, NotBootstrapped, SflError, StateConflict
from .metrics import JsonlMetricWriter, MetricRecord, available_gpu_memory_bytes
from .suffix import SuffixTrainer
from .tensor_codec import (
    TensorSpec,
    decode_named_tensors,
    decode_tensor,
    encode_named_tensors,
    encode_tensor,
)


class SuffixRpcService:
    def __init__(
        self,
        pb2: ModuleType,
        trainer: SuffixTrainer,
        *,
        run_id: str,
        cut_layer: int,
        metrics: JsonlMetricWriter,
        replay_capacity: int = 10_000,
    ) -> None:
        self.pb2 = pb2
        self.trainer = trainer
        self.run_id = run_id
        self.cut_layer = cut_layer
        self.metrics = metrics
        self.replay_capacity = replay_capacity
        self._lock = RLock()
        self._replies: dict[tuple[str, int, int], tuple[bytes, bytes]] = {}

    def TrainSplitStep(self, request: object, context: grpc.ServicerContext) -> object:
        started = time.perf_counter()
        try:
            _validate_protocol(int(request.protocol_version))
            _validate_client_id(request.client_id)
            if int(request.global_round) <= 0:
                raise InvalidRequest("global_round must be positive")
            if int(request.cut_layer) != self.cut_layer:
                raise InvalidRequest(
                    f"cut_layer {request.cut_layer} does not match server cut {self.cut_layer}"
                )
            key = (request.client_id, int(request.global_round), int(request.local_step))
            digest = hashlib.sha256(request.SerializeToString(deterministic=True)).digest()
            with self._lock:
                replay = self._replies.get(key)
                if replay is not None:
                    prior_digest, serialized_reply = replay
                    if prior_digest != digest:
                        raise DuplicateUpdate(
                            "client reused a round/local-step key with different tensor content"
                        )
                    response = self.pb2.SplitStepResponse.FromString(serialized_reply)
                    return response
                activation = decode_tensor(
                    request.activation,
                    spec=TensorSpec("boundary_activation", torch.float32, 3),
                )
                token_ids = decode_tensor(
                    request.token_ids, spec=TensorSpec("token_ids", torch.int64, 2)
                )
                attention_mask = decode_tensor(
                    request.attention_mask,
                    spec=TensorSpec("attention_mask", torch.int64, 2),
                )
                loss_mask = decode_tensor(
                    request.loss_mask,
                    spec=TensorSpec("loss_mask", torch.int64, 2),
                )
                if (
                    activation.shape[:2] != token_ids.shape
                    or token_ids.shape != attention_mask.shape
                    or token_ids.shape != loss_mask.shape
                ):
                    raise InvalidRequest("split tensor batch/sequence dimensions do not match")
                if not torch.logical_or(attention_mask == 0, attention_mask == 1).all():
                    raise InvalidRequest("attention mask values must be zero or one")
                result = self.trainer.train_step(
                    activation, token_ids, attention_mask, loss_mask
                )
                response = self.pb2.SplitStepResponse(
                    activation_gradient=encode_tensor(
                        result.activation_gradient,
                        "boundary_activation_gradient",
                        self.pb2.Tensor,
                    ),
                    loss=result.loss,
                    token_accuracy=result.token_accuracy,
                    server_step=result.server_step,
                )
                if len(self._replies) >= self.replay_capacity:
                    self._replies.pop(next(iter(self._replies)))
                self._replies[key] = (digest, response.SerializeToString(deterministic=True))
            elapsed = time.perf_counter() - started
            self.metrics.append(
                MetricRecord(
                    schema_version=1,
                    run_id=self.run_id,
                    client_id=request.client_id,
                    global_round=int(request.global_round),
                    local_step=int(request.local_step),
                    batch_size=int(token_ids.shape[0]),
                    sequence_length=int(token_ids.shape[1]),
                    loss=result.loss,
                    token_accuracy=result.token_accuracy,
                    split_rpc_bytes=request.ByteSize() + response.ByteSize(),
                    duration_seconds=elapsed,
                    host_gpu_memory_available_bytes=available_gpu_memory_bytes(
                        self.trainer.device
                    ),
                )
            )
            return response
        except SflError as error:
            _abort(context, error)
        except Exception:
            logging.exception("unexpected suffix step failure")
            context.abort(grpc.StatusCode.INTERNAL, "suffix step failed; inspect host logs")


class CoordinatorRpcService:
    def __init__(self, pb2: ModuleType, coordinator: RoundCoordinator) -> None:
        self.pb2 = pb2
        self.coordinator = coordinator

    def Bootstrap(self, request: object, context: grpc.ServicerContext) -> object:
        try:
            _validate_protocol(int(request.protocol_version))
            tensors = decode_named_tensors(request.prefix_tensors)
            snapshot = self.coordinator.bootstrap(request.client_id, tensors)
            return self.pb2.BootstrapResponse(
                state=_wire_state(self.pb2, snapshot.state),
                global_round=snapshot.global_round,
                prefix_tensors=encode_named_tensors(snapshot.tensors, self.pb2.Tensor),
            )
        except SflError as error:
            _abort(context, error)
        except Exception:
            logging.exception("unexpected coordinator bootstrap failure")
            context.abort(grpc.StatusCode.INTERNAL, "bootstrap failed; inspect host logs")

    def FetchRound(self, request: object, context: grpc.ServicerContext) -> object:
        try:
            _validate_protocol(int(request.protocol_version))
            snapshot = self.coordinator.fetch(
                request.client_id, int(request.last_completed_round)
            )
            return _fetch_response(self.pb2, snapshot)
        except SflError as error:
            _abort(context, error)
        except Exception:
            logging.exception("unexpected coordinator fetch failure")
            context.abort(grpc.StatusCode.INTERNAL, "round fetch failed; inspect host logs")

    def SubmitUpdate(self, request: object, context: grpc.ServicerContext) -> object:
        try:
            _validate_protocol(int(request.protocol_version))
            tensors = decode_named_tensors(request.prefix_tensors)
            snapshot = self.coordinator.submit(
                request.client_id,
                int(request.global_round),
                int(request.processed_sequences),
                tensors,
            )
            return self.pb2.SubmitUpdateResponse(
                state=_wire_state(self.pb2, snapshot.state),
                global_round=snapshot.global_round,
                submissions_received=snapshot.submissions_received,
                submission_quorum=snapshot.submission_quorum,
            )
        except SflError as error:
            _abort(context, error)
        except Exception:
            logging.exception("unexpected coordinator submission failure")
            context.abort(grpc.StatusCode.INTERNAL, "round submission failed; inspect host logs")


def make_server(bind: str, max_message_bytes: int, worker_threads: int) -> grpc.Server:
    if max_message_bytes <= 0 or worker_threads <= 0:
        raise InvalidRequest("RPC message size and worker count must be positive")
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=worker_threads),
        options=(
            ("grpc.max_receive_message_length", max_message_bytes),
            ("grpc.max_send_message_length", max_message_bytes),
        ),
    )
    if server.add_insecure_port(bind) == 0:
        raise RuntimeError(f"could not bind gRPC server to {bind}")
    return server


def _fetch_response(pb2: ModuleType, snapshot: RoundSnapshot) -> object:
    return pb2.FetchRoundResponse(
        state=_wire_state(pb2, snapshot.state),
        global_round=snapshot.global_round,
        prefix_tensors=encode_named_tensors(snapshot.tensors, pb2.Tensor),
        submissions_received=snapshot.submissions_received,
        submission_quorum=snapshot.submission_quorum,
    )


def _wire_state(pb2: ModuleType, state: RoundState) -> int:
    return {
        RoundState.TRAIN: pb2.TRAIN,
        RoundState.WAIT: pb2.WAIT,
        RoundState.DONE: pb2.DONE,
    }[state]


def _validate_protocol(version: int) -> None:
    if version != PROTOCOL_VERSION:
        raise InvalidRequest(
            f"unsupported protocol_version {version}; expected {PROTOCOL_VERSION}"
        )


def _validate_client_id(client_id: str) -> None:
    if not client_id or len(client_id) > 128:
        raise InvalidRequest("client_id must contain 1 to 128 characters")
    if any(character.isspace() or ord(character) < 33 or ord(character) > 126 for character in client_id):
        raise InvalidRequest("client_id must contain printable ASCII without whitespace")


def _abort(context: grpc.ServicerContext, error: SflError) -> None:
    if isinstance(error, InvalidRequest):
        code = grpc.StatusCode.INVALID_ARGUMENT
    elif isinstance(error, DuplicateUpdate):
        code = grpc.StatusCode.ALREADY_EXISTS
    elif isinstance(error, NotBootstrapped):
        code = grpc.StatusCode.FAILED_PRECONDITION
    elif isinstance(error, StateConflict):
        code = grpc.StatusCode.FAILED_PRECONDITION
    else:
        code = grpc.StatusCode.UNKNOWN
    context.abort(code, str(error))
