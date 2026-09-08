"""Strict little-endian protobuf tensor encoding and decoding."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np
import torch

from .errors import InvalidRequest


FLOAT32 = 1
INT64 = 2


class TensorMessage(Protocol):
    name: str
    shape: Sequence[int]
    dtype: int
    data: bytes


@dataclass(frozen=True)
class TensorLimits:
    max_rank: int = 8
    max_dimension: int = 1_048_576
    max_elements: int = 67_108_864
    max_bytes: int = 268_435_456


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: torch.dtype
    rank: int
    shape: tuple[int | None, ...] | None = None


def _dtype_details(wire_dtype: int) -> tuple[torch.dtype, np.dtype, int]:
    if wire_dtype == FLOAT32:
        return torch.float32, np.dtype("<f4"), 4
    if wire_dtype == INT64:
        return torch.int64, np.dtype("<i8"), 8
    raise InvalidRequest(f"unsupported tensor dtype value {wire_dtype}")


def _checked_shape(shape: Iterable[int], limits: TensorLimits) -> tuple[int, ...]:
    normalized = tuple(int(dimension) for dimension in shape)
    if not normalized:
        raise InvalidRequest("tensor rank must be at least one")
    if len(normalized) > limits.max_rank:
        raise InvalidRequest(
            f"tensor rank {len(normalized)} exceeds maximum {limits.max_rank}"
        )
    for dimension in normalized:
        if dimension <= 0:
            raise InvalidRequest("tensor dimensions must be positive")
        if dimension > limits.max_dimension:
            raise InvalidRequest(
                f"tensor dimension {dimension} exceeds maximum {limits.max_dimension}"
            )
    elements = prod(normalized)
    if elements > limits.max_elements:
        raise InvalidRequest(
            f"tensor has {elements} elements; maximum is {limits.max_elements}"
        )
    return normalized


def decode_tensor(
    message: TensorMessage,
    *,
    spec: TensorSpec | None = None,
    limits: TensorLimits = TensorLimits(),
    require_finite: bool = True,
) -> torch.Tensor:
    """Decode one tensor after checking identity, layout, and numeric validity."""

    if not message.name or len(message.name) > 512:
        raise InvalidRequest("tensor name must contain 1 to 512 characters")
    shape = _checked_shape(message.shape, limits)
    torch_dtype, numpy_dtype, item_size = _dtype_details(int(message.dtype))
    expected_bytes = prod(shape) * item_size
    if expected_bytes > limits.max_bytes:
        raise InvalidRequest(
            f"tensor byte length {expected_bytes} exceeds maximum {limits.max_bytes}"
        )
    if len(message.data) != expected_bytes:
        raise InvalidRequest(
            f"tensor {message.name!r} has {len(message.data)} data bytes; "
            f"expected {expected_bytes} for shape {shape}"
        )
    if spec is not None:
        if message.name != spec.name:
            raise InvalidRequest(
                f"expected tensor name {spec.name!r}, received {message.name!r}"
            )
        if torch_dtype != spec.dtype:
            raise InvalidRequest(
                f"tensor {message.name!r} has dtype {torch_dtype}; expected {spec.dtype}"
            )
        if len(shape) != spec.rank:
            raise InvalidRequest(
                f"tensor {message.name!r} has rank {len(shape)}; expected {spec.rank}"
            )
        if spec.shape is not None:
            if len(spec.shape) != len(shape):
                raise InvalidRequest(f"invalid expected shape for tensor {message.name!r}")
            for actual, expected in zip(shape, spec.shape, strict=True):
                if expected is not None and actual != expected:
                    raise InvalidRequest(
                        f"tensor {message.name!r} has shape {shape}; expected {spec.shape}"
                    )

    # Copy severs the tensor's lifetime from the protobuf backing bytes.
    array = np.frombuffer(message.data, dtype=numpy_dtype).copy().reshape(shape)
    tensor = torch.from_numpy(array)
    if require_finite and tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise InvalidRequest(f"tensor {message.name!r} contains NaN or infinity")
    return tensor


def encode_tensor(tensor: torch.Tensor, name: str, message_type: type) -> object:
    """Encode a dense CPU copy into a protobuf Tensor-compatible class."""

    if not name or len(name) > 512:
        raise InvalidRequest("tensor name must contain 1 to 512 characters")
    if tensor.layout != torch.strided:
        raise InvalidRequest(f"tensor {name!r} must use dense strided layout")
    if tensor.dtype == torch.float32:
        wire_dtype, numpy_dtype = FLOAT32, np.dtype("<f4")
    elif tensor.dtype == torch.int64:
        wire_dtype, numpy_dtype = INT64, np.dtype("<i8")
    else:
        raise InvalidRequest(
            f"tensor {name!r} has unsupported dtype {tensor.dtype}; "
            "only float32 and int64 are allowed"
        )
    value = tensor.detach().to(device="cpu").contiguous()
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise InvalidRequest(f"tensor {name!r} contains NaN or infinity")
    array = np.asarray(value).astype(numpy_dtype, copy=False)
    return message_type(
        name=name,
        shape=list(value.shape),
        dtype=wire_dtype,
        data=array.tobytes(order="C"),
    )


def decode_named_tensors(
    messages: Iterable[TensorMessage],
    *,
    expected: Mapping[str, TensorSpec] | None = None,
    limits: TensorLimits = TensorLimits(),
) -> dict[str, torch.Tensor]:
    """Decode named tensors, rejecting duplicates and schema drift."""

    result: dict[str, torch.Tensor] = {}
    for message in messages:
        if message.name in result:
            raise InvalidRequest(f"duplicate tensor name {message.name!r}")
        spec = expected.get(message.name) if expected is not None else None
        if expected is not None and spec is None:
            raise InvalidRequest(f"unexpected tensor name {message.name!r}")
        result[message.name] = decode_tensor(message, spec=spec, limits=limits)
    if expected is not None:
        missing = sorted(set(expected).difference(result))
        if missing:
            raise InvalidRequest(f"missing tensors: {', '.join(missing)}")
    if not result:
        raise InvalidRequest("at least one named tensor is required")
    return result


def specs_from_state(state: Mapping[str, torch.Tensor]) -> dict[str, TensorSpec]:
    return {
        name: TensorSpec(name, tensor.dtype, tensor.ndim, tuple(tensor.shape))
        for name, tensor in state.items()
    }


def encode_named_tensors(
    state: Mapping[str, torch.Tensor], message_type: type
) -> list[object]:
    return [encode_tensor(state[name], name, message_type) for name in sorted(state)]

