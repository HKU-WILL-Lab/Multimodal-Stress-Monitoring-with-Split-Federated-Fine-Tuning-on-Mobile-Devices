from __future__ import annotations

import struct

import pytest
import torch

from sfl_clean.errors import InvalidRequest
from sfl_clean.proto_runtime import load_bindings
from sfl_clean.tensor_codec import TensorSpec, decode_named_tensors, decode_tensor, encode_tensor


pb2, _ = load_bindings()


def test_float32_round_trip_is_little_endian() -> None:
    value = torch.tensor([[1.5, -2.25]], dtype=torch.float32)
    message = encode_tensor(value, "weights", pb2.Tensor)
    assert message.data == struct.pack("<ff", 1.5, -2.25)
    decoded = decode_tensor(
        message, spec=TensorSpec("weights", torch.float32, 2, (1, 2))
    )
    assert torch.equal(decoded, value)


@pytest.mark.parametrize(
    "message, match",
    [
        (pb2.Tensor(name="x", shape=[2], dtype=pb2.Tensor.FLOAT32, data=b"123"), "data bytes"),
        (
            pb2.Tensor(
                name="x",
                shape=[1],
                dtype=pb2.Tensor.FLOAT32,
                data=struct.pack("<f", float("nan")),
            ),
            "NaN",
        ),
        (pb2.Tensor(name="x", shape=[0], dtype=pb2.Tensor.FLOAT32), "positive"),
        (pb2.Tensor(name="x", shape=[1], dtype=99, data=b"1234"), "unsupported"),
    ],
)
def test_malformed_tensors_are_rejected(message: object, match: str) -> None:
    with pytest.raises(InvalidRequest, match=match):
        decode_tensor(message)


def test_named_tensor_duplicates_and_schema_are_rejected() -> None:
    message = encode_tensor(torch.ones(2), "a", pb2.Tensor)
    with pytest.raises(InvalidRequest, match="duplicate"):
        decode_named_tensors([message, message])
    with pytest.raises(InvalidRequest, match="unexpected"):
        decode_named_tensors(
            [message], expected={"b": TensorSpec("b", torch.float32, 1, (2,))}
        )

