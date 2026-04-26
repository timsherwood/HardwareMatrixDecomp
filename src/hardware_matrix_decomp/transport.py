"""Protobuf serialization for tensor activation messages between tile grids.

The in-process pipeline passes numpy arrays directly for speed.  This module
provides serialize / deserialize helpers so that the same tile grids can be
wired over gRPC for parallel distributed simulation without changing their
forward() interface.
"""

from __future__ import annotations

import numpy as np

from .proto import tile_pb2


def tensor_to_proto(x: np.ndarray, tile_id: str = "", sequence_id: int = 0) -> bytes:
    """Serialise an activation tensor into a TileRequest protobuf message."""
    msg = tile_pb2.TileRequest()
    msg.tile_id = tile_id
    msg.sequence_id = sequence_id
    msg.activation.values.extend(x.astype(np.float32).flatten().tolist())
    msg.activation.shape.extend(x.shape)
    return msg.SerializeToString()


def proto_to_tensor(data: bytes) -> tuple[np.ndarray, str, int]:
    """Deserialise a TileRequest and return (activation, tile_id, sequence_id)."""
    msg = tile_pb2.TileRequest()
    msg.ParseFromString(data)
    shape = tuple(msg.activation.shape)
    arr = np.array(list(msg.activation.values), dtype=np.float32).reshape(shape)
    return arr, msg.tile_id, msg.sequence_id


def response_to_proto(x: np.ndarray, tile_id: str = "", sequence_id: int = 0) -> bytes:
    """Serialise an output tensor into a TileResponse protobuf message."""
    msg = tile_pb2.TileResponse()
    msg.tile_id = tile_id
    msg.sequence_id = sequence_id
    msg.output.values.extend(x.astype(np.float32).flatten().tolist())
    msg.output.shape.extend(x.shape)
    return msg.SerializeToString()


def proto_to_response(data: bytes) -> tuple[np.ndarray, str, int]:
    """Deserialise a TileResponse and return (output, tile_id, sequence_id)."""
    msg = tile_pb2.TileResponse()
    msg.ParseFromString(data)
    shape = tuple(msg.output.shape)
    arr = np.array(list(msg.output.values), dtype=np.float32).reshape(shape)
    return arr, msg.tile_id, msg.sequence_id
