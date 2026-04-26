import numpy as np

from hardware_matrix_decomp.transport import (
    proto_to_response,
    proto_to_tensor,
    response_to_proto,
    tensor_to_proto,
)


def test_request_roundtrip() -> None:
    x = np.random.default_rng(0).standard_normal((4, 16)).astype(np.float32)
    data = tensor_to_proto(x, tile_id="t0", sequence_id=3)
    recovered, tile_id, seq_id = proto_to_tensor(data)
    assert tile_id == "t0"
    assert seq_id == 3
    np.testing.assert_allclose(recovered, x, rtol=1e-6)


def test_response_roundtrip() -> None:
    x = np.random.default_rng(1).standard_normal((8, 32)).astype(np.float32)
    data = response_to_proto(x, tile_id="grid_A", sequence_id=1)
    recovered, tile_id, seq_id = proto_to_response(data)
    assert tile_id == "grid_A"
    assert seq_id == 1
    np.testing.assert_allclose(recovered, x, rtol=1e-6)


def test_shape_preserved_after_roundtrip() -> None:
    x = np.ones((3, 7, 5), dtype=np.float32).reshape(3, 35)
    data = tensor_to_proto(x)
    recovered, _, _ = proto_to_tensor(data)
    assert recovered.shape == x.shape
