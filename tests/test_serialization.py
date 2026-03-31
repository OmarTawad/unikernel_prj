"""Tests for tensor serialization."""

import numpy as np

from unisplit.shared.serialization import (
    bytes_to_tensor,
    decode_payload,
    encode_payload,
    tensor_to_bytes,
)


class TestSerialization:
    def test_bytes_roundtrip_float32(self):
        original = np.random.randn(32, 78).astype(np.float32)
        raw = tensor_to_bytes(original)
        recovered = bytes_to_tensor(raw, (32, 78), "float32")
        assert np.allclose(original, recovered)

    def test_bytes_roundtrip_int8(self):
        original = np.random.randint(-127, 127, size=(64,)).astype(np.int8)
        raw = tensor_to_bytes(original)
        recovered = bytes_to_tensor(raw, (64,), "int8")
        assert np.array_equal(original, recovered)

    def test_base64_roundtrip(self):
        original = np.random.randn(128).astype(np.float32)
        encoded = encode_payload(original)
        assert isinstance(encoded, str)
        decoded = decode_payload(encoded, (128,), "float32")
        assert np.allclose(original, decoded)

    def test_base64_roundtrip_2d(self):
        original = np.random.randn(64, 76).astype(np.float32)
        encoded = encode_payload(original)
        decoded = decode_payload(encoded, (64, 76), "float32")
        assert np.allclose(original, decoded)
        assert decoded.shape == (64, 76)
