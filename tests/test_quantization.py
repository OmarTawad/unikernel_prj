"""Tests for int8 quantization."""

import numpy as np

from unisplit.shared.quantization import dequantize_int8, quantize_int8


class TestQuantization:
    def test_quantize_dequantize_roundtrip(self):
        tensor = np.random.randn(64, 76).astype(np.float32)
        quantized, params = quantize_int8(tensor)
        recovered = dequantize_int8(quantized, params)

        assert quantized.dtype == np.int8
        assert recovered.dtype == np.float32
        assert recovered.shape == tensor.shape

        # Max error should be bounded by scale
        max_err = np.abs(tensor - recovered).max()
        assert max_err < 2 * params.scale

    def test_quantize_preserves_sign(self):
        tensor = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        quantized, params = quantize_int8(tensor)
        recovered = dequantize_int8(quantized, params)
        assert recovered[0] < 0
        assert abs(recovered[1]) < params.scale
        assert recovered[2] > 0

    def test_quantize_zero_tensor(self):
        tensor = np.zeros((10,), dtype=np.float32)
        quantized, params = quantize_int8(tensor)
        recovered = dequantize_int8(quantized, params)
        assert np.allclose(recovered, 0.0)

    def test_quantize_relative_error(self):
        """Paper claims <0.3% F1 degradation. Check max relative error < 2%."""
        tensor = np.random.randn(1000).astype(np.float32) * 10
        quantized, params = quantize_int8(tensor)
        recovered = dequantize_int8(quantized, params)
        input_range = np.abs(tensor).max()
        max_err = np.abs(tensor - recovered).max()
        rel_err = max_err / input_range
        assert rel_err < 0.02  # 2% relative error bound

    def test_quantize_4x_compression(self):
        """int8 should be 4x smaller than float32."""
        tensor = np.random.randn(100).astype(np.float32)
        quantized, _ = quantize_int8(tensor)
        assert quantized.nbytes == tensor.nbytes // 4
