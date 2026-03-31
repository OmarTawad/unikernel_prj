"""Symmetric int8 quantization for communication-efficient activation transport.

Implements the post-split quantization described in the paper (§4.3):
the intermediate activation h_k(x) is quantized to 8-bit integers before
transmission, reducing the communication payload by 4× relative to float32.
"""

from __future__ import annotations

import numpy as np

from unisplit.shared.schemas import QuantizationParams


def quantize_int8(tensor: np.ndarray) -> tuple[np.ndarray, QuantizationParams]:
    """Symmetric int8 quantization.

    Maps float32 values to int8 range [-127, 127] using:
        scale = max(|tensor|) / 127
        quantized = round(tensor / scale)

    Args:
        tensor: Float32 numpy array to quantize.

    Returns:
        Tuple of (quantized int8 array, quantization parameters).
    """
    tensor = tensor.astype(np.float32)

    abs_max = np.abs(tensor).max()
    if abs_max < 1e-10:
        # All-zero tensor — scale doesn't matter
        scale = 1.0
    else:
        scale = float(abs_max / 127.0)

    quantized = np.clip(np.round(tensor / scale), -127, 127).astype(np.int8)
    params = QuantizationParams(scale=scale, zero_point=0, dtype="int8")
    return quantized, params


def dequantize_int8(quantized: np.ndarray, params: QuantizationParams) -> np.ndarray:
    """Dequantize int8 tensor back to float32.

    Reconstructs: tensor ≈ quantized * scale

    Args:
        quantized: Int8 numpy array.
        params: Quantization parameters from quantize_int8.

    Returns:
        Dequantized float32 numpy array.
    """
    return quantized.astype(np.float32) * params.scale
