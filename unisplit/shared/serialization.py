"""Tensor serialization utilities for edge-cloud transport.

Handles numpy array ↔ bytes ↔ base64 string conversions for JSON-based
API transport.
"""

from __future__ import annotations

import base64

import numpy as np


def tensor_to_bytes(arr: np.ndarray) -> bytes:
    """Convert numpy array to raw bytes."""
    return arr.tobytes()


def bytes_to_tensor(data: bytes, shape: list[int] | tuple[int, ...], dtype: str = "float32") -> np.ndarray:
    """Reconstruct numpy array from raw bytes.

    Args:
        data: Raw byte buffer.
        shape: Target tensor shape.
        dtype: Numpy dtype string ('float32', 'int8', etc.).

    Returns:
        Reconstructed numpy array.
    """
    np_dtype = np.dtype(dtype)
    arr = np.frombuffer(data, dtype=np_dtype)
    return arr.reshape(shape)


def encode_payload(arr: np.ndarray) -> str:
    """Encode numpy array as base64 string for JSON transport.

    Args:
        arr: Numpy array to encode.

    Returns:
        Base64-encoded string of the array's raw bytes.
    """
    raw = tensor_to_bytes(arr)
    return base64.b64encode(raw).decode("ascii")


def decode_payload(b64str: str, shape: list[int] | tuple[int, ...], dtype: str = "float32") -> np.ndarray:
    """Decode base64 string back to numpy array.

    Args:
        b64str: Base64-encoded tensor bytes.
        shape: Target tensor shape.
        dtype: Numpy dtype string.

    Returns:
        Reconstructed numpy array.
    """
    raw = base64.b64decode(b64str)
    return bytes_to_tensor(raw, shape, dtype)
