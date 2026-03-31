"""Cloud inference engine — orchestrates deserialization, dequantization, and inference."""

from __future__ import annotations

import logging
import time

import numpy as np

from unisplit.cloud.backends import InferenceBackend
from unisplit.shared.quantization import dequantize_int8
from unisplit.shared.schemas import InferenceTiming, QuantizationParams, SplitInferenceRequest
from unisplit.shared.serialization import decode_payload

logger = logging.getLogger("unisplit.cloud.inference")


class CloudInferenceEngine:
    """Orchestrates the cloud-side inference pipeline.

    Pipeline: deserialize → dequantize (if int8) → infer → time everything
    """

    def __init__(self, backend: InferenceBackend):
        self.backend = backend

    def run_split_inference(
        self, request: SplitInferenceRequest
    ) -> tuple[np.ndarray, InferenceTiming]:
        """Execute split inference and return logits + timing.

        Args:
            request: Validated split inference request.

        Returns:
            Tuple of (logits array, timing breakdown).
        """
        total_start = time.perf_counter()

        # Step 1: Deserialize tensor
        deser_start = time.perf_counter()
        activation = decode_payload(
            request.tensor_payload, request.shape, request.dtype
        )
        deserialize_ms = (time.perf_counter() - deser_start) * 1000

        # Step 2: Dequantize if needed
        dequant_start = time.perf_counter()
        if request.dtype == "int8" and request.quantization_params is not None:
            activation = dequantize_int8(activation, request.quantization_params)
        dequantize_ms = (time.perf_counter() - dequant_start) * 1000

        # Step 3: Run inference
        logits, inference_ms = self.backend.infer(activation, request.split_id)

        total_ms = (time.perf_counter() - total_start) * 1000

        timing = InferenceTiming(
            deserialize_ms=round(deserialize_ms, 3),
            dequantize_ms=round(dequantize_ms, 3),
            inference_ms=round(inference_ms, 3),
            total_ms=round(total_ms, 3),
        )

        logger.debug(
            f"Split inference: split_id={request.split_id}, "
            f"timing={timing.total_ms:.2f}ms"
        )

        return logits, timing

    def run_full_inference(self, input_data: np.ndarray) -> tuple[np.ndarray, float]:
        """Run full model inference."""
        return self.backend.infer_full(input_data)
