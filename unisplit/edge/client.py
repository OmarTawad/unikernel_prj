"""HTTP client for edge→cloud communication.

Sends split inference requests to the cloud service and returns responses.
Also measures round-trip time from the edge perspective.
"""

from __future__ import annotations

import logging
import time
import uuid

import httpx
import numpy as np

from unisplit.model.registry import get_split_info
from unisplit.shared.constants import DEFAULT_MODEL_VERSION
from unisplit.shared.quantization import quantize_int8
from unisplit.shared.schemas import (
    QuantizationParams,
    SplitInferenceRequest,
    SplitInferenceResponse,
)
from unisplit.shared.serialization import encode_payload

logger = logging.getLogger("unisplit.edge.client")


class CloudClient:
    """Synchronous HTTP client for communicating with the cloud service."""

    def __init__(
        self,
        cloud_url: str = "http://localhost:8000",
        timeout_seconds: float = 10.0,
        model_version: str = DEFAULT_MODEL_VERSION,
        use_quantization: bool = True,
    ):
        self.cloud_url = cloud_url.rstrip("/")
        self.timeout = timeout_seconds
        self.model_version = model_version
        self.use_quantization = use_quantization
        self._client = httpx.Client(timeout=timeout_seconds)

    def health_check(self) -> bool:
        """Check if the cloud service is healthy."""
        try:
            resp = self._client.get(f"{self.cloud_url}/health")
            return resp.status_code == 200
        except Exception:
            return False

    def send_activation(
        self,
        activation: np.ndarray,
        split_id: int,
        request_id: str | None = None,
        trace_metadata: dict | None = None,
    ) -> tuple[SplitInferenceResponse, float]:
        """Send intermediate activation to cloud for split inference.

        Args:
            activation: h_k(x) numpy array from edge partition.
            split_id: Split point identifier.
            request_id: Optional request ID (auto-generated if None).
            trace_metadata: Optional trace context.

        Returns:
            Tuple of (SplitInferenceResponse, round_trip_time_ms).
            The RTT is measured from edge perspective (send → receive).
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        # Quantize if enabled
        if self.use_quantization:
            quantized, quant_params = quantize_int8(activation)
            payload = encode_payload(quantized)
            dtype = "int8"
        else:
            payload = encode_payload(activation.astype(np.float32))
            quant_params = None
            dtype = "float32"

        shape = list(activation.shape)

        request = SplitInferenceRequest(
            request_id=request_id,
            split_id=split_id,
            tensor_payload=payload,
            shape=shape,
            dtype=dtype,
            quantization_params=quant_params,
            model_version=self.model_version,
            trace_metadata=trace_metadata,
        )

        # Measure round-trip time from edge perspective
        rtt_start = time.perf_counter()

        try:
            resp = self._client.post(
                f"{self.cloud_url}/infer/split",
                json=request.model_dump(),
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.error(f"Cloud request failed: {e}")
            raise

        rtt_ms = (time.perf_counter() - rtt_start) * 1000

        response = SplitInferenceResponse(**resp.json())
        return response, rtt_ms

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
