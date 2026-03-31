"""Tests for Pydantic schemas."""

import uuid

from unisplit.shared.schemas import (
    FeasibilityReport,
    HealthResponse,
    InferenceTiming,
    QuantizationParams,
    SplitInferenceRequest,
    SplitInferenceResponse,
    SplitMemoryProfile,
)


class TestSchemas:
    def test_health_response(self):
        resp = HealthResponse(status="ok", version="v0.1.0")
        assert resp.status == "ok"
        data = resp.model_dump()
        assert data["status"] == "ok"

    def test_quantization_params(self):
        params = QuantizationParams(scale=0.5, zero_point=0, dtype="int8")
        assert params.scale == 0.5
        roundtrip = QuantizationParams(**params.model_dump())
        assert roundtrip.scale == params.scale

    def test_split_inference_request_valid(self):
        req = SplitInferenceRequest(
            request_id=str(uuid.uuid4()),
            split_id=7,
            tensor_payload="AAAA",
            shape=[64],
            dtype="float32",
            model_version="v0.1.0",
        )
        assert req.split_id == 7

    def test_split_inference_request_invalid_split_id(self):
        import pytest
        with pytest.raises(Exception):
            SplitInferenceRequest(
                request_id="test",
                split_id=5,  # Not in SUPPORTED_SPLIT_IDS
                tensor_payload="AAAA",
                shape=[64],
                dtype="float32",
                model_version="v0.1.0",
            )

    def test_split_inference_response(self):
        timing = InferenceTiming(
            deserialize_ms=0.1, dequantize_ms=0.0,
            inference_ms=1.5, total_ms=1.6,
        )
        resp = SplitInferenceResponse(
            request_id="test",
            split_id=7,
            predicted_class=0,
            predicted_label="Benign",
            probabilities=[1.0] + [0.0] * 33,
            model_version="v0.1.0",
            timing=timing,
        )
        assert resp.predicted_class == 0
        assert resp.timing.total_ms == 1.6

    def test_split_memory_profile(self):
        profile = SplitMemoryProfile(
            split_id=3,
            split_name="after_block1",
            cumulative_weight_bytes=1000,
            peak_activation_bytes=2000,
            runtime_overhead_bytes=2097152,
            total_edge_memory_bytes=2100152,
            communication_payload_float32_bytes=9984,
            communication_payload_int8_bytes=2496,
            output_shape=[32, 78],
            feasible=True,
        )
        assert profile.feasible
        data = profile.model_dump()
        roundtrip = SplitMemoryProfile(**data)
        assert roundtrip.split_id == 3
