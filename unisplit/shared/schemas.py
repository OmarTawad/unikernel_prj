"""Pydantic v2 schemas for all UniSplit API contracts and data models.

These schemas define the wire format between edge and cloud, plus internal
data structures for memory profiles and feasibility reports.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field, field_validator

from unisplit.shared.constants import SUPPORTED_SPLIT_IDS


# ── Quantization ────────────────────────────────────────────────────────────

class QuantizationParams(BaseModel):
    """Parameters for symmetric int8 quantization."""
    scale: float = Field(..., description="Quantization scale factor")
    zero_point: int = Field(0, description="Quantization zero point (0 for symmetric)")
    dtype: str = Field("int8", description="Quantized dtype")


# ── Inference Request / Response ────────────────────────────────────────────

class SplitInferenceRequest(BaseModel):
    """Request from edge to cloud for split inference."""
    request_id: str = Field(..., description="Unique request identifier (UUID)")
    split_id: int = Field(..., description="Split point identifier")
    tensor_payload: str = Field(..., description="Base64-encoded tensor bytes")
    shape: list[int] = Field(..., description="Tensor shape, e.g. [32, 78]")
    dtype: str = Field("float32", description="Tensor dtype: 'float32' or 'int8'")
    quantization_params: QuantizationParams | None = Field(
        None, description="Quantization params if dtype is int8"
    )
    model_version: str = Field(..., description="Model/partition version tag")
    trace_metadata: dict[str, Any] | None = Field(
        None, description="Optional trace context"
    )
    edge_timestamp_ms: float = Field(
        default_factory=lambda: time.time() * 1000,
        description="Edge-side send timestamp (ms since epoch)",
    )

    @field_validator("split_id")
    @classmethod
    def validate_split_id(cls, v: int) -> int:
        if v not in SUPPORTED_SPLIT_IDS:
            raise ValueError(
                f"split_id {v} not in supported set {SUPPORTED_SPLIT_IDS}"
            )
        return v

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        if v not in ("float32", "int8"):
            raise ValueError(f"dtype must be 'float32' or 'int8', got '{v}'")
        return v


class InferenceTiming(BaseModel):
    """Timing breakdown for cloud-side inference."""
    deserialize_ms: float = Field(..., description="Payload deserialization time")
    dequantize_ms: float = Field(0.0, description="Dequantization time (0 if float32)")
    inference_ms: float = Field(..., description="Model forward pass time")
    total_ms: float = Field(..., description="Total cloud processing time (τ_t)")


class SplitInferenceResponse(BaseModel):
    """Response from cloud to edge after split inference."""
    request_id: str
    split_id: int
    predicted_class: int = Field(..., description="Predicted class index (0–33)")
    predicted_label: str = Field(..., description="Human-readable class name")
    probabilities: list[float] = Field(..., description="34-element softmax output")
    model_version: str
    timing: InferenceTiming
    trace_metadata: dict[str, Any] | None = None
    status: str = Field("ok", description="'ok' or 'error'")
    error: str | None = None


# ── Full Inference (testing) ────────────────────────────────────────────────

class FullInferenceRequest(BaseModel):
    """Request for full model inference (testing/admin path)."""
    request_id: str
    tensor_payload: str
    shape: list[int]
    dtype: str = "float32"
    model_version: str = "v0.1.0"


class FullInferenceResponse(BaseModel):
    """Response for full model inference."""
    request_id: str
    predicted_class: int
    predicted_label: str
    probabilities: list[float]
    model_version: str
    inference_ms: float
    status: str = "ok"
    error: str | None = None


# ── Health / Readiness ──────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Liveness probe response."""
    status: str = "ok"
    version: str = ""


class ReadinessResponse(BaseModel):
    """Readiness probe response."""
    ready: bool
    loaded_split_ids: list[int] = Field(default_factory=list)
    model_version: str = ""
    backend_type: str = ""


# ── Split Info ──────────────────────────────────────────────────────────────

class SplitPointInfo(BaseModel):
    """Metadata for a single split point."""
    split_id: int
    name: str
    output_shape: list[int]
    payload_float32_bytes: int
    payload_int8_bytes: int
    feasible: bool = True


class SplitInfoResponse(BaseModel):
    """Response listing available split points."""
    split_points: list[SplitPointInfo]
    feasible_split_ids: list[int]
    budget_bytes: int
    overhead_bytes: int


# ── Config ──────────────────────────────────────────────────────────────────

class ConfigResponse(BaseModel):
    """Effective configuration dump."""
    config: dict[str, Any]


# ── Memory Profiling ───────────────────────────────────────────────────────

class LayerMemoryProfile(BaseModel):
    """Memory profile for a single layer."""
    layer_index: int
    layer_name: str
    weight_bytes: int = Field(..., description="Parameter memory in bytes")
    activation_bytes: int = Field(..., description="Output activation size in bytes")
    output_shape: list[int]


class SplitMemoryProfile(BaseModel):
    """Memory profile for a split point (edge runtime memory)."""
    split_id: int
    split_name: str
    cumulative_weight_bytes: int = Field(..., description="W_k: total weight bytes for layers 1..k")
    peak_activation_bytes: int = Field(..., description="A_k: max activation across layers 1..k")
    runtime_overhead_bytes: int = Field(..., description="δ: OS/runtime overhead")
    total_edge_memory_bytes: int = Field(..., description="mem(k) = W_k + A_k + δ")
    # Communication payload (separate from edge runtime memory)
    communication_payload_float32_bytes: int = Field(
        ..., description="Size of h_k(x) in float32 bytes"
    )
    communication_payload_int8_bytes: int = Field(
        ..., description="Size of h_k(x) in int8 bytes"
    )
    output_shape: list[int]
    feasible: bool = Field(..., description="mem(k) ≤ B")


class FeasibilityReport(BaseModel):
    """Complete feasibility report for a given budget."""
    budget_bytes: int
    overhead_bytes: int
    layer_profiles: list[LayerMemoryProfile]
    split_profiles: list[SplitMemoryProfile]
    feasible_split_ids: list[int]
    total_model_parameters: int
    total_model_weight_bytes: int


# ── Model Artifact Metadata ────────────────────────────────────────────────

class ModelArtifactMeta(BaseModel):
    """Metadata stored alongside exported partition files."""
    split_id: int
    partition_type: str = Field(..., description="'edge' or 'cloud'")
    model_version: str
    input_shape: list[int]
    output_shape: list[int]
    parameter_count: int
    export_timestamp: float = Field(default_factory=time.time)
    source_checkpoint: str = ""
