"""API route handlers for the cloud inference service.

Endpoints:
    GET  /health          — Liveness probe
    GET  /ready           — Readiness (partitions loaded?)
    GET  /splits          — Available split point info
    GET  /config/effective — Current config dump
    POST /infer/split     — Split inference
    POST /infer/full      — Full model inference (testing)
"""

from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from unisplit.model.registry import SPLIT_REGISTRY
from unisplit.shared.constants import CLASS_NAMES, DEFAULT_MODEL_VERSION
from unisplit.shared.schemas import (
    ConfigResponse,
    FullInferenceRequest,
    FullInferenceResponse,
    HealthResponse,
    ReadinessResponse,
    SplitInferenceRequest,
    SplitInferenceResponse,
    SplitInfoResponse,
    SplitPointInfo,
)
from unisplit.shared.serialization import decode_payload

router = APIRouter()


def get_state() -> dict:
    """Deferred import to avoid circular dependency with app.py."""
    from unisplit.cloud.app import _state
    return _state


@router.get("/health", response_model=HealthResponse)
async def health():
    """Liveness probe."""
    return HealthResponse(status="ok", version=DEFAULT_MODEL_VERSION)


@router.get("/ready", response_model=ReadinessResponse)
async def ready():
    """Readiness probe — checks if model partitions are loaded."""
    state = get_state()
    backend = state.get("backend")
    config = state.get("config")

    if backend is None:
        return ReadinessResponse(ready=False)

    return ReadinessResponse(
        ready=backend.is_ready(),
        loaded_split_ids=backend.loaded_split_ids(),
        model_version=config.cloud.model_version if config else "",
        backend_type=config.cloud.backend_type if config else "",
    )


@router.get("/splits", response_model=SplitInfoResponse)
async def splits():
    """Available split point information with payload sizes."""
    state = get_state()
    config = state.get("config")
    backend = state.get("backend")
    loaded = backend.loaded_split_ids() if backend else []

    split_infos = []
    for split_id, entry in SPLIT_REGISTRY.items():
        split_infos.append(SplitPointInfo(
            split_id=split_id,
            name=entry.name,
            output_shape=list(entry.output_shape),
            payload_float32_bytes=entry.payload_float32_bytes,
            payload_int8_bytes=entry.payload_int8_bytes,
            feasible=split_id in loaded,
        ))

    budget = config.memory_budget.budget_bytes if config else 0
    overhead = config.memory_budget.overhead_bytes if config else 0

    return SplitInfoResponse(
        split_points=split_infos,
        feasible_split_ids=loaded,
        budget_bytes=budget,
        overhead_bytes=overhead,
    )


@router.get("/config/effective", response_model=ConfigResponse)
async def effective_config():
    """Dump the effective configuration."""
    state = get_state()
    config = state.get("config")
    if config is None:
        return ConfigResponse(config={})
    return ConfigResponse(config=config.model_dump())


@router.post("/infer/split", response_model=SplitInferenceResponse)
async def infer_split(request: SplitInferenceRequest):
    """Split inference endpoint.

    Receives an intermediate activation h_k(x) from the edge,
    runs the cloud partition g_k, and returns the prediction with timing.
    """
    state = get_state()
    engine = state.get("engine")

    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    backend = state.get("backend")
    if request.split_id not in backend.loaded_split_ids():
        raise HTTPException(
            status_code=400,
            detail=f"No partition loaded for split_id={request.split_id}. "
                   f"Available: {backend.loaded_split_ids()}"
        )

    try:
        logits, timing = engine.run_split_inference(request)

        # Softmax for probabilities
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities = (exp_logits / exp_logits.sum(axis=-1, keepdims=True)).flatten().tolist()

        predicted_class = int(np.argmax(logits))
        predicted_label = CLASS_NAMES[predicted_class] if predicted_class < len(CLASS_NAMES) else f"class_{predicted_class}"

        config = state.get("config")
        model_version = config.cloud.model_version if config else DEFAULT_MODEL_VERSION

        return SplitInferenceResponse(
            request_id=request.request_id,
            split_id=request.split_id,
            predicted_class=predicted_class,
            predicted_label=predicted_label,
            probabilities=probabilities,
            model_version=model_version,
            timing=timing,
            trace_metadata=request.trace_metadata,
            status="ok",
        )

    except Exception as e:
        return SplitInferenceResponse(
            request_id=request.request_id,
            split_id=request.split_id,
            predicted_class=-1,
            predicted_label="error",
            probabilities=[],
            model_version=DEFAULT_MODEL_VERSION,
            timing={"deserialize_ms": 0, "dequantize_ms": 0, "inference_ms": 0, "total_ms": 0},
            status="error",
            error=str(e),
        )


@router.post("/infer/full", response_model=FullInferenceResponse)
async def infer_full(request: FullInferenceRequest):
    """Full model inference (testing/admin path)."""
    state = get_state()
    engine = state.get("engine")

    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    try:
        input_data = decode_payload(request.tensor_payload, request.shape, request.dtype)
        logits, inference_ms = engine.run_full_inference(input_data)

        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities = (exp_logits / exp_logits.sum(axis=-1, keepdims=True)).flatten().tolist()

        predicted_class = int(np.argmax(logits))
        predicted_label = CLASS_NAMES[predicted_class] if predicted_class < len(CLASS_NAMES) else f"class_{predicted_class}"

        return FullInferenceResponse(
            request_id=request.request_id,
            predicted_class=predicted_class,
            predicted_label=predicted_label,
            probabilities=probabilities,
            model_version=request.model_version,
            inference_ms=round(inference_ms, 3),
        )

    except Exception as e:
        return FullInferenceResponse(
            request_id=request.request_id,
            predicted_class=-1,
            predicted_label="error",
            probabilities=[],
            model_version=request.model_version,
            inference_ms=0.0,
            status="error",
            error=str(e),
        )
