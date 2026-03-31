"""FastAPI application with lifespan management."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from unisplit.cloud.backends import create_backend
from unisplit.cloud.inference import CloudInferenceEngine
from unisplit.cloud.model_registry import ModelRegistry
from unisplit.cloud.routes import router
from unisplit.shared.config import load_config
from unisplit.shared.logging import setup_logging

# Module-level state (populated during lifespan)
_state: dict = {}


def get_state() -> dict:
    """Get application state."""
    return _state


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to generate/propagate X-Request-ID."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def create_app(config_path: str | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config_path: Path to config YAML. If None, uses defaults + env vars.

    Returns:
        Configured FastAPI app.
    """
    config = load_config(config_path)
    cc = config.cloud

    logger = setup_logging(level=cc.log_level, fmt="json", name="unisplit.cloud")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan: load model partitions on startup."""
        logger.info(f"Starting cloud service with backend={cc.backend_type}")

        # Discover partitions
        registry = ModelRegistry(cc.partition_dir)
        available = registry.discover()

        # Create backend and load partitions
        backend = create_backend(cc.backend_type)
        if available:
            backend.load_partitions(cc.partition_dir, available)
            logger.info(f"Loaded {len(available)} partitions: {available}")
        else:
            logger.warning("No partitions found — service will start but /ready will be false")

        # Create inference engine
        engine = CloudInferenceEngine(backend)

        # Store in app state
        _state["config"] = config
        _state["registry"] = registry
        _state["backend"] = backend
        _state["engine"] = engine
        _state["logger"] = logger

        yield

        logger.info("Cloud service shutting down")

    app = FastAPI(
        title="UniSplit Cloud Inference Service",
        description="Cloud-side split inference for IoT anomaly detection",
        version=cc.model_version,
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(router)

    return app
