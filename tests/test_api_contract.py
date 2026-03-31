"""Tests for the cloud API contract using FastAPI TestClient."""

import uuid
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from unisplit.model.cnn import IoTCNN
from unisplit.model.partition import export_all_partitions
from unisplit.shared.serialization import encode_payload


@pytest.fixture
def cloud_app():
    """Create a test cloud app with exported partitions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and export model
        model = IoTCNN()
        model.eval()
        export_all_partitions(model, f"{tmpdir}/partitions")

        # Create app with temp config
        import os
        os.environ["UNISPLIT_CONFIG_PATH"] = ""

        from unisplit.cloud.app import create_app
        app = create_app(None)

        # Manually set up state
        from unisplit.cloud.backends import PyTorchCPUBackend
        from unisplit.cloud.inference import CloudInferenceEngine
        from unisplit.cloud.model_registry import ModelRegistry
        from unisplit.shared.config import UniSplitConfig

        config = UniSplitConfig()
        registry = ModelRegistry(f"{tmpdir}/partitions")
        available = registry.discover()

        backend = PyTorchCPUBackend()
        backend.load_partitions(f"{tmpdir}/partitions", available)
        engine = CloudInferenceEngine(backend)

        from unisplit.cloud.app import _state
        _state["config"] = config
        _state["registry"] = registry
        _state["backend"] = backend
        _state["engine"] = engine

        yield TestClient(app)


class TestCloudAPI:
    def test_health(self, cloud_app):
        resp = cloud_app.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ready(self, cloud_app):
        resp = cloud_app.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is True
        assert len(data["loaded_split_ids"]) > 0

    def test_splits(self, cloud_app):
        resp = cloud_app.get("/splits")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["split_points"]) == 6

    def test_config(self, cloud_app):
        resp = cloud_app.get("/config/effective")
        assert resp.status_code == 200
        assert "config" in resp.json()

    def test_infer_split(self, cloud_app):
        # Create a valid activation for split_id=7 (shape 64)
        activation = np.random.randn(64).astype(np.float32)
        payload = encode_payload(activation)

        resp = cloud_app.post("/infer/split", json={
            "request_id": str(uuid.uuid4()),
            "split_id": 7,
            "tensor_payload": payload,
            "shape": [64],
            "dtype": "float32",
            "model_version": "v0.1.0",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert 0 <= data["predicted_class"] < 34
        assert len(data["probabilities"]) == 34
        assert data["timing"]["total_ms"] > 0

    def test_infer_split_invalid_id(self, cloud_app):
        resp = cloud_app.post("/infer/split", json={
            "request_id": "test",
            "split_id": 5,
            "tensor_payload": "AAAA",
            "shape": [10],
            "dtype": "float32",
            "model_version": "v0.1.0",
        })
        assert resp.status_code == 422  # Validation error
