"""Tests for model partition consistency.

Core invariant: forward_from(forward_to(x, k), k) ≈ forward(x) for all k.
"""

import tempfile
from pathlib import Path

import torch

from unisplit.model.cnn import IoTCNN
from unisplit.model.partition import (
    export_all_partitions,
    load_cloud_partition,
    load_edge_partition,
)
from unisplit.shared.constants import SUPPORTED_SPLIT_IDS


class TestPartition:
    def test_split_consistency(self, model, sample_input):
        """h_k + g_k == f(x) for all supported split IDs."""
        with torch.no_grad():
            full_output = model(sample_input)

            for split_id in [0, 3, 6, 7, 8]:
                h = model.forward_to(sample_input, split_id)
                y = model.forward_from(h, split_id)
                diff = (full_output - y).abs().max().item()
                assert diff < 1e-4, (
                    f"Split {split_id}: max diff = {diff}"
                )

    def test_local_only_consistency(self, model, sample_input):
        """split_id=9 should return same as full forward."""
        with torch.no_grad():
            full = model(sample_input)
            local = model.forward_to(sample_input, 9)
            diff = (full - local).abs().max().item()
            assert diff < 1e-6

    def test_export_and_reload(self, model, sample_input):
        """Export partitions, reload, verify consistency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_all_partitions(model, tmpdir, model_version="test")

            for split_id in SUPPORTED_SPLIT_IDS:
                edge_dir = Path(tmpdir) / f"edge_k{split_id}"
                cloud_dir = Path(tmpdir) / f"cloud_k{split_id}"
                assert edge_dir.exists(), f"Edge partition missing for k={split_id}"
                assert cloud_dir.exists(), f"Cloud partition missing for k={split_id}"
                assert (edge_dir / "metadata.json").exists()
                assert (cloud_dir / "metadata.json").exists()

    def test_export_reload_inference(self, model, sample_input):
        """Full round-trip: export → load → infer → match original."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_all_partitions(model, tmpdir)

            with torch.no_grad():
                full_output = model(sample_input)

                for split_id in [3, 7]:
                    edge_model = load_edge_partition(tmpdir, split_id)
                    cloud_model = load_cloud_partition(tmpdir, split_id)

                    h = edge_model.forward_to(sample_input, split_id)
                    y = cloud_model.forward_from(h, split_id)

                    diff = (full_output - y).abs().max().item()
                    assert diff < 1e-4, (
                        f"Reload split {split_id}: max diff = {diff}"
                    )
