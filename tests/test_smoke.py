"""End-to-end smoke test (runs without external services)."""

import tempfile

import numpy as np
import torch

from unisplit.model.cnn import IoTCNN
from unisplit.model.partition import export_all_partitions
from unisplit.profiler.feasibility import FeasibilityCalculator
from unisplit.profiler.memory import ModelMemoryProfiler
from unisplit.profiler.profile_store import load_profile, save_profile
from unisplit.shared.quantization import dequantize_int8, quantize_int8
from unisplit.shared.serialization import decode_payload, encode_payload


class TestSmoke:
    def test_full_pipeline(self):
        """Model → profile → export → quantize → serialize → deserialize → infer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create model
            model = IoTCNN()
            model.eval()

            # 2. Profile memory
            profiler = ModelMemoryProfiler(model)
            calc = FeasibilityCalculator(profiler, budget_bytes=25_165_824)
            report = calc.compute_report()
            assert len(report.feasible_split_ids) > 0

            # 3. Save/load profile
            save_profile(report, f"{tmpdir}/profile.json")
            loaded_report = load_profile(f"{tmpdir}/profile.json")
            assert loaded_report.feasible_split_ids == report.feasible_split_ids

            # 4. Export partitions
            export_all_partitions(model, f"{tmpdir}/partitions")

            # 5. Run split inference pipeline
            x = torch.randn(1, 80)
            split_id = 7

            with torch.no_grad():
                # Edge: forward_to
                h = model.forward_to(x, split_id)
                h_np = h.numpy()

                # Edge: quantize
                quantized, params = quantize_int8(h_np)

                # Edge: serialize
                payload = encode_payload(quantized)

                # Cloud: deserialize
                recovered = decode_payload(payload, list(quantized.shape), "int8")

                # Cloud: dequantize
                dequantized = dequantize_int8(recovered, params)

                # Cloud: forward_from
                logits = model.forward_from(
                    torch.from_numpy(dequantized), split_id
                )

                # Verify output
                assert logits.shape == (1, 34)
                predicted = logits.argmax(dim=1).item()
                assert 0 <= predicted < 34
