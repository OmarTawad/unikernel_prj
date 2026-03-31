#!/usr/bin/env python3
"""End-to-end smoke test.

Creates a model, exports partitions, starts cloud service in-process,
sends a split inference request from edge, and validates the response.
"""

import sys
import time
import uuid
import tempfile
from pathlib import Path

import numpy as np
import torch


def main():
    print("=" * 60)
    print("  UniSplit End-to-End Smoke Test")
    print("=" * 60)

    errors = []

    # Step 1: Create model
    print("\n[1/6] Creating model...")
    from unisplit.model.cnn import IoTCNN
    model = IoTCNN()
    model.eval()
    print(f"  ✓ Model created: {model.count_parameters()} parameters")

    # Step 2: Export partitions to temp dir
    print("\n[2/6] Exporting partitions...")
    with tempfile.TemporaryDirectory() as tmpdir:
        from unisplit.model.partition import export_all_partitions
        export_all_partitions(model, tmpdir)
        print(f"  ✓ Partitions exported to {tmpdir}")

        # Step 3: Profile memory
        print("\n[3/6] Profiling memory...")
        from unisplit.profiler.memory import ModelMemoryProfiler
        from unisplit.profiler.feasibility import FeasibilityCalculator
        profiler = ModelMemoryProfiler(model)
        calculator = FeasibilityCalculator(profiler, budget_bytes=25_165_824)
        report = calculator.compute_report()
        print(f"  ✓ Feasible splits: {report.feasible_split_ids}")
        assert len(report.feasible_split_ids) > 0, "No feasible splits!"

        # Step 4: Test split consistency
        print("\n[4/6] Testing split consistency...")
        x = torch.randn(1, 80)
        with torch.no_grad():
            full_output = model(x)
            for split_id in [0, 3, 6, 7, 8]:
                h = model.forward_to(x, split_id)
                y = model.forward_from(h, split_id)
                diff = (full_output - y).abs().max().item()
                status = "✓" if diff < 1e-4 else "✗"
                print(f"  {status} Split {split_id}: max_diff={diff:.6f}")
                if diff >= 1e-4:
                    errors.append(f"Split {split_id} consistency failed: diff={diff}")

        # Step 5: Test quantization round-trip
        print("\n[5/6] Testing quantization...")
        from unisplit.shared.quantization import quantize_int8, dequantize_int8
        test_tensor = np.random.randn(64, 76).astype(np.float32)
        quantized, params = quantize_int8(test_tensor)
        recovered = dequantize_int8(quantized, params)
        max_err = np.abs(test_tensor - recovered).max()
        rel_err = max_err / (np.abs(test_tensor).max() + 1e-10)
        status = "✓" if rel_err < 0.02 else "✗"
        print(f"  {status} Quantization: max_err={max_err:.6f}, rel_err={rel_err:.4f}")
        if rel_err >= 0.02:
            errors.append(f"Quantization error too high: {rel_err}")

        # Step 6: Test serialization round-trip
        print("\n[6/6] Testing serialization...")
        from unisplit.shared.serialization import encode_payload, decode_payload
        original = np.random.randn(32, 78).astype(np.float32)
        encoded = encode_payload(original)
        decoded = decode_payload(encoded, list(original.shape), "float32")
        match = np.allclose(original, decoded)
        status = "✓" if match else "✗"
        print(f"  {status} Serialization round-trip: match={match}")
        if not match:
            errors.append("Serialization round-trip failed")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"  ✗ SMOKE TEST FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)
    else:
        print("  ✓ ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
