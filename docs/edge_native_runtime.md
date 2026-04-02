# Edge-Native Runtime (T02a, VPS/QEMU Correctness)

This stage adds a practical split-7 edge-native path that is executable on VPS
now and portable to Raspberry Pi later.

## What This Adds

- Python export from `partitions/edge_k7/partition.pt` to C-friendly artifacts
- Plain-C split-7 forward path:
  - Conv1D -> BatchNorm -> ReLU -> Conv1D -> BatchNorm -> ReLU -> GlobalAvgPool
- Symmetric int8 quantization in C (matching Python contract)
- Real host-side HTTP POST to cloud `/infer/split`
- Tests for export correctness, forward parity, quant parity, and cloud roundtrip

## Artifact Format

Export command:

```bash
make export-edge-c
```

Output directory: `edge_native/artifacts/edge_k7_c/`

Files:

- `manifest.json` (schema + split metadata + tensor descriptors + eps)
- Tensor binaries (`*.bin`) as little-endian float32
- Optional deterministic references:
  - `reference_input.bin`
  - `reference_activation_k7.bin`

The C runtime does not parse `.pt` files.

## Build and Validate

```bash
# Build C runtime and test binaries
make c-edge-build

# Export + forward/export checks
make c-edge-forward-verify

# Quantization parity
make c-edge-quant-verify

# End-to-end host-side POST /infer/split
make c-edge-roundtrip
```

## Transport Architecture

`transport.h` defines a narrow transport contract:

- `transport_post_json(const transport_cfg_t*, const char* path, const char* req_json, char** resp_json_out)`

Current implementation uses POSIX sockets over HTTP/1.1 (`Connection: close`).

Later migration path on Pi/Unikraft:

- keep model/export/quantization unchanged
- replace only transport implementation with lwIP/Unikraft network path

## Raspberry Pi Transition Notes

Ready now for Pi transition:

- C artifact format
- C split-7 forward path
- cloud JSON contract compatibility

Still deferred to Pi hardware stage:

- PMU (`lib-pmu`) real counters
- NEON optimization and performance tuning
- lwIP runtime replacement and MQTT ingestion path
- INA219 and board-specific instrumentation
