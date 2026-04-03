# Edge-Native Runtime (Multi-Split, VPS/QEMU Correctness)

This stage extends the original split-7 vertical slice into a reusable
edge-native runtime architecture across all supported split IDs
`{0,3,6,7,8,9}`.

## What This Adds

- Python export from `partitions/edge_k*` to C-friendly artifacts for all splits
- Plain-C multi-split forward path:
  - `k0`: passthrough input
  - `k3`: block1
  - `k6`: block2
  - `k7`: global average pool
  - `k8`: fc1 + relu
  - `k9`: logits
- Symmetric int8 quantization in C (matching Python contract)
- Generic host-side HTTP POST to cloud `/infer/split` with pluggable transport backend
- Backend selection across `posix`, `ukstub`, and `lwip` (stub) transport backends
- C-side LinUCB controller scaffolding with synthetic sanity tests
- Tests for export correctness, multi-split forward parity, quant parity, and cloud roundtrip

## Artifact Format

Export all splits:

```bash
make export-edge-c-all
```

Output root: `edge_native/artifacts/c_splits/`

Per split directory: `edge_native/artifacts/c_splits/edge_k{split_id}/`

- `manifest.json` (schema + split metadata + tensor descriptors + eps)
- Tensor binaries (`*.bin`) as little-endian float32
- Optional deterministic references:
  - `reference_input.bin`
  - `reference_activation.bin`

The C runtime does not parse `.pt` files.

## Build and Validate

```bash
# Build C runtime and test binaries
make c-edge-build

# Export + forward checks for legacy k7 path
make c-edge-forward-verify

# Export + forward checks for all supported splits
make c-edge-forward-verify-all

# Quantization parity
make c-edge-quant-verify

# C controller/LinUCB sanity checks
make c-edge-controller-verify

# Failure-path hardening checks
make c-edge-failure-verify

# End-to-end host-side POST /infer/split (k7 compatibility path)
make c-edge-roundtrip

# End-to-end host-side POST /infer/split (generic multi-split path)
make c-edge-roundtrip-generic

# VPS evidence-producing roundtrip matrix (k3/k7/k8)
make c-edge-roundtrip-vps

# Unikraft/QEMU edge-runtime selftest (generic runtime linked)
make uk-edge-validate

# Pi readiness helpers
make pi-readiness-manifest
make pi-readiness-check
make pi-boot-payload
```

## Transport Architecture

`transport_backend.h` defines a backend-pluggable transport client:

- `transport_client_t` with function pointers (`post_json`, `destroy`)
- `transport_posix_create(...)` for current host-side implementation
- `transport_ukstub_create(...)` for deterministic Unikraft-oriented stub path
- `transport_lwip_create(...)` currently as explicit not-implemented stub
- `transport_create_by_name(...)` for backend selection (`posix` / `ukstub` / `lwip`)
- `transport_client_post_json(...)` for cloud client use

Current implementation uses POSIX sockets over HTTP/1.1.

Later migration path on Pi/Unikraft:

- keep model/export/runtime/quantization unchanged
- replace only `lwip` backend implementation with real Unikraft/lwIP path

## Raspberry Pi Transition Notes

Ready now for Pi transition:

- C artifact format for all supported splits
- C multi-split forward path and split dispatch
- cloud JSON contract compatibility
- transport backend abstraction
- C controller scaffolding hooks

Still deferred to Pi hardware stage:

- PMU (`lib-pmu`) real counters
- NEON optimization and performance tuning
- lwIP transport backend replacement and MQTT ingestion path
- INA219 and board-specific instrumentation
