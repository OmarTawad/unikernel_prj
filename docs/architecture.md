# Architecture

## Overview

UniSplit implements memory-constrained adaptive split inference for IoT anomaly detection. The system splits a compact 1-D CNN between an edge device and a cloud server, with the split point selected adaptively by a contextual bandit policy.

## Current Stage: VPS / Docker

The current implementation runs entirely on a single VPS using Docker containers. The edge device is simulated as a Docker container with memory constraints. The cloud runs as a FastAPI service.

**Required now**: CPU-only, Docker-first
**Future-ready**: GPU backends, Raspberry Pi, Unikraft runtime

## Components

### 1. Model (`unisplit/model/`)
- 1-D CNN with exact paper architecture
- 6 supported split points: {0, 3, 6, 7, 8, 9}
- `forward_to(x, k)` — edge partition
- `forward_from(h, k)` — cloud partition
- Partition export/load as PyTorch state_dicts

### 2. Memory-Feasibility Pipeline (`unisplit/profiler/`)
- **Core architectural component** implementing paper §3.2
- Per-layer profiling: weight bytes, activation bytes
- `mem(k) = W_k + A_k + δ` computation
- Feasible split set K(B) generation
- Profile persistence as JSON artifacts
- Separates edge runtime memory from communication payload

### 3. Cloud Service (`unisplit/cloud/`)
- FastAPI with 6 endpoints
- `PyTorchCPUBackend` (default and only working backend)
- Timing instrumentation at each stage
- Returns actual observed latency τ_t

### 4. Edge Simulator (`unisplit/edge/`)
- Edge partition inference on CPU
- Context vector extraction:
  - τ̂_t: EWMA of observed RTTs (pre-decision)
  - u_t: CPU utilization
  - Ĥ_t: Softmax entropy at k_min
- `ReplayFileSource` for dataset replay
- httpx client for cloud communication
- int8 quantization before send

### 5. Policies (`unisplit/policies/`)
- Static: kmin, kmax, fixed, local-only
- LinUCB: full contextual bandit with Algorithm 1

### 6. Training (`unisplit/training/`)
- CIC-IoT2023 preprocessing pipeline
- PyTorch Dataset with normalization
- Training loop with class weighting, checkpointing
- CLI: train, validate, test, dry-run

## Data Flow

```
1. IngestionSource provides (features, label)
2. ContextExtractor computes c_t = [τ̂_t, u_t, Ĥ_t]
3. SplitPolicy selects k_t from K(B) using c_t
4. EdgeRunner runs forward_to(x, k_t) → h_k(x)
5. quantize_int8(h_k) → compressed payload
6. CloudClient sends to POST /infer/split
7. CloudInferenceEngine: deserialize → dequantize → forward_from → logits
8. Response: {predicted_class, timing.total_ms (= τ_t)}
9. Edge updates RTT estimate with observed round-trip
10. Policy updates with reward r_t = 1[ŷ==y] - λτ_t
```

## Future Extensibility

| Component | Current | Future |
|-----------|---------|--------|
| Edge runtime | Docker container | Unikraft unikernel |
| Edge hardware | CPU (VPS) | Raspberry Pi ARM |
| Cloud backend | PyTorch CPU | ONNX Runtime GPU |
| Ingestion | File replay | MQTT broker |
| Network | localhost/Docker net | Wi-Fi + tc/netem |
