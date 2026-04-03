# UniSplit

**Memory-Constrained Adaptive Split Inference for Edge-Cloud IoT Anomaly Detection**

A research prototype implementing edge-cloud split inference with memory-constrained feasibility for IoT anomaly detection using the CIC-IoT2023 dataset.

## Quick Start

```bash
# 1. Install
make install

# 2. Run tests
make test

# 3. Dry-run training (synthetic data, validates architecture)
make dry-run

# 4. Profile model memory
make profile-memory

# 5. Run smoke test
make smoke-test
```

## Full Pipeline

```bash
# Dataset (manual download required — see docs/dataset.md)
bash scripts/download_dataset.sh
make preprocess-data

# Training
make train

# Export model partitions
make export-partitions

# Run services
make run-cloud    # Terminal 1
make run-edge     # Terminal 2

# Docker deployment
make docker-build
make docker-up
make docker-logs
make docker-down
```

## Architecture

```
Edge (Docker/Unikraft)           Cloud (Docker/VPS)
┌─────────────────┐             ┌──────────────────┐
│  IoT Input x_t  │             │  Cloud Partition  │
│        ↓        │   h_k(x)   │   g_k(h) → ŷ     │
│  Edge Partition  │ ─────────→ │                   │
│  h_k = f_1:k(x) │   (int8)   │   Returns:        │
│        ↓        │             │   - predicted ŷ   │
│  Bandit Policy   │ ←───────── │   - latency τ_t   │
│  (selects k_t)   │   ŷ, τ_t  │                   │
└─────────────────┘             └──────────────────┘
     mem(k) ≤ B                     PyTorch CPU
```

**Key innovation**: The edge memory budget B is a hard constraint. The feasible split set K(B) = {k : mem(k) ≤ B} is computed offline. A LinUCB bandit adaptively selects split points using estimated RTT, CPU utilization, and uncertainty.

## Model

Compact 1-D CNN (paper §5):
- Conv1D(1→32, k=3) → BN → ReLU → Conv1D(32→64, k=3) → BN → ReLU → GlobalAvgPool → FC(64→128) → ReLU → FC(128→34)
- Input: 80-feature network flow vector
- Output: 34 classes (33 attacks + benign)
- Parameters: ~98K

## Supported Split Points

| ID | Name | Edge Runs | Activation Shape | Payload (f32) |
|----|------|-----------|-----------------|---------------|
| 0 | input | nothing | (1, 80) | 320 B |
| 3 | after_block1 | Conv+BN+ReLU | (32, 78) | 9.8 KB |
| 6 | after_block2 | 2×Conv+BN+ReLU | (64, 76) | 19.5 KB |
| 7 | after_pool | +GlobalAvgPool | (64,) | 256 B |
| 8 | after_fc1 | +FC(128) | (128,) | 512 B |
| 9 | local_only | full model | (34,) | 0 B |

## Project Structure

```
unisplit/
├── shared/      # Contracts, config, serialization, quantization
├── model/       # CNN definition, split registry, partitioning
├── profiler/    # Memory-feasibility pipeline (core component)
├── training/    # Dataset, preprocessing, training loop, CLI
├── cloud/       # FastAPI inference service
├── edge/        # Edge simulator, context extraction, ingestion
├── policies/    # Static + LinUCB split policies
└── experiments/ # Experiment orchestration, metrics
```

## Documentation

- [Architecture](docs/architecture.md) — System design and components
- [Protocol](docs/protocol.md) — Edge-cloud API contract
- [Training](docs/training.md) — Training pipeline details
- [Dataset](docs/dataset.md) — CIC-IoT2023 setup
- [Deployment](docs/deployment.md) — Docker deployment guide
- [Examples](docs/examples.md) — Usage examples
- [Edge Native Runtime](docs/edge_native_runtime.md) — C runtime + transport architecture
- [Pre-Pi Checklist](docs/pre_pi_validation_checklist.md) — Pre-hardware validation baseline
- [Raspberry Pi Handoff](docs/raspberry_pi_handoff.md) — Pi package split + day-1 checklist

## Requirements

- Python 3.11+
- PyTorch, FastAPI, Pydantic v2, NumPy, scikit-learn
- Docker + Docker Compose (for containerized deployment)
- No GPU required (CPU-first design)

## License

MIT
