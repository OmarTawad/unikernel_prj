# Training Guide

## Prerequisites

1. Dataset preprocessed (see [dataset.md](dataset.md))
2. Environment installed (`make install`)

## Training Commands

```bash
# Dry-run (validates architecture, no real data needed)
make dry-run

# Full training (requires preprocessed data)
make train

# Validate on validation set
make validate

# Test on test set
make test-model

# Export partitions from best checkpoint
make export-partitions
```

## Configuration

Edit `configs/training.yaml`:

```yaml
training:
  batch_size: 256
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
  seed: 42
  device: "cpu"
  use_class_weights: true
  scheduler_patience: 5
  scheduler_factor: 0.5
```

## Training Pipeline

1. **Data loading**: `CICIoT2023Dataset` reads preprocessed .npy arrays
2. **Normalization**: Min-max normalization using training set statistics
3. **Loss**: CrossEntropyLoss with optional per-class weights
4. **Optimizer**: Adam with ReduceLROnPlateau scheduler
5. **Checkpointing**: Best (by val F1), latest, and periodic
6. **Metrics**: JSONL log at `checkpoints/metrics.jsonl`

## Class Imbalance

CIC-IoT2023 has significant class imbalance. The training code supports:
- Inverse-frequency class weights (enabled by default)
- Stratified train/val/test splits

## After Training

```bash
# Profile memory for partition feasibility
make profile-memory

# Export edge/cloud partitions for all 6 split points
make export-partitions

# Start cloud service
make run-cloud
```
