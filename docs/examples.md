# Examples

## 1. Profile Model Memory

```bash
# Default 24 MB budget
python scripts/profile_memory.py --budget 24M --output profiles/default.json

# Tight budget experiment
python scripts/profile_memory.py --budget 8M --output profiles/tight_8mb.json
```

## 2. Single Sample Inference

```bash
# Start cloud service first
make run-cloud &

# Run single sample (synthetic) through split_id=7
python -m unisplit.edge.cli single --split-id 7
```

## 3. Run an Experiment

```bash
# Ensure cloud is running and data is preprocessed
python -m unisplit.experiments.replay \
    --config configs/experiments/static_kmin.yaml \
    --max-samples 1000
```

## 4. Compare Policies

```python
from unisplit.policies import create_policy
import numpy as np

feasible = [0, 3, 6, 7, 8, 9]
context = np.array([20.0, 0.5, 1.5])  # [rtt_ms, cpu_frac, entropy]

# Static policies
for name in ["static_kmin", "static_kmax"]:
    policy = create_policy(name, feasible)
    print(f"{name}: split_id={policy.select(context)}")

# LinUCB
policy = create_policy("linucb", feasible, alpha=1.0)
print(f"linucb: split_id={policy.select(context)}")
```

## 5. Export Partitions Programmatically

```python
import torch
from unisplit.model.cnn import IoTCNN
from unisplit.model.partition import export_all_partitions

model = IoTCNN()
model.load_state_dict(torch.load("checkpoints/best.pt")["model_state_dict"])
export_all_partitions(model, "partitions/", model_version="v0.1.0")
```

## 6. Network Simulation Scripts

```bash
# Generate tc/netem scripts
python -c "
from unisplit.experiments.netem import generate_netem_script
generate_netem_script(delay_ms=50, jitter_ms=10, loss_pct=1.0, 
                      output_path='scripts/netem_setup.sh')
"

# Run (requires sudo)
sudo bash scripts/netem_setup.sh
```
