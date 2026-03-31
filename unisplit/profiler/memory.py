"""Per-layer memory profiling for the IoTCNN model.

Implements the memory model from paper §3.2:
    mem(k) = W_k (cumulative weights) + A_k (peak activations) + δ (overhead)

This profiler traces a forward pass to measure actual tensor sizes,
producing per-layer weight and activation byte counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from unisplit.model.cnn import IoTCNN
from unisplit.shared.constants import NUM_FEATURES
from unisplit.shared.schemas import LayerMemoryProfile


@dataclass
class LayerProfile:
    """Internal layer profiling result."""
    index: int
    name: str
    weight_bytes: int
    activation_bytes: int
    output_shape: list[int]


class ModelMemoryProfiler:
    """Profiles the IoTCNN model for per-layer memory usage.

    Measures weight bytes and activation bytes for each named layer group,
    enabling the feasibility calculator to compute mem(k) for each split point.
    """

    def __init__(self, model: IoTCNN | None = None, num_features: int = NUM_FEATURES):
        self.num_features = num_features
        if model is None:
            model = IoTCNN(num_features=num_features)
        self.model = model
        self.model.eval()
        self._layer_profiles: list[LayerProfile] | None = None

    def profile(self) -> list[LayerProfile]:
        """Run a traced forward pass and measure per-layer memory.

        Returns:
            List of LayerProfile for each layer group.
        """
        if self._layer_profiles is not None:
            return self._layer_profiles

        profiles = []
        x = torch.randn(1, 1, self.num_features)

        # Layer group mapping to split boundaries:
        # block1 contains layers 1-3 → split after = 3
        # block2 contains layers 4-6 → split after = 6
        # pool is layer 7 → split after = 7
        # fc1 is layer 8 → split after = 8
        # fc2 is layer 9 → split after = 9

        layer_groups = [
            (1, "block1", self.model.block1),
            (4, "block2", self.model.block2),
            (7, "pool", self.model.pool),
            (8, "fc1", self.model.fc1),
            (9, "fc2", self.model.fc2),
        ]

        with torch.no_grad():
            for start_idx, name, module in layer_groups:
                # Count weight bytes
                weight_bytes = 0
                for p in module.parameters():
                    weight_bytes += p.numel() * p.element_size()

                # Run layer and measure output activation size
                x = module(x)
                if name == "pool":
                    # Pool outputs (batch, 64, 1) → squeeze will happen later
                    activation_shape = list(x.shape[1:])  # [64, 1]
                    activation_bytes = x[0].numel() * x.element_size()
                    # But for memory purposes, we track the un-squeezed shape
                else:
                    activation_shape = list(x.shape[1:])
                    activation_bytes = x[0].numel() * x.element_size()

                profiles.append(LayerProfile(
                    index=start_idx,
                    name=name,
                    weight_bytes=weight_bytes,
                    activation_bytes=activation_bytes,
                    output_shape=activation_shape,
                ))

                # Squeeze after pool for correct downstream shapes
                if name == "pool":
                    x = x.squeeze(-1)

        self._layer_profiles = profiles
        return profiles

    def get_layer_memory_profiles(self) -> list[LayerMemoryProfile]:
        """Get layer profiles as Pydantic schema objects."""
        profiles = self.profile()
        return [
            LayerMemoryProfile(
                layer_index=p.index,
                layer_name=p.name,
                weight_bytes=p.weight_bytes,
                activation_bytes=p.activation_bytes,
                output_shape=p.output_shape,
            )
            for p in profiles
        ]

    def get_cumulative_weight_bytes(self, split_id: int) -> int:
        """Compute W_k: cumulative weight bytes for layers 1..k.

        W_k = Σ weight_bytes for all layer groups up through split_id.
        """
        profiles = self.profile()
        total = 0
        # Map split_id to which layer groups are included
        split_to_groups = {
            0: [],
            3: ["block1"],
            6: ["block1", "block2"],
            7: ["block1", "block2", "pool"],
            8: ["block1", "block2", "pool", "fc1"],
            9: ["block1", "block2", "pool", "fc1", "fc2"],
        }
        included = split_to_groups.get(split_id, [])
        for p in profiles:
            if p.name in included:
                total += p.weight_bytes
        return total

    def get_peak_activation_bytes(self, split_id: int) -> int:
        """Compute A_k: peak activation memory across layers 1..k.

        A_k = max(activation_bytes) for all layer groups up through split_id.
        At runtime, the edge needs to hold at least the largest intermediate
        activation buffer.
        """
        profiles = self.profile()
        split_to_groups = {
            0: [],
            3: ["block1"],
            6: ["block1", "block2"],
            7: ["block1", "block2", "pool"],
            8: ["block1", "block2", "pool", "fc1"],
            9: ["block1", "block2", "pool", "fc1", "fc2"],
        }
        included = split_to_groups.get(split_id, [])
        if not included:
            return 0

        # Also include input activation
        input_bytes = self.num_features * 4  # float32 input
        activations = [input_bytes]
        for p in profiles:
            if p.name in included:
                activations.append(p.activation_bytes)
        return max(activations)

    def get_communication_payload_bytes(self, split_id: int, dtype: str = "float32") -> int:
        """Get the size of the activation tensor h_k(x) that gets transmitted.

        This is SEPARATE from edge runtime memory — it's the payload size.
        """
        from unisplit.model.registry import get_split_info
        info = get_split_info(split_id)
        num_floats = info.payload_floats
        if dtype == "float32":
            return num_floats * 4
        elif dtype == "int8":
            return num_floats * 1
        else:
            raise ValueError(f"Unknown dtype: {dtype}")

    def get_total_model_parameters(self) -> int:
        """Total model parameters."""
        return self.model.count_parameters()

    def get_total_model_weight_bytes(self) -> int:
        """Total model weight bytes (all layers, float32)."""
        return sum(p.numel() * p.element_size() for p in self.model.parameters())
