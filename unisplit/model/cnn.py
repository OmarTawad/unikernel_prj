"""1-D CNN for IoT anomaly detection — exact paper architecture.

Architecture (paper §5):
    Conv1D(1→32, k=3) → BN(32) → ReLU →
    Conv1D(32→64, k=3) → BN(64) → ReLU →
    GlobalAvgPool1d →
    FC(64→128) → ReLU →
    FC(128→34)

Input:  80-feature flow vector, reshaped to (batch, 1, 80)
Output: 34-class logits
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from unisplit.shared.constants import NUM_CLASSES, NUM_FEATURES, SUPPORTED_SPLIT_IDS


class IoTCNN(nn.Module):
    """Compact 1-D CNN for IoT anomaly detection.

    The model is structured in named blocks to enable clean splitting
    at the supported split points {0, 3, 6, 7, 8, 9}.
    """

    def __init__(
        self,
        num_features: int = NUM_FEATURES,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        # Block 1: Conv1D(1→32, k=3) + BN + ReLU  (layers 1–3)
        self.block1 = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv1d(1, 32, kernel_size=3, padding=0)),
                ("bn1", nn.BatchNorm1d(32)),
                ("relu1", nn.ReLU(inplace=True)),
            ])
        )

        # Block 2: Conv1D(32→64, k=3) + BN + ReLU  (layers 4–6)
        self.block2 = nn.Sequential(
            OrderedDict([
                ("conv2", nn.Conv1d(32, 64, kernel_size=3, padding=0)),
                ("bn2", nn.BatchNorm1d(64)),
                ("relu2", nn.ReLU(inplace=True)),
            ])
        )

        # Pooling: GlobalAvgPool  (layer 7)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # FC head: FC(64→128) + ReLU + FC(128→34)  (layers 8–9)
        self.fc1 = nn.Sequential(
            OrderedDict([
                ("linear1", nn.Linear(64, 128)),
                ("relu3", nn.ReLU(inplace=True)),
            ])
        )
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass.

        Args:
            x: Input tensor of shape (batch, num_features) or (batch, 1, num_features).

        Returns:
            Logits of shape (batch, num_classes).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, 80)

        x = self.block1(x)    # (batch, 32, 78)
        x = self.block2(x)    # (batch, 64, 76)
        x = self.pool(x)      # (batch, 64, 1)
        x = x.squeeze(-1)     # (batch, 64)
        x = self.fc1(x)       # (batch, 128)
        x = self.fc2(x)       # (batch, 34)
        return x

    def forward_to(self, x: torch.Tensor, split_id: int) -> torch.Tensor:
        """Run the edge partition: layers 1 through split_id.

        Args:
            x: Input tensor of shape (batch, num_features) or (batch, 1, num_features).
            split_id: One of SUPPORTED_SPLIT_IDS.

        Returns:
            Intermediate activation h_k(x).
        """
        if split_id not in SUPPORTED_SPLIT_IDS:
            raise ValueError(f"split_id {split_id} not in {SUPPORTED_SPLIT_IDS}")

        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, 80)

        if split_id == 0:
            # No edge compute — return reshaped input
            return x

        # Block 1
        x = self.block1(x)  # (batch, 32, 78)
        if split_id == 3:
            return x

        # Block 2
        x = self.block2(x)  # (batch, 64, 76)
        if split_id == 6:
            return x

        # Pool
        x = self.pool(x).squeeze(-1)  # (batch, 64)
        if split_id == 7:
            return x

        # FC1
        x = self.fc1(x)  # (batch, 128)
        if split_id == 8:
            return x

        # FC2 (local-only)
        x = self.fc2(x)  # (batch, 34)
        return x  # split_id == 9

    def forward_from(self, h: torch.Tensor, split_id: int) -> torch.Tensor:
        """Run the cloud partition: layers split_id+1 through L.

        Args:
            h: Intermediate activation from forward_to.
            split_id: One of SUPPORTED_SPLIT_IDS.

        Returns:
            Logits of shape (batch, num_classes).
        """
        if split_id not in SUPPORTED_SPLIT_IDS:
            raise ValueError(f"split_id {split_id} not in {SUPPORTED_SPLIT_IDS}")

        if split_id == 9:
            # Local-only — h is already the logits
            return h

        x = h

        if split_id == 0:
            # Cloud runs full model
            x = self.block1(x)

        if split_id <= 3:
            x = self.block2(x)

        if split_id <= 6:
            x = self.pool(x).squeeze(-1)

        if split_id <= 7:
            x = self.fc1(x)

        x = self.fc2(x)
        return x

    def get_layer_groups(self) -> OrderedDict[str, nn.Module]:
        """Return named sub-modules in execution order for profiling."""
        return OrderedDict([
            ("block1", self.block1),
            ("block2", self.block2),
            ("pool", self.pool),
            ("fc1", self.fc1),
            ("fc2", self.fc2),
        ])

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
