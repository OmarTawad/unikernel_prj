"""Context vector extraction for the bandit policy.

Extracts the pre-decision context vector c_t = (τ̂_t, u_t, Ĥ_t):
    - τ̂_t: Estimated RTT (EWMA of recent observed round-trip times)
    - u_t:  CPU utilization
    - Ĥ_t:  Softmax entropy at shallowest feasible split (uncertainty signal)

IMPORTANT: τ̂_t is the PRE-DECISION estimate used to select k_t.
           τ_t returned by the cloud is the POST-INFERENCE actual latency.
           These are different values used at different stages.
"""

from __future__ import annotations

import logging
import time
from collections import deque

import numpy as np
import psutil
import torch

from unisplit.model.cnn import IoTCNN

logger = logging.getLogger("unisplit.edge.context")


class ContextExtractor:
    """Extracts the 3-dimensional context vector for split point selection.

    Maintains state for EWMA RTT estimation and computes CPU utilization
    and uncertainty on demand.
    """

    def __init__(
        self,
        rtt_ewma_alpha: float = 0.3,
        initial_rtt_ms: float = 20.0,
    ):
        """Initialize context extractor.

        Args:
            rtt_ewma_alpha: EWMA smoothing factor for RTT estimates.
                Higher = more weight on recent observations.
            initial_rtt_ms: Initial RTT estimate before any observations.
        """
        self.rtt_ewma_alpha = rtt_ewma_alpha
        self._rtt_estimate_ms = initial_rtt_ms
        self._rtt_history: deque[float] = deque(maxlen=100)

    def get_estimated_rtt(self) -> float:
        """Get the pre-decision estimated RTT τ̂_t.

        This is the EWMA of recently observed round-trip times.
        Used to SELECT the split point (pre-decision).

        Returns:
            Estimated RTT in milliseconds.
        """
        return self._rtt_estimate_ms

    def update_rtt_estimate(self, observed_rtt_ms: float) -> None:
        """Update RTT estimate with a newly observed round-trip time.

        Called AFTER each cloud round-trip completes.
        Uses exponentially weighted moving average.

        Args:
            observed_rtt_ms: Actual measured round-trip time in ms.
        """
        self._rtt_history.append(observed_rtt_ms)
        self._rtt_estimate_ms = (
            self.rtt_ewma_alpha * observed_rtt_ms
            + (1 - self.rtt_ewma_alpha) * self._rtt_estimate_ms
        )

    def get_cpu_utilization(self) -> float:
        """Get current edge CPU utilization u_t ∈ [0, 1].

        Returns:
            CPU utilization as fraction (0.0 to 1.0).
        """
        # Use a very short interval to avoid blocking
        cpu_pct = psutil.cpu_percent(interval=0.05)
        return cpu_pct / 100.0

    def get_uncertainty(
        self,
        model: IoTCNN | None,
        x: np.ndarray,
        k_min: int,
    ) -> float:
        """Compute softmax entropy Ĥ_t at the shallowest feasible split.

        The uncertainty signal from paper §4.2:
        High entropy → edge partition is uncertain → prefer deeper split
        Low entropy → easy sample → shallow split suffices

        For split_id=0 (no edge compute), returns max entropy (log(C))
        as there's no information from the edge partition.

        Args:
            model: Edge model (or None if k_min=0).
            x: Input features as numpy array.
            k_min: Shallowest feasible split ID.

        Returns:
            Softmax entropy Ĥ_t ∈ [0, log(C)].
        """
        if k_min == 0 or model is None:
            # No edge compute — maximum uncertainty
            return float(np.log(34))  # log(C) for C=34

        try:
            tensor = torch.from_numpy(x).float()
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)

            with torch.no_grad():
                h = model.forward_to(tensor, k_min)
                # Compute softmax entropy of the intermediate representation
                # Use a simple linear projection to get pseudo-logits
                flat = h.view(1, -1)
                # Softmax of the raw activation values as uncertainty proxy
                probs = torch.softmax(flat.mean(dim=0) if flat.shape[1] > 1 else flat.squeeze(), dim=0)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

            return float(entropy)

        except Exception as e:
            logger.warning(f"Uncertainty computation failed: {e}")
            return float(np.log(34))  # Fall back to max uncertainty

    def get_context_vector(
        self,
        model: IoTCNN | None = None,
        x: np.ndarray | None = None,
        k_min: int = 0,
    ) -> np.ndarray:
        """Compute the full context vector c_t = [τ̂_t, u_t, Ĥ_t].

        Args:
            model: Edge model for uncertainty computation.
            x: Input features.
            k_min: Shallowest feasible split.

        Returns:
            Context vector as numpy array of shape (3,).
        """
        rtt = self.get_estimated_rtt()
        cpu_util = self.get_cpu_utilization()

        if model is not None and x is not None:
            uncertainty = self.get_uncertainty(model, x, k_min)
        else:
            uncertainty = float(np.log(34))

        return np.array([rtt, cpu_util, uncertainty], dtype=np.float32)

    @property
    def rtt_history(self) -> list[float]:
        """Return recent RTT observations."""
        return list(self._rtt_history)
