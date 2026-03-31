"""Edge simulator — orchestrates the full edge-side inference loop.

Flow per sample:
    1. Get sample from IngestionSource
    2. Extract context vector (RTT estimate, CPU util, uncertainty)
    3. Select split point via SplitPolicy
    4. Run edge partition via EdgeRunner
    5. If split_id < 9: quantize, send to cloud, receive prediction
    6. If split_id == 9: use local prediction
    7. Update policy with reward
    8. Update RTT estimate with observed round-trip
    9. Record metrics
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from unisplit.edge.client import CloudClient
from unisplit.edge.context import ContextExtractor
from unisplit.edge.ingestion import IngestionSource
from unisplit.edge.runner import EdgeRunner

logger = logging.getLogger("unisplit.edge.simulator")


@dataclass
class SampleResult:
    """Result from processing a single sample."""
    sample_idx: int
    split_id: int
    predicted_class: int
    true_label: int
    correct: bool
    offloaded: bool
    edge_inference_ms: float
    cloud_latency_ms: float       # τ_t from cloud (actual observed)
    rtt_ms: float                 # Edge-measured round-trip time
    total_latency_ms: float       # End-to-end for this sample
    context_rtt_estimate: float   # τ̂_t (pre-decision estimate)
    context_cpu_util: float       # u_t
    context_uncertainty: float    # Ĥ_t


@dataclass
class SimulationResults:
    """Aggregated results from a simulation run."""
    samples: list[SampleResult] = field(default_factory=list)
    total_samples: int = 0
    total_correct: int = 0
    total_offloaded: int = 0


class EdgeSimulator:
    """Orchestrates edge-side split inference simulation."""

    def __init__(
        self,
        runner: EdgeRunner,
        client: CloudClient,
        context_extractor: ContextExtractor,
        policy,  # SplitPolicy
        feasible_split_ids: list[int],
        edge_model=None,  # IoTCNN for uncertainty computation
    ):
        self.runner = runner
        self.client = client
        self.context = context_extractor
        self.policy = policy
        self.feasible_split_ids = feasible_split_ids
        self.edge_model = edge_model
        self.k_min = min(feasible_split_ids)

    def run(
        self,
        source: IngestionSource,
        max_samples: int = -1,
        lambda_latency: float = 0.001,
    ) -> SimulationResults:
        """Run the simulation over an ingestion source.

        Args:
            source: Sample source to replay.
            max_samples: Max samples to process (-1 = all).
            lambda_latency: Latency weight λ in reward function.

        Returns:
            SimulationResults with per-sample details.
        """
        results = SimulationResults()
        logger.info(f"Starting simulation, feasible splits: {self.feasible_split_ids}")

        for idx, (features, true_label) in enumerate(source):
            if max_samples > 0 and idx >= max_samples:
                break

            sample_start = time.perf_counter()

            # Step 2: Extract context (pre-decision)
            ctx = self.context.get_context_vector(
                model=self.edge_model,
                x=features,
                k_min=self.k_min,
            )

            # Step 3: Select split point
            split_id = self.policy.select(ctx)

            # Step 4: Run edge partition
            activation, edge_ms = self.runner.run(features, split_id)

            # Steps 5-6: Cloud or local
            if split_id < 9 and split_id != 9:
                # Offload to cloud
                response, rtt_ms = self.client.send_activation(
                    activation, split_id,
                )
                predicted_class = response.predicted_class
                cloud_latency_ms = response.timing.total_ms
                offloaded = True

                # Step 8: Update RTT estimate
                self.context.update_rtt_estimate(rtt_ms)
            else:
                # Local-only inference
                predicted_class = int(np.argmax(activation))
                cloud_latency_ms = 0.0
                rtt_ms = 0.0
                offloaded = False

            total_ms = (time.perf_counter() - sample_start) * 1000
            correct = predicted_class == true_label

            # Step 7: Update policy
            # Reward: r_t = 1[ŷ == y] - λ × τ_t
            actual_latency = cloud_latency_ms if offloaded else edge_ms
            reward = (1.0 if correct else 0.0) - lambda_latency * actual_latency
            self.policy.update(ctx, split_id, reward)

            # Step 9: Record
            result = SampleResult(
                sample_idx=idx,
                split_id=split_id,
                predicted_class=predicted_class,
                true_label=true_label,
                correct=correct,
                offloaded=offloaded,
                edge_inference_ms=edge_ms,
                cloud_latency_ms=cloud_latency_ms,
                rtt_ms=rtt_ms,
                total_latency_ms=total_ms,
                context_rtt_estimate=ctx[0],
                context_cpu_util=ctx[1],
                context_uncertainty=ctx[2],
            )
            results.samples.append(result)
            results.total_samples += 1
            results.total_correct += int(correct)
            results.total_offloaded += int(offloaded)

            if (idx + 1) % 100 == 0:
                acc = results.total_correct / results.total_samples
                offload_rate = results.total_offloaded / results.total_samples
                logger.info(
                    f"  [{idx + 1}] acc={acc:.4f}, offload_rate={offload_rate:.2%}, "
                    f"split={split_id}"
                )

        logger.info(
            f"Simulation complete: {results.total_samples} samples, "
            f"acc={results.total_correct / max(results.total_samples, 1):.4f}, "
            f"offloaded={results.total_offloaded}/{results.total_samples}"
        )

        return results
