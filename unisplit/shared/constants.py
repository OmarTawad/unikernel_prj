"""Shared constants for UniSplit.

All supported split IDs, feature counts, class counts, and default memory
budgets are defined here as the single source of truth.
"""

from __future__ import annotations

# ── Model Architecture ──────────────────────────────────────────────────────

NUM_FEATURES: int = 80
"""Number of input features per network flow sample."""

NUM_CLASSES: int = 34
"""Number of output classes (33 attack types + benign)."""

# ── Supported Split Points ──────────────────────────────────────────────────
# These are the implementation split IDs. They correspond to logical block
# boundaries in the 1-D CNN, not all possible layer boundaries.
#
#   0  →  input (no edge compute)
#   3  →  after block1 (Conv1D+BN+ReLU)
#   6  →  after block2 (Conv1D+BN+ReLU)
#   7  →  after GlobalAvgPool
#   8  →  after FC1+ReLU
#   9  →  local-only (full model on edge)

SUPPORTED_SPLIT_IDS: list[int] = [0, 3, 6, 7, 8, 9]
"""The 6 supported split point identifiers."""

SPLIT_NAMES: dict[int, str] = {
    0: "input",
    3: "after_block1",
    6: "after_block2",
    7: "after_pool",
    8: "after_fc1",
    9: "local_only",
}

# ── Memory Defaults ─────────────────────────────────────────────────────────

DEFAULT_MEMORY_BUDGET_BYTES: int = 25_165_824  # 24 MB
"""Default edge memory budget in bytes (paper default)."""

DEFAULT_OVERHEAD_BYTES: int = 2_097_152  # 2 MB
"""Default OS/runtime overhead δ in bytes."""

# ── Class Labels ────────────────────────────────────────────────────────────

CLASS_NAMES: list[str] = [
    "Benign",
    "DDoS-RSTFINFlood",
    "DDoS-PSHACK_Flood",
    "DDoS-SYN_Flood",
    "DDoS-UDP_Flood",
    "DDoS-TCP_Flood",
    "DDoS-ICMP_Flood",
    "DDoS-SynonymousIP_Flood",
    "DDoS-ACK_Fragmentation",
    "DDoS-UDP_Fragmentation",
    "DDoS-ICMP_Fragmentation",
    "DDoS-SlowLoris",
    "DDoS-HTTP_Flood",
    "DoS-UDP_Flood",
    "DoS-SYN_Flood",
    "DoS-TCP_Flood",
    "DoS-HTTP_Flood",
    "Recon-PingSweep",
    "Recon-OSScan",
    "Recon-PortScan",
    "Recon-HostDiscovery",
    "VulnerabilityScan",
    "BrowserHijacking",
    "CommandInjection",
    "XSS",
    "SqlInjection",
    "Backdoor_Malware",
    "Uploading_Attack",
    "BruteForce",
    "DictionaryBruteForce",
    "MITM-ArpSpoofing",
    "DNS_Spoofing",
    "Mirai-greeth_flood",
    "Mirai-udpplain",
]
"""34 class label names from CIC-IoT2023 (33 attacks + benign)."""

# ── API Defaults ────────────────────────────────────────────────────────────

DEFAULT_CLOUD_HOST: str = "0.0.0.0"
DEFAULT_CLOUD_PORT: int = 8000
DEFAULT_MODEL_VERSION: str = "v0.1.0"
