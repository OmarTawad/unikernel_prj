"""tc/netem helper scripts for simulating network conditions.

These are NOT auto-executed. They generate shell scripts that the user
can run manually to inject latency and packet loss.
"""

from __future__ import annotations


def generate_netem_script(
    interface: str = "eth0",
    delay_ms: int = 50,
    jitter_ms: int = 10,
    loss_pct: float = 1.0,
    output_path: str | None = None,
) -> str:
    """Generate a tc/netem shell script for network simulation.

    Simulates the paper's experimental setup (§5):
    tc netem to inject variable latency U[5, 200] ms and packet loss 0-5%.

    Args:
        interface: Network interface name.
        delay_ms: Base delay in milliseconds.
        jitter_ms: Jitter in milliseconds.
        loss_pct: Packet loss percentage.
        output_path: Optional file path to save the script.

    Returns:
        Shell script as string.
    """
    script = f"""#!/bin/bash
# UniSplit network simulation using tc/netem
# This script adds artificial latency and packet loss to simulate
# the edge-cloud network conditions described in the paper.
#
# Usage: sudo bash {output_path or 'netem_setup.sh'}
# To remove: sudo tc qdisc del dev {interface} root

set -e

INTERFACE="{interface}"
DELAY_MS={delay_ms}
JITTER_MS={jitter_ms}
LOSS_PCT={loss_pct}

echo "Setting up netem on $INTERFACE"
echo "  Delay: ${{DELAY_MS}}ms ± ${{JITTER_MS}}ms"
echo "  Loss:  ${{LOSS_PCT}}%"

# Remove existing qdisc if present
sudo tc qdisc del dev $INTERFACE root 2>/dev/null || true

# Add netem qdisc
sudo tc qdisc add dev $INTERFACE root netem \\
    delay ${{DELAY_MS}}ms ${{JITTER_MS}}ms distribution normal \\
    loss ${{LOSS_PCT}}%

echo "✓ Network simulation active on $INTERFACE"
echo "To remove: sudo tc qdisc del dev $INTERFACE root"
"""

    if output_path:
        with open(output_path, "w") as f:
            f.write(script)

    return script


def generate_cleanup_script(
    interface: str = "eth0",
    output_path: str | None = None,
) -> str:
    """Generate a cleanup script to remove netem rules."""
    script = f"""#!/bin/bash
# Remove tc/netem rules
sudo tc qdisc del dev {interface} root 2>/dev/null || true
echo "✓ Network simulation removed from {interface}"
"""

    if output_path:
        with open(output_path, "w") as f:
            f.write(script)

    return script
