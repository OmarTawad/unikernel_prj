# Unikraft QEMU Validation (T01, VPS Path)

This document defines the current T01 validation path for the unikernel side on
the VPS using QEMU ARM64 emulation.

## Goal

Validate boot/build/integration correctness only:

- Build a minimal Unikraft application for `qemu/arm64`
- Boot it with serial output on the VPS
- Confirm expected hello-world output

This is **not** a performance workflow and must not be used for final timing
claims.

## Environment Assumptions

- Host: Ubuntu 24.04 x86_64 VPS
- QEMU: `qemu-system-aarch64` available
- No nested KVM (`/dev/kvm` unavailable)
- Acceleration mode: TCG emulation only

## Install `kraft` (if missing)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://get.kraftkit.sh | sh
```

Then ensure your shell can see `kraft` (for example by reloading profile files
or adding the install location to `PATH`).

## Build Dependencies

The Unikraft configure/build flow also requires parser and runtime helpers:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends flex bison socat
```

## Repo-Native Commands

```bash
# Check prerequisites and show available QEMU accelerators
make uk-check

# Build the ARM64 Unikraft hello target
make uk-build

# Run in emulation-only mode (TCG)
make uk-run

# End-to-end deterministic validation (recommended)
make uk-validate
```

The source-based minimal app used by this flow lives in:

`edge_native/unikraft_hello/` (`Kraftfile`, `Makefile.uk`, `Config.uk`, `main.c`)

## Expected Output

During a successful validation run, the log should contain hello-world output
from the unikernel, and the script will finish with:

```text
[ok] T01 QEMU ARM64 unikernel boot validation passed (correctness mode).
```

Run logs are written to:

`artifacts/qemu/unikraft_hello_arm64.log`

## Boundaries of This Stage

- Valid now: boot correctness, build correctness, serial-console verification
- Not valid now: Raspberry Pi hardware behavior, PMU correctness, INA219 data
- Not valid now: performance or latency claims from emulation

## Next Hook (Post-T01)

After this boot path is stable, the next incremental step is adding a minimal
edge-native client scaffold that preserves the existing cloud contract in
`docs/protocol.md` and can later be integrated with split-activation transport.
