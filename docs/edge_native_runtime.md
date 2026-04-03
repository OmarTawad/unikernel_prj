# Edge-Native Runtime (Multi-Split, Pi-Ready Baseline)

This stage provides a correctness-first edge-native baseline that is reusable for
QEMU validation now and Raspberry Pi bring-up later.

## What Is Implemented

- Multi-split plain-C forward runtime for split IDs `{0,3,6,7,8,9}`
- C-friendly artifact export (`manifest.json` + little-endian float32 `.bin`)
- Cloud contract-preserving client for `/infer/split`
- Backend-pluggable transport (`posix`, `ukstub`, `lwip`)
- Deterministic Unikraft edge self-test app with serial acceptance markers
- Pi handoff scripts for image candidate build + boot-media layout

## Locked Artifact Strategy

**Strategy: embed exported tensors into the unikernel image at build time.**

- Source export remains `edge_native/artifacts/c_splits/edge_k9` (superset tensors).
- Generator: `scripts/generate_embedded_edge_model.py`
- Generated embedded model sources:
  - `edge_native/unikraft_edge_selftest/generated/embedded_model.h`
  - `edge_native/unikraft_edge_selftest/generated/embedded_model.c`
- Runtime in unikernel uses embedded model loader (`edge_model_load_embedded`),
  not host filesystem reads.

This avoids day-1 filesystem/initrd uncertainty on Pi and keeps bring-up focused
on boot + network + protocol correctness.

## Transport Architecture

`transport_backend.h` exposes a backend-agnostic client API.

Backends:
- `posix`: host sockets backend for VPS/userland validation
- `ukstub`: deterministic synthetic backend for deterministic self-test paths
- `lwip`: real HTTP sockets backend intended for Unikraft/Pi runtime path

Current day-1 Pi endpoint:
- `http://204.168.156.245:8000`

Backend selection is runtime-configurable in the Unikraft app via command-line
arguments (see Pi boot cmdline template).

`lwip` endpoint format expectation right now:
- `http://<ipv4>:<port>` (IPv4 literal, not DNS hostname)

## Runtime Config Ingestion (Unikraft)

The Unikraft app consumes runtime options from kernel command-line arguments:

- `--split-id`
- `--backend`
- `--endpoint`
- `--path`
- `--timeout`
- `--retries`
- `--no-post`
- `--no-quant`
- `--no-controller`

Template source for Pi bring-up arguments:
- `configs/pi_boot/cmdline.txt.template`
- `configs/pi_edge_runtime.env.example`

Override path for custom day-1 builds:
- `UNISPLIT_PI_ENV_FILE=/abs/path/to/pi_runtime.env make pi-boot-media`

## Build + Validation Commands

```bash
# Export C artifacts and regenerate embedded model source
make uk-edge-embed-artifacts

# Build/validate Unikraft edge runtime on QEMU
make uk-edge-build
make uk-edge-validate

# Build deterministic Pi image candidate
make pi-image-build

# Prepare explicit boot-media layout
make pi-boot-media
```

## Pi Transition Notes

Ready now:
- Embedded model strategy locked and implemented
- Deterministic image output path + boot-media recipe
- Runtime config ingestion + acceptance markers
- Real lwip backend implementation under stable transport abstraction

Still hardware-only:
- PMU/lib-pmu
- INA219 instrumentation
- Final on-device networking and timing/perf characterization
- NEON optimization
- MQTT-on-device integration validation
