# Raspberry Pi 4 Phase-1 Platform Audit

This audit records the exact state of the local Unikraft snapshot used for the
fresh-start Raspberry Pi 4 phase-1 proof-of-life path.

## Source-of-truth scope

- Repo app used as the local vendor source: `edge_native/unikraft_pi_uart_pof`
- Vendored Unikraft snapshot inspected under:
  `edge_native/unikraft_pi_uart_pof/.unikraft/unikraft`
- New phase-1 app path:
  `edge_native/unikraft_pi4_phase1`

## What already exists locally

The local Unikraft snapshot already contains BCM2711-specific fragments:

- `plat/kvm/Config.uk` defines `CONFIG_KVM_VMM_RPI4`
- `plat/kvm/arm/rpi4_bpt64.S` exists and identity-maps the Pi4 BCM2711 layout
- `plat/kvm/include/kvm-arm64/image.h` sets `RAM_BASE_ADDR` to `0x00000000`
  for `CONFIG_KVM_VMM_RPI4`
- `plat/kvm/Makefile.uk` already selects `arm/rpi4_bpt64.S` when
  `CONFIG_KVM_VMM_RPI4=y`
- `CONFIG_KVM_VMM_RPI4` already implies PL011 early console and GICv2 support

## What was missing from the repo-usable path

The repo still lacked a trustworthy Pi4 phase-1 path because:

- all tracked Pi handoff commands were wired around `platform: qemu`
- the active payload path selected `CONFIG_KVM_VMM_QEMU=y`
- the tracked handoff flow produced EFI artifacts, not direct-boot `kernel8.img`
- `CONFIG_KVM_BOOT_PROTO_LXBOOT` did not accept `CONFIG_KVM_VMM_RPI4`, so the
  local Pi4 path could not emit a Linux-arm64 direct-boot image through the
  normal build flow

## Minimum local patch required

The minimum local Unikraft patch is stored in:

- `configs/patches/unikraft_pi4_platform.patch`

That patch does one thing:

- allow `CONFIG_KVM_VMM_RPI4` to use `CONFIG_KVM_BOOT_PROTO_LXBOOT`

## What did not need to be added

After auditing the vendored snapshot, these pieces did not need entirely new
source files:

- a new `rpi4_bpt64.S` file
- a new Pi4 RAM-base definition
- a new PL011 UART MMIO implementation

One local source fix was still required:

- the existing `rpi4_bpt64.S` file was present but malformed and did not
  assemble until its pseudo-op formatting was corrected

The rest of the missing work was selection and integration:

- a tracked Pi4 phase-1 app
- a tracked Pi4 defconfig
- a repo-native build script that selects Pi4 instead of QEMU
- a direct-boot SD staging script for `kernel8.img`

## Truth boundary

This repo now has a local Pi4-selected build path, but VPS-side success still
only proves:

- the Pi4 platform fragments are selected in the build
- the image emitted is a direct-boot Linux-arm64 image
- the staged SD tree matches the intended direct-boot layout

It does not prove bare-metal correctness until the image is booted on a real
Raspberry Pi 4 and the UART markers appear in order.
