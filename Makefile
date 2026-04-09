SHELL := /bin/bash
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest

.PHONY: help install test dry-run preprocess-data train train-resume validate test-model \
        export-partitions profile-memory run-cloud run-edge smoke-test \
        docker-build docker-up docker-down docker-logs clean
.PHONY: uk-check uk-build uk-run uk-validate
.PHONY: uk-edge-embed-artifacts uk-edge-build uk-edge-run uk-edge-validate uk-edge-build-pi
.PHONY: export-edge-c export-edge-c-all c-edge-build c-edge-forward-verify c-edge-forward-verify-all c-edge-quant-verify c-edge-controller-verify c-edge-failure-verify c-edge-roundtrip c-edge-roundtrip-generic c-edge-roundtrip-vps
.PHONY: prepi-validate pi-readiness-manifest pi-readiness-check pi-boot-payload pi-image-build pi-boot-media
.PHONY: pi4-phase1-audit pi4-phase1-build pi4-phase1-stage pi4-phase1-handoff
.PHONY: pi4-phase2-audit pi4-phase2-build pi4-phase2-stage pi4-phase2-handoff
.PHONY: pi4-phase3-audit pi4-phase3-build pi4-phase3-stage pi4-phase3-handoff
.PHONY: pi-uefi-check pi-uefi-stage pi-uefi-build pi-uefi-model-gate pi-uefi-discover-nonkvm pi-uefi-branchb-diagnose pi-uefi-boot-media pi-uefi-handoff

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ──────────────────────────────────────────────

install: ## Create venv, install all dependencies
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	@echo "✓ Install complete. Activate with: source $(VENV)/bin/activate"

# ── Testing ────────────────────────────────────────────

test: ## Run all tests
	$(PYTEST) tests/ -v --tb=short

test-unit: ## Run unit tests only (exclude smoke/integration)
	$(PYTEST) tests/ -v --tb=short -k "not smoke"

dry-run: ## Training dry-run (2 batches, synthetic data)
	$(PYTHON) -m unisplit.training.cli dry-run

# ── Dataset ────────────────────────────────────────────

download-data: ## Download CIC-IoT2023 dataset (may require manual steps)
	bash scripts/download_dataset.sh

preprocess-data: ## Preprocess raw dataset → processed numpy arrays
	$(PYTHON) scripts/preprocess_dataset.py --config configs/dataset.yaml

# ── Training ───────────────────────────────────────────

train: ## Train the model from scratch (requires preprocessed data)
	$(PYTHON) -m unisplit.training.cli train --config configs/default.yaml

train-resume: ## Resume training from latest checkpoint
	$(PYTHON) -m unisplit.training.cli train --config configs/default.yaml --resume checkpoints/latest.pt

validate: ## Validate model on val set
	$(PYTHON) -m unisplit.training.cli validate --config configs/default.yaml

test-model: ## Test model on test set
	$(PYTHON) -m unisplit.training.cli test --config configs/default.yaml

# ── Model Export ───────────────────────────────────────

export-partitions: ## Export edge/cloud partitions from best checkpoint
	$(PYTHON) scripts/export_partitions.py --checkpoint checkpoints/best.pt --output-dir partitions/

profile-memory: ## Profile model memory and compute feasibility
	$(PYTHON) scripts/profile_memory.py --budget 24M --output profiles/default.json

# ── Services ───────────────────────────────────────────

run-cloud: ## Start cloud inference service (foreground)
	$(PYTHON) scripts/run_cloud.py --config configs/cloud.yaml

run-edge: ## Run edge simulator
	$(PYTHON) scripts/run_edge.py --config configs/edge.yaml

smoke-test: ## End-to-end smoke test (cloud must be running)
	$(PYTHON) scripts/smoke_test.py

# ── Docker ─────────────────────────────────────────────

docker-build: ## Build Docker images
	docker compose -f deploy/docker-compose.yml build

docker-up: ## Start Docker services
	docker compose -f deploy/docker-compose.yml up -d

docker-down: ## Stop Docker services
	docker compose -f deploy/docker-compose.yml down

docker-logs: ## Follow Docker logs
	docker compose -f deploy/docker-compose.yml logs -f

# ── Unikraft / QEMU ARM64 Validation (T01) ─────────────

uk-check: ## Check Unikraft/QEMU ARM64 validation prerequisites
	@command -v qemu-system-aarch64 >/dev/null || (echo "Missing qemu-system-aarch64" && exit 1)
	@command -v aarch64-linux-gnu-gcc >/dev/null || (echo "Missing aarch64-linux-gnu-gcc" && exit 1)
	@command -v kraft >/dev/null || (echo "Missing kraft (install via https://get.kraftkit.sh)" && exit 1)
	@command -v flex >/dev/null || (echo "Missing flex (install: apt-get install flex)" && exit 1)
	@command -v bison >/dev/null || (echo "Missing bison (install: apt-get install bison)" && exit 1)
	@command -v socat >/dev/null || (echo "Missing socat (install: apt-get install socat)" && exit 1)
	@qemu-system-aarch64 -accel help

uk-build: ## Build ARM64 Unikraft hello target
	kraft --no-prompt --log-type basic build --plat qemu --arch arm64 edge_native/unikraft_hello

uk-run: ## Run ARM64 Unikraft hello on QEMU (TCG emulation)
	timeout 30s kraft --no-prompt --log-type basic run --plat qemu --arch arm64 --disable-acceleration edge_native/unikraft_hello

uk-validate: ## Full T01 validation flow
	bash scripts/validate_t01_unikraft_qemu.sh

uk-edge-build: ## Build ARM64 Unikraft edge selftest target
	$(MAKE) uk-edge-embed-artifacts
	kraft --no-prompt --log-type basic build --plat qemu --arch arm64 edge_native/unikraft_edge_selftest

uk-edge-run: ## Run ARM64 Unikraft edge selftest on QEMU (TCG emulation)
	timeout 40s kraft --no-prompt --log-type basic run --plat qemu --arch arm64 --disable-acceleration edge_native/unikraft_edge_selftest

uk-edge-validate: ## Validate Unikraft edge selftest boot markers and log
	bash scripts/validate_uk_edge_selftest_qemu.sh

uk-edge-embed-artifacts: ## Generate embedded model sources from exported edge_k9 artifacts
	$(MAKE) export-edge-c-all
	$(PYTHON) scripts/generate_embedded_edge_model.py --artifact-dir edge_native/artifacts/c_splits/edge_k9 --output-dir edge_native/unikraft_edge_selftest/generated

uk-edge-build-pi: ## Deprecated alias for Pi handoff build
	@echo "[deprecated] uk-edge-build-pi is removed for paper-aligned Pi4 UEFI flow. Use: make pi-uefi-build" >&2
	@exit 1

# ── Edge-Native Runtime (T02a) ──────────────────────────

export-edge-c: ## Export split-7 edge artifacts for C runtime
	$(PYTHON) scripts/export_edge_c_artifacts.py --partitions-dir partitions --split-id 7 --out-dir edge_native/artifacts/edge_k7_c --model-version v0.1.0 --source-checkpoint checkpoints/best.pt

export-edge-c-all: ## Export all supported edge splits for C runtime
	$(PYTHON) scripts/export_edge_c_artifacts.py --all --partitions-dir partitions --out-root-dir edge_native/artifacts/c_splits --model-version v0.1.0 --source-checkpoint checkpoints/best.pt

c-edge-build: ## Configure and build edge-native C runtime/test binaries
	cmake -S edge_native/runtime -B edge_native/runtime/build
	cmake --build edge_native/runtime/build -j

c-edge-forward-verify: ## Run export + C forward correctness tests
	$(PYTEST) tests/test_edge_native_export.py tests/test_edge_native_forward.py -v --tb=short

c-edge-forward-verify-all: ## Run multi-split C forward correctness tests
	$(PYTEST) tests/test_edge_native_export.py tests/test_edge_native_forward.py tests/test_edge_native_forward_multi.py -v --tb=short

c-edge-quant-verify: ## Run C/Python quantization parity test
	$(PYTEST) tests/test_edge_native_quant.py -v --tb=short

c-edge-controller-verify: ## Run C controller/LinUCB sanity tests
	$(PYTEST) tests/test_edge_native_controller.py -v --tb=short

c-edge-failure-verify: ## Run failure-path hardening tests
	$(PYTEST) tests/test_edge_native_failures.py tests/test_edge_native_transport_failures.py -v --tb=short

c-edge-roundtrip: ## Run host-side C cloud /infer/split roundtrip test
	$(PYTEST) tests/test_edge_native_cloud_roundtrip.py -v --tb=short

c-edge-roundtrip-generic: ## Run generic multi-split C cloud /infer/split roundtrip test
	$(PYTEST) tests/test_edge_native_cloud_roundtrip_generic.py -v --tb=short

c-edge-roundtrip-vps: ## Run VPS roundtrip matrix and persist evidence artifacts
	$(MAKE) export-edge-c-all
	$(MAKE) c-edge-build
	bash scripts/run_vps_roundtrip_matrix.sh

prepi-validate: ## Run the full pre-Pi validation baseline and emit report
	bash scripts/run_prepi_validation.sh

pi-readiness-manifest: ## Generate concrete Pi readiness manifest
	$(PYTHON) scripts/generate_pi_readiness_manifest.py --repo-root . --output artifacts/pi_handoff/latest/pi_readiness_manifest.json

pi-readiness-check: ## Verify pre-hardware Pi readiness paths and artifacts
	bash scripts/check_pi_readiness.sh

pi-boot-payload: ## Build Pi handoff payload tarball with manifest and C artifacts
	bash scripts/prepare_pi_boot_payload.sh

pi4-phase1-audit: ## Audit the local Pi4 platform path and reject QEMU/EFI drift
	bash scripts/build_pi4_phase1.sh --audit-only

pi4-phase1-build: ## Build direct-boot Pi4 phase-1 image and stage kernel8.img
	bash scripts/build_pi4_phase1.sh

pi4-phase1-stage: ## Assemble direct-boot Pi4 SD-card tree from a pinned firmware bundle
	bash scripts/stage_pi4_phase1_sdcard.sh

pi4-phase1-handoff: ## One-shot Pi4 phase-1 build plus SD-card staging
	$(MAKE) pi4-phase1-build
	$(MAKE) pi4-phase1-stage

pi4-phase2-audit: ## Audit the frozen-source Pi4 phase-2 boot-restore lane
	bash scripts/build_pi4_phase2.sh --audit-only

pi4-phase2-build: ## Build direct-boot Pi4 phase-2 image from the frozen phase-1 source snapshot
	bash scripts/build_pi4_phase2.sh

pi4-phase2-stage: ## Assemble direct-boot Pi4 phase-2 SD-card tree from a pinned firmware bundle
	bash scripts/stage_pi4_phase2_sdcard.sh

pi4-phase2-handoff: ## One-shot Pi4 phase-2 build plus SD-card staging
	$(MAKE) pi4-phase2-build
	$(MAKE) pi4-phase2-stage

pi4-phase3-audit: ## Audit the frozen-source Pi4 phase-3 selftest lane
	bash scripts/build_pi4_phase3.sh --audit-only

pi4-phase3-build: ## Build direct-boot Pi4 phase-3 image from the frozen phase-2 source snapshot
	bash scripts/build_pi4_phase3.sh

pi4-phase3-stage: ## Assemble direct-boot Pi4 phase-3 SD-card tree from a pinned firmware bundle
	bash scripts/stage_pi4_phase3_sdcard.sh

pi4-phase3-handoff: ## One-shot Pi4 phase-3 build plus SD-card staging
	$(MAKE) pi4-phase3-build
	$(MAKE) pi4-phase3-stage

pi-uefi-check: ## Check tooling + repo inputs for Pi4 UEFI handoff flow
	@command -v kraft >/dev/null || (echo "Missing kraft" && exit 1)
	@command -v unzip >/dev/null || (echo "Missing unzip" && exit 1)
	@command -v sha256sum >/dev/null || (echo "Missing sha256sum" && exit 1)
	@command -v python3 >/dev/null || (echo "Missing python3" && exit 1)
	@test -f configs/pi_uefi_bundle.lock.json || (echo "Missing configs/pi_uefi_bundle.lock.json" && exit 1)
	@test -f edge_native/unikraft_pi_uart_pof/Kraftfile || (echo "Missing edge_native/unikraft_pi_uart_pof/Kraftfile" && exit 1)
	@rg -n "platform:\\s*qemu" edge_native/unikraft_pi_uart_pof/Kraftfile >/dev/null || (echo "Pi UEFI app Kraftfile must define platform: qemu for EFI-stub flow" && exit 1)

pi-uefi-stage: ## Stage immutable pinned pftf/RPi4 bundle into handoff tree
	bash scripts/stage_pi_uefi_bundle.sh

pi-uefi-build: ## Build minimal Unikraft EFI/arm64 UART proof payload
	@bash scripts/build_pi_uefi_payload.sh
	@bash scripts/pi_uefi_model_gate.sh || { bash scripts/discover_pi_uefi_nonkvm.sh; exit 1; }

pi-uefi-model-gate: ## Reject Pi handoff when payload model is KVM/QEMU-derived and emit blocker report
	bash scripts/pi_uefi_model_gate.sh

pi-uefi-discover-nonkvm: ## Discover/prototype true non-KVM Pi UEFI payload path and emit report
	bash scripts/discover_pi_uefi_nonkvm.sh

pi-uefi-branchb-diagnose: ## Build + gate + forced non-KVM discovery report for Branch-B diagnosis
	@bash scripts/build_pi_uefi_payload.sh
	@bash scripts/pi_uefi_model_gate.sh || { bash scripts/discover_pi_uefi_nonkvm.sh; exit 1; }

pi-uefi-boot-media: ## Inject EFI payload and finalize Pi4 UEFI boot-media tree
	bash scripts/prepare_pi_uefi_boot_media.sh

pi-uefi-handoff: ## One-shot Pi4 UEFI handoff assembly (stage + build + package)
	$(MAKE) pi-uefi-check
	$(MAKE) pi-uefi-stage
	$(MAKE) pi-uefi-build
	$(MAKE) pi-uefi-boot-media

pi-image-build: ## Deprecated: direct-firmware kernel8 workflow is removed
	@echo "[deprecated] pi-image-build is removed for paper-aligned Pi4 UEFI flow. Use: make pi-uefi-build" >&2
	@exit 1

pi-boot-media: ## Deprecated: direct-firmware boot-media workflow is removed
	@echo "[deprecated] pi-boot-media is removed for paper-aligned Pi4 UEFI flow. Use: make pi-uefi-boot-media" >&2
	@exit 1

# ── Cleanup ────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	rm -rf $(VENV) build/ dist/ *.egg-info .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
