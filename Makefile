SHELL := /bin/bash
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest

.PHONY: help install test dry-run preprocess-data train train-resume validate test-model \
        export-partitions profile-memory run-cloud run-edge smoke-test \
        docker-build docker-up docker-down docker-logs clean
.PHONY: uk-check uk-build uk-run uk-validate
.PHONY: export-edge-c export-edge-c-all c-edge-build c-edge-forward-verify c-edge-forward-verify-all c-edge-quant-verify c-edge-controller-verify c-edge-roundtrip c-edge-roundtrip-generic

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

c-edge-roundtrip: ## Run host-side C cloud /infer/split roundtrip test
	$(PYTEST) tests/test_edge_native_cloud_roundtrip.py -v --tb=short

c-edge-roundtrip-generic: ## Run generic multi-split C cloud /infer/split roundtrip test
	$(PYTEST) tests/test_edge_native_cloud_roundtrip_generic.py -v --tb=short

# ── Cleanup ────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	rm -rf $(VENV) build/ dist/ *.egg-info .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
