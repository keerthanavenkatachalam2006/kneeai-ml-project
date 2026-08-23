# Knee Intelligence Platform — developer entrypoints
#
# Local development targets the deterministic core (imaging, geometry,
# measurements, implants, planning, simulation, safety). Model training runs on
# Google Colab: see notebooks/.

PYTHON ?= python3.11
VENV   ?= .venv
BIN    := $(VENV)/bin

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
.PHONY: venv
venv: ## Create the virtualenv
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip setuptools wheel

.PHONY: install
install: venv ## Install core + imaging + api + dev extras
	$(BIN)/pip install -e ".[imaging,api,dev]"

.PHONY: install-core
install-core: venv ## Install only the deterministic core (no DICOM/API deps)
	$(BIN)/pip install -e ".[dev]"

.PHONY: install-viz
install-viz: ## Add the optional heavy 3D stack (VTK / PyVista / Open3D)
	$(BIN)/pip install -e ".[viz]"

# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------
.PHONY: test
test: ## Run the test suite
	$(BIN)/pytest

.PHONY: test-cov
test-cov: ## Run tests with coverage
	$(BIN)/pytest --cov=kneeai --cov-report=term-missing

.PHONY: lint
lint: ## Ruff lint
	$(BIN)/ruff check src tests

.PHONY: fmt
fmt: ## Ruff format + autofix
	$(BIN)/ruff check --fix src tests
	$(BIN)/ruff format src tests

.PHONY: typecheck
typecheck: ## mypy
	$(BIN)/mypy

.PHONY: check
check: lint typecheck test ## All quality gates

# ---------------------------------------------------------------------------
# Product
# ---------------------------------------------------------------------------
.PHONY: demo
demo: ## End-to-end pipeline on the synthetic phantom knee (no patient data)
	$(BIN)/kneeai demo run --out artifacts/demo

.PHONY: api
api: ## Run the FastAPI backend (dev reload, loopback only — there is no auth)
	$(BIN)/uvicorn kneeai.api.main:app --reload --host 127.0.0.1 --port 8000

.PHONY: infer
infer: ## Segment + plan one real OAIZIB-CM case with the trained model
	$(BIN)/kneeai infer --case-id $(CASE) --source ai_prediction \
		--image data/raw/oaizib/imagesTs/$(subst oaizib-,oaizib_,$(CASE))_0000.nii.gz \
		--side right --strict

.PHONY: model-card
model-card: ## Rebuild the model card from the training history
	$(BIN)/python scripts/write_model_card.py

# ---------------------------------------------------------------------------
# The externally supplied checkpoint (best_model.pth)
# ---------------------------------------------------------------------------
.PHONY: identify-model
identify-model: ## Recover an unlabelled checkpoint's config by scoring against ground truth
	$(BIN)/python scripts/identify_checkpoint.py --cases 417 --spacings 1.0 0.5

.PHONY: diagnose-model
diagnose-model: ## Confusion matrix + axis-flip search for the active checkpoint
	$(BIN)/python scripts/diagnose_checkpoint.py --case 417 --spacing 0.5

.PHONY: register-model
register-model: ## Wrap + register best_model.pth and point the adapter at it
	$(BIN)/python scripts/register_best_model.py --activate

.PHONY: evaluate-model
evaluate-model: ## Measure the active model on the held-out val split (slow: ~3 min/case)
	$(BIN)/python scripts/evaluate_segresnet.py

# ---------------------------------------------------------------------------
# OA classifier
# ---------------------------------------------------------------------------
.PHONY: oa-train
oa-train: ## Fit the OA classifier on ground-truth masks vs the cohort's KL grades
	$(BIN)/python scripts/train_oa_classifier.py

.PHONY: oa-train-honest
oa-train-honest: ## Refit and score on the model's OWN masks (needs evaluate-model first)
	$(BIN)/python scripts/train_oa_classifier.py --predicted-masks artifacts/val_predictions

.PHONY: backfill
backfill: ## Add oa.json + robotic_readiness.json to existing cases (no inference)
	$(BIN)/python scripts/backfill_case.py

.PHONY: notebook
notebook: ## Regenerate the Colab training notebook
	$(BIN)/python scripts/make_colab_notebook.py

.PHONY: db-init
db-init: ## Create database tables
	$(BIN)/kneeai db init

.PHONY: seed-implants
seed-implants: ## Load the reference (non-clinical) implant catalog
	$(BIN)/kneeai implants seed

.PHONY: registry
registry: ## Validate the dataset registry
	$(BIN)/kneeai datasets validate

.PHONY: web
web: ## Run the Next.js surgeon dashboard
	cd apps/web && npm run dev

.PHONY: web-install
web-install: ## Install web dependencies
	cd apps/web && npm install

.PHONY: web-clean
web-clean: ## Clear the Next build cache (fixes "Cannot find module './NNN.js'")
	rm -rf apps/web/.next
	@echo "cleared apps/web/.next — restart 'make web'"

.PHONY: web-build
web-build: ## Production build. STOP 'make web' first: both write to .next
	cd apps/web && npx tsc --noEmit && npx next build

# ---------------------------------------------------------------------------
# Infra
# ---------------------------------------------------------------------------
.PHONY: up
up: ## Start Postgres + Redis + Orthanc
	docker compose up -d

.PHONY: down
down: ## Stop infra
	docker compose down

.PHONY: clean
clean: ## Remove build/test caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
