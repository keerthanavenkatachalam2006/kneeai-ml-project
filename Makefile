# Knee Intelligence Platform — developer entrypoints
#
# Local development targets the deterministic core (imaging, geometry,
# measurements, implants, planning, simulation, safety). Model training runs on
# Google Colab: see notebooks/.
#
# POSIX ONLY. This file needs `make`, and $(BIN) below assumes a Unix virtualenv
# layout (.venv/bin, not .venv\Scripts). On Windows use ./morp.ps1 instead, which
# carries the same target names:  .\morp.ps1 help

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

.PHONY: site
site: ## Run the public showcase site (explains the project; no API, port 3001)
	cd site && npm run dev

.PHONY: site-install
site-install: ## Install showcase site dependencies
	cd site && npm install

.PHONY: site-build
site-build: ## Static export of the showcase site to site/out
	cd site && npm run build

.PHONY: site-shots
site-shots: ## Retake the showcase screenshots (needs 'make api' and 'make web' running)
	cd site && npm run shots

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
