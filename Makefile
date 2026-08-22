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
api: ## Run the FastAPI backend (dev reload)
	$(BIN)/uvicorn kneeai.api.main:app --reload --port 8000

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
