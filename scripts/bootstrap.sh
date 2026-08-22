#!/usr/bin/env bash
# Bootstrap the local development environment.
#
# Local dev installs the deterministic core + imaging + API + dev tooling.
# Deep-learning dependencies (torch / MONAI) are deliberately NOT installed
# locally: training runs on Google Colab (see notebooks/).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV="${VENV:-$ROOT/.venv}"
LOG="$ROOT/artifacts/bootstrap.log"

mkdir -p "$ROOT/artifacts"
: > "$LOG"

log() { printf '%s\n' "$*" | tee -a "$LOG"; }

if [ ! -x "$VENV/bin/python" ]; then
  log "==> creating virtualenv at $VENV using $PYTHON_BIN"
  "$PYTHON_BIN" -m venv "$VENV"
fi

PY="$VENV/bin/python"
log "==> python: $("$PY" -V 2>&1)"

log "==> upgrading pip toolchain"
"$PY" -m pip install --upgrade pip setuptools wheel >>"$LOG" 2>&1

EXTRAS="${EXTRAS:-imaging,api,dev}"
log "==> installing knee-intelligence with extras: $EXTRAS"
if ! "$PY" -m pip install -e ".[$EXTRAS]" >>"$LOG" 2>&1; then
  log "!! full install failed; retrying with core+dev only"
  log "   (usually a missing wheel for this Python version — check $LOG)"
  "$PY" -m pip install -e ".[dev]" >>"$LOG" 2>&1
fi

log "==> import check"
"$PY" - <<'PYCODE' 2>&1 | tee -a "$LOG"
import importlib

import kneeai

print(f"kneeai {kneeai.__version__} imported")

optional = {
    "pydicom": "DICOM parsing",
    "SimpleITK": "medical image IO",
    "fastapi": "backend API",
    "sqlalchemy": "persistence",
    "trimesh": "mesh geometry",
    "shapely": "2D footprint geometry",
    "skimage": "marching cubes",
    "torch": "deep learning (expected only on Colab)",
    "monai": "medical DL (expected only on Colab)",
}
for mod, why in optional.items():
    try:
        importlib.import_module(mod)
        status = "ok"
    except ImportError:
        status = "absent"
    print(f"  {mod:<12} {status:<7} {why}")
PYCODE

log ""
log "==> done. Activate with: source $VENV/bin/activate"
log "    Full log: $LOG"
