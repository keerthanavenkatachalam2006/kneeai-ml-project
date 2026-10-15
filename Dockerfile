# =================================================================
# KneeAI – Backend Dockerfile (Multi-stage, Production)
# =================================================================
# Stage 1: builder  – installs Python deps into an isolated venv
# Stage 2: runtime  – lean image with only what is needed to serve
#
# Build:
#   docker build -t kneeai-backend:latest .
#
# Run locally (no GPU):
#   docker run --rm -p 8000:8000 --env-file .env.docker kneeai-backend:latest
#
# AWS EC2 (via docker-compose.prod.yml):
#   docker compose -f docker-compose.prod.yml up -d
# =================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Build-time system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libglib2.0-0 \
        libgl1-mesa-glx \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV VENV=/opt/venv
RUN python -m venv $VENV
ENV PATH="$VENV/bin:$PATH"

# Upgrade tooling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install backend Python deps (cached separately from app code)
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install the kneeai source package
COPY pyproject.toml README.md /workspace/
COPY src/ /workspace/src/
WORKDIR /workspace
RUN pip install --no-cache-dir --no-deps -e .


# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime-only system libs (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1-mesa-glx \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
ENV VENV=/opt/venv
COPY --from=builder $VENV $VENV
ENV PATH="$VENV/bin:$PATH"

# ── Application code ──────────────────────────────────────────────
WORKDIR /app

COPY backend/  /app/backend/
COPY src/      /app/src/
COPY scripts/  /app/scripts/
COPY models/   /app/models/
COPY results/  /app/results/

# Writable runtime dirs (populated by Docker volumes on EC2)
RUN mkdir -p /app/data/raw \
             /app/data/processed \
             /app/data/metadata \
             /app/artifacts/meshes \
             /app/artifacts/cases

# ── Security: non-root user ───────────────────────────────────────
RUN groupadd -r kneeai && useradd -r -g kneeai -d /app kneeai
RUN chown -R kneeai:kneeai /app
USER kneeai

# ── Runtime env ──────────────────────────────────────────────────
ENV PYTHONPATH="/app/src:/app"
ENV PYTHONUNBUFFERED=1
ENV KNEEAI_ENV=production

# ── Health / Ports ───────────────────────────────────────────────
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# ── Entry point ──────────────────────────────────────────────────
# Single worker keeps model weights resident in RAM.
# --timeout-keep-alive 75 satisfies AWS ALB idle-timeout default.
CMD ["uvicorn", "backend.app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "75", \
     "--access-log"]
