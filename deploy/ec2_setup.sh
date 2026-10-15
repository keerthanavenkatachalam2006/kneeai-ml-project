#!/bin/bash
# =================================================================
# KneeAI – EC2 Instance Bootstrap Script
# =================================================================
# Run this ONCE on a fresh Amazon Linux 2023 / Ubuntu 22.04 EC2
# instance to install Docker, clone the repo, and start the stack.
#
# Recommended EC2 instance type: t3.xlarge (4 vCPU, 16 GB RAM)
# For GPU inference: g4dn.xlarge (NVIDIA T4, 16 GB RAM)
#
# Usage:
#   chmod +x ec2_setup.sh
#   ./ec2_setup.sh
# =================================================================

set -euo pipefail

REPO_URL="https://github.com/shamsundarsk/MR.git"
APP_DIR="$HOME/kneeai"
EC2_PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 || echo "localhost")

echo "============================================"
echo " KneeAI EC2 Bootstrap"
echo " Host IP: $EC2_PUBLIC_IP"
echo "============================================"

# ── 1. Update system ─────────────────────────────────────────────
echo "[1/7] Updating system packages..."
if command -v dnf &>/dev/null; then
    # Amazon Linux 2023
    sudo dnf update -y
    sudo dnf install -y git curl wget htop
else
    # Ubuntu 22.04
    sudo apt-get update -y
    sudo apt-get install -y git curl wget htop ca-certificates gnupg
fi

# ── 2. Install Docker ────────────────────────────────────────────
echo "[2/7] Installing Docker..."
if ! command -v docker &>/dev/null; then
    if command -v dnf &>/dev/null; then
        sudo dnf install -y docker
        sudo systemctl enable docker
        sudo systemctl start docker
    else
        # Ubuntu: official Docker repo
        sudo install -m 0755 -d /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
            | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        echo \
          "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
          https://download.docker.com/linux/ubuntu \
          $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
          | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        sudo apt-get update -y
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
        sudo systemctl enable docker
        sudo systemctl start docker
    fi
    sudo usermod -aG docker "$USER"
    echo "Docker installed. NOTE: log out and back in to use docker without sudo."
else
    echo "Docker already installed: $(docker --version)"
fi

# ── 3. Install Docker Compose plugin ────────────────────────────
echo "[3/7] Checking Docker Compose..."
if ! docker compose version &>/dev/null; then
    COMPOSE_VERSION="2.27.0"
    sudo mkdir -p /usr/local/lib/docker/cli-plugins
    sudo curl -SL \
        "https://github.com/docker/compose/releases/download/v${COMPOSE_VERSION}/docker-compose-linux-$(uname -m)" \
        -o /usr/local/lib/docker/cli-plugins/docker-compose
    sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
fi
echo "Docker Compose: $(docker compose version)"

# ── 4. Clone / update repository ────────────────────────────────
echo "[4/7] Cloning KneeAI repository..."
if [ -d "$APP_DIR" ]; then
    echo "Directory exists – pulling latest..."
    cd "$APP_DIR"
    git pull
else
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

# ── 5. Create runtime .env.docker ───────────────────────────────
echo "[5/7] Configuring .env.docker..."
cd "$APP_DIR"

# Generate a real JWT secret
JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(48))" 2>/dev/null \
             || openssl rand -base64 48 | tr -d '\n')

cat > .env.docker <<EOF
KNEEAI_ENV=production
KNEEAI_LOG_LEVEL=INFO
KNEEAI_DATA_ROOT=/app/data
KNEEAI_ARTIFACT_ROOT=/app/artifacts
KNEEAI_DATABASE_URL=sqlite:///./artifacts/kneeai.db
KNEEAI_REDIS_URL=redis://localhost:6379/0
KNEEAI_API_HOST=0.0.0.0
KNEEAI_API_PORT=8000
KNEEAI_JWT_SECRET=${JWT_SECRET}
KNEEAI_CORS_ORIGINS=http://${EC2_PUBLIC_IP}:3000,http://localhost:3000
KNEEAI_MODEL_REGISTRY=/app/models
KNEEAI_MLFLOW_TRACKING_URI=/app/artifacts/mlruns
KNEEAI_IMPLANT_CATALOG=builtin:reference-v0
KNEEAI_ALLOW_NONCLINICAL_IMPLANTS=true
NEXT_PUBLIC_API_URL=http://${EC2_PUBLIC_IP}:8000
EOF

echo ".env.docker created (JWT secret auto-generated)"

# ── 6. Build & start containers ──────────────────────────────────
echo "[6/7] Building Docker images (this takes 5-10 min first time)..."
# Use newgrp to pick up docker group without logout if needed
sg docker -c "docker compose -f docker-compose.prod.yml build --no-cache"
sg docker -c "docker compose -f docker-compose.prod.yml up -d"

# ── 7. Verify deployment ─────────────────────────────────────────
echo "[7/7] Verifying deployment..."
sleep 10

API_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    "http://localhost:8000/" 2>/dev/null || echo "000")

if [ "$API_STATUS" = "200" ]; then
    echo ""
    echo "============================================"
    echo " ✅ KneeAI deployed successfully!"
    echo "============================================"
    echo " Backend API:  http://${EC2_PUBLIC_IP}:8000"
    echo " API Docs:     http://${EC2_PUBLIC_IP}:8000/docs"
    echo " Frontend UI:  http://${EC2_PUBLIC_IP}:3000"
    echo ""
    echo " Container status:"
    sg docker -c "docker compose -f docker-compose.prod.yml ps"
    echo ""
    echo " IMPORTANT: Open EC2 Security Group inbound rules for:"
    echo "   TCP 8000 (Backend API)"
    echo "   TCP 3000 (Frontend UI)"
    echo "============================================"
else
    echo ""
    echo "⚠️  API health check returned HTTP $API_STATUS"
    echo "   Check logs with: docker compose -f docker-compose.prod.yml logs backend"
fi
