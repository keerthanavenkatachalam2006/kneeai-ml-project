#!/bin/bash
# =================================================================
# KneeAI – Redeploy / Update Script (run on EC2 after code changes)
# =================================================================
# Usage:
#   ssh ec2-user@<EC2-IP>
#   cd ~/kneeai
#   ./deploy/ec2_deploy.sh
# =================================================================

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

echo "===================================="
echo " KneeAI – Redeployment"
echo " Dir: $APP_DIR"
echo "===================================="

# Pull latest code
echo "[1/4] Pulling latest from GitHub..."
git pull

# Rebuild changed images only
echo "[2/4] Rebuilding Docker images..."
docker compose -f docker-compose.prod.yml build

# Rolling restart (backend first, then frontend)
echo "[3/4] Restarting services..."
docker compose -f docker-compose.prod.yml up -d --no-deps backend
sleep 15
docker compose -f docker-compose.prod.yml up -d --no-deps frontend

# Verify
echo "[4/4] Health check..."
sleep 10
HTTP=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ || echo "000")

if [ "$HTTP" = "200" ]; then
    echo ""
    echo "✅ Redeployment successful!"
    docker compose -f docker-compose.prod.yml ps
else
    echo "❌ Health check failed (HTTP $HTTP)"
    echo "--- Backend logs (last 30 lines) ---"
    docker compose -f docker-compose.prod.yml logs --tail=30 backend
    exit 1
fi
