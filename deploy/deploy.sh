#!/usr/bin/env bash
###############################################################################
# deploy/deploy.sh
#
# Zero-downtime re-deploy: git pull → docker rebuild → container restart.
# Run this from the VM whenever you push a new commit.
#
# Usage (on the VM):
#   cd /opt/tradingbot
#   ./deploy/deploy.sh
#
# Or from your local machine via SSH:
#   ssh user@VM_IP "cd /opt/tradingbot && ./deploy/deploy.sh"
###############################################################################

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}[deploy]${NC} $*"; }
warn() { echo -e "${YELLOW}[deploy]${NC} $*"; }

info "=== Trading Bot Re-deploy ==="
info "Directory : $APP_DIR"
info "Time      : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# ── 1. Pull latest code ──────────────────────────────────────────────────────
info "Pulling latest commits from origin/master..."
git fetch origin
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/master)

if [[ "$LOCAL" == "$REMOTE" ]]; then
    warn "Already up-to-date ($(git log -1 --oneline))"
    warn "Force rebuild? Run: docker compose up -d --build --force-recreate"
    exit 0
fi

git pull --ff-only
info "Updated: $(git log -1 --oneline)"

# ── 2. Build new image ───────────────────────────────────────────────────────
info "Building Docker image..."
docker compose build --no-cache tradingbot

# ── 3. Restart container (rolling — no port downtime) ───────────────────────
info "Restarting container..."
docker compose up -d --force-recreate tradingbot

# ── 4. Wait for health check ─────────────────────────────────────────────────
info "Waiting for health check..."
MAX=30
for i in $(seq 1 $MAX); do
    if docker compose ps tradingbot | grep -q "healthy"; then
        info "Container is healthy after ${i}×5s"
        break
    fi
    if [[ $i -eq $MAX ]]; then
        warn "Health check not confirmed after ${MAX}×5s — check logs:"
        warn "  docker compose logs --tail=50 tradingbot"
    fi
    sleep 5
done

# ── 5. Show status ───────────────────────────────────────────────────────────
info "Container status:"
docker compose ps

info ""
info "=== Deploy complete ==="
info "Dashboard: http://$(curl -s --max-time 3 ifconfig.me 2>/dev/null || echo 'VM_IP'):8000/metals"
info "Logs     : docker compose logs -f tradingbot"
