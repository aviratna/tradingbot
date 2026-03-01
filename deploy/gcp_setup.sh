#!/usr/bin/env bash
###############################################################################
# deploy/gcp_setup.sh
#
# ONE-SHOT bootstrap script for a fresh GCP Debian/Ubuntu Linux VM.
# Run this ONCE after creating your VM instance.
#
# Usage (on the VM):
#   curl -o gcp_setup.sh https://raw.githubusercontent.com/YOUR_USER/tradingbot/master/deploy/gcp_setup.sh
#   chmod +x gcp_setup.sh
#   sudo ./gcp_setup.sh
#
# What it does:
#   1. Updates system packages
#   2. Installs Docker + Docker Compose plugin
#   3. Adds current user to docker group
#   4. Clones the repo (or updates if already cloned)
#   5. Copies .env.example → .env (you fill in keys afterward)
#   6. Installs systemd service for auto-start on reboot
#   7. Starts the container
###############################################################################

set -euo pipefail

# ── Config — edit these ──────────────────────────────────────────────────────
REPO_URL="https://github.com/aviratna/tradingbot.git"
APP_DIR="/opt/tradingbot"
SERVICE_NAME="tradingbot"
APP_USER="tradingbot"          # non-root user that will run the container
PORT=8000
# ────────────────────────────────────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

[[ $EUID -ne 0 ]] && error "Run this script as root (sudo ./gcp_setup.sh)"

# ── 1. System update ─────────────────────────────────────────────────────────
info "Updating system packages..."
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq curl git ca-certificates gnupg lsb-release

# ── 2. Docker installation ───────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    info "Installing Docker..."
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/debian/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/$(. /etc/os-release && echo "$ID") \
      $(lsb_release -cs) stable" \
      > /etc/apt/sources.list.d/docker.list

    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io \
                           docker-buildx-plugin docker-compose-plugin
    systemctl enable docker
    systemctl start docker
    info "Docker installed: $(docker --version)"
else
    info "Docker already installed: $(docker --version)"
fi

# ── 3. Create app user ───────────────────────────────────────────────────────
if ! id "$APP_USER" &>/dev/null; then
    info "Creating user: $APP_USER"
    useradd -m -s /bin/bash "$APP_USER"
fi
usermod -aG docker "$APP_USER"
info "User $APP_USER added to docker group"

# ── 4. Clone or update repo ──────────────────────────────────────────────────
if [[ -d "$APP_DIR/.git" ]]; then
    info "Repo already cloned — pulling latest..."
    sudo -u "$APP_USER" git -C "$APP_DIR" pull --ff-only
else
    info "Cloning repo to $APP_DIR..."
    git clone "$REPO_URL" "$APP_DIR"
    chown -R "$APP_USER":"$APP_USER" "$APP_DIR"
fi

# ── 5. Create .env if missing ────────────────────────────────────────────────
if [[ ! -f "$APP_DIR/.env" ]]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    chown "$APP_USER":"$APP_USER" "$APP_DIR/.env"
    warn ".env created from .env.example"
    warn ">>> EDIT $APP_DIR/.env and add your API keys before starting! <<<"
    warn "    nano $APP_DIR/.env"
else
    info ".env already exists — skipping"
fi

# ── 6. Open firewall port (GCP uses external firewall rules, but set ufw too) ─
if command -v ufw &>/dev/null; then
    ufw allow "$PORT/tcp" || true
    info "UFW: port $PORT opened"
fi

# ── 7. Install systemd service ───────────────────────────────────────────────
info "Installing systemd service: $SERVICE_NAME"
cat > "/etc/systemd/system/${SERVICE_NAME}.service" << EOF
[Unit]
Description=Trading Bot (FastAPI + OSINT)
After=docker.service network-online.target
Requires=docker.service
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
User=$APP_USER
WorkingDirectory=$APP_DIR
ExecStart=/usr/bin/docker compose up -d --build
ExecStop=/usr/bin/docker compose down
Restart=on-failure
RestartSec=30
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}.service"
info "Systemd service enabled (auto-starts on reboot)"

# ── 8. Build and start ───────────────────────────────────────────────────────
info "Building Docker image and starting container..."
cd "$APP_DIR"
sudo -u "$APP_USER" docker compose pull --quiet 2>/dev/null || true
sudo -u "$APP_USER" docker compose up -d --build

info ""
info "================================================================"
info "  Setup complete!"
info "  App directory : $APP_DIR"
info "  Dashboard URL : http://$(curl -s ifconfig.me):$PORT/metals"
info "  Health check  : http://$(curl -s ifconfig.me):$PORT/health"
info ""
info "  NEXT STEPS:"
info "  1. Edit API keys:  nano $APP_DIR/.env"
info "  2. Restart app:    sudo systemctl restart $SERVICE_NAME"
info "  3. View logs:      cd $APP_DIR && docker compose logs -f"
info "================================================================"
