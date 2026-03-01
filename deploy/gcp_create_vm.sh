#!/usr/bin/env bash
###############################################################################
# deploy/gcp_create_vm.sh
#
# Run this ONCE from your LOCAL machine (with gcloud SDK installed and
# authenticated) to create the GCP VM + firewall rule.
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT_ID
#
# Usage:
#   chmod +x deploy/gcp_create_vm.sh
#   ./deploy/gcp_create_vm.sh
###############################################################################

set -euo pipefail

# ── Config — edit these ──────────────────────────────────────────────────────
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
VM_NAME="tradingbot"
ZONE="us-central1-a"           # change to your preferred zone
MACHINE_TYPE="e2-small"        # 2 vCPU, 2 GB RAM — sufficient for this app
DISK_SIZE="20GB"
IMAGE_FAMILY="debian-12"
IMAGE_PROJECT="debian-cloud"
PORT=8000
FIREWALL_RULE="allow-tradingbot"
# ────────────────────────────────────────────────────────────────────────────

[[ -z "$PROJECT_ID" ]] && { echo "ERROR: set gcloud project first: gcloud config set project YOUR_PROJECT"; exit 1; }

echo "Creating VM: $VM_NAME in $ZONE (project: $PROJECT_ID)"

# ── 1. Create firewall rule ───────────────────────────────────────────────────
echo "Creating firewall rule: $FIREWALL_RULE..."
gcloud compute firewall-rules create "$FIREWALL_RULE" \
    --project="$PROJECT_ID" \
    --allow="tcp:$PORT" \
    --source-ranges="0.0.0.0/0" \
    --target-tags="tradingbot" \
    --description="Allow Trading Bot dashboard traffic on port $PORT" \
    2>/dev/null || echo "(firewall rule already exists, skipping)"

# ── 2. Create VM instance ─────────────────────────────────────────────────────
echo "Creating VM instance: $VM_NAME..."
gcloud compute instances create "$VM_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size="$DISK_SIZE" \
    --boot-disk-type="pd-balanced" \
    --tags="tradingbot" \
    --metadata=startup-script='#! /bin/bash
# Auto-run setup on first boot
if [ ! -f /opt/.tradingbot_setup_done ]; then
    echo "First boot — downloading and running setup script..."
    apt-get install -y -qq curl git
    curl -fsSL https://raw.githubusercontent.com/aviratna/tradingbot/master/deploy/gcp_setup.sh \
        -o /tmp/gcp_setup.sh
    chmod +x /tmp/gcp_setup.sh
    /tmp/gcp_setup.sh
    touch /opt/.tradingbot_setup_done
fi'

# ── 3. Get the external IP ────────────────────────────────────────────────────
echo ""
echo "Waiting for VM to be ready..."
sleep 5
EXTERNAL_IP=$(gcloud compute instances describe "$VM_NAME" \
    --zone="$ZONE" \
    --format="get(networkInterfaces[0].accessConfigs[0].natIP)")

echo ""
echo "================================================================"
echo "  VM created successfully!"
echo "  Instance : $VM_NAME"
echo "  Zone     : $ZONE"
echo "  IP       : $EXTERNAL_IP"
echo ""
echo "  The startup script will automatically:"
echo "    - Install Docker"
echo "    - Clone the repo"
echo "    - Start the container"
echo "  (This takes ~3-5 minutes on first boot)"
echo ""
echo "  NEXT STEPS:"
echo "  1. SSH into VM:"
echo "     gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "  2. Set your API keys:"
echo "     nano /opt/tradingbot/.env"
echo ""
echo "  3. Restart the app:"
echo "     cd /opt/tradingbot && docker compose restart"
echo ""
echo "  Dashboard: http://$EXTERNAL_IP:$PORT/metals"
echo "  Health:    http://$EXTERNAL_IP:$PORT/health"
echo "================================================================"
