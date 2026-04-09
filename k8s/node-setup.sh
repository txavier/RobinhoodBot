#!/bin/bash
# ============================================================================
# RobinhoodBot — Node Setup & Update Configuration
# ============================================================================
# Configures Ubuntu worker nodes for the K8s cluster:
#   - Pins Kubernetes packages (kubelet, kubeadm, kubectl) to a specific version
#   - Configures unattended-upgrades for security-only updates
#   - Blacklists kernel and K8s packages from auto-upgrade
#   - Ensures NFS client, containerd prerequisites are installed
#
# Run on each node via SSH, or use the DaemonSet (k8s/node-setup-daemonset.yaml)
# for cluster-wide application.
#
# Usage:
#   ssh kube@<node-ip> 'bash -s' < k8s/node-setup.sh
#   # Or for a specific K8s version:
#   ssh kube@<node-ip> 'bash -s' < k8s/node-setup.sh --k8s-version 1.35.3
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — edit these to match your cluster
# ---------------------------------------------------------------------------
K8S_VERSION="${1:-1.35.3}"          # Pin to this K8s minor.patch
ALLOWED_KERNEL=""                    # Leave empty to blacklist all kernel upgrades
                                     # Set to e.g. "6.8.0-100" to only allow that version

log() { echo "[node-setup] $(date '+%Y-%m-%d %H:%M:%S') $*"; }

# ---------------------------------------------------------------------------
# 1. Pin Kubernetes packages
# ---------------------------------------------------------------------------
log "Pinning Kubernetes packages to v${K8S_VERSION}..."

# apt-mark hold prevents apt upgrade from touching these
sudo apt-mark hold kubelet kubeadm kubectl 2>/dev/null || true

# Also pin via apt preferences (belt + suspenders)
cat <<EOF | sudo tee /etc/apt/preferences.d/kubernetes-pin > /dev/null
# Pin Kubernetes packages to ${K8S_VERSION} — managed by node-setup.sh
Package: kubelet kubeadm kubectl
Pin: version ${K8S_VERSION}*
Pin-Priority: 1001
EOF

log "K8s packages pinned to v${K8S_VERSION}"

# ---------------------------------------------------------------------------
# 2. Configure unattended-upgrades
# ---------------------------------------------------------------------------
log "Configuring unattended-upgrades..."

# Enable automatic security updates
cat <<'EOF' | sudo tee /etc/apt/apt.conf.d/20auto-upgrades > /dev/null
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
EOF

# Configure what gets upgraded and what's blacklisted
cat <<EOF | sudo tee /etc/apt/apt.conf.d/50unattended-upgrades > /dev/null
// Unattended-Upgrade configuration — managed by node-setup.sh
// Only allow security updates from Ubuntu
Unattended-Upgrade::Allowed-Origins {
    "\${distro_id}:\${distro_codename}-security";
};

// NEVER auto-upgrade these packages:
// - Kubernetes: version changes need coordinated rolling upgrade
// - Kernels: new kernels have caused node instability (6.8.0-106 crashes)
//   Kernel updates should be tested on one node first, then rolled out.
Unattended-Upgrade::Package-Blacklist {
    "kubelet";
    "kubeadm";
    "kubectl";
    "linux-image-*";
    "linux-headers-*";
    "linux-modules-*";
    "linux-base";
};

// Auto-reboot is handled by kured (Kubernetes Reboot Daemon), not here
Unattended-Upgrade::Automatic-Reboot "false";

// Email notifications (optional)
// Unattended-Upgrade::Mail "root";

// Remove unused kernel packages after upgrade
Unattended-Upgrade::Remove-Unused-Kernel-Packages "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
EOF

log "Unattended-upgrades configured (security-only, kernel+K8s blacklisted)"

# ---------------------------------------------------------------------------
# 3. Ensure required packages
# ---------------------------------------------------------------------------
log "Ensuring required packages are installed..."

PACKAGES_NEEDED=()
dpkg -l nfs-common >/dev/null 2>&1 || PACKAGES_NEEDED+=(nfs-common)
dpkg -l open-iscsi >/dev/null 2>&1 || PACKAGES_NEEDED+=(open-iscsi)

if [ ${#PACKAGES_NEEDED[@]} -gt 0 ]; then
    log "Installing: ${PACKAGES_NEEDED[*]}"
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${PACKAGES_NEEDED[@]}"
fi

# ---------------------------------------------------------------------------
# 4. Disable WiFi power management (rtw88/rtw_8822ce)
# ---------------------------------------------------------------------------
log "Configuring WiFi power management fix..."

cat <<EOF | sudo tee /etc/modprobe.d/rtw88-power.conf > /dev/null
# Disable WiFi power saving — managed by node-setup.sh
# Prevents rtw_8822ce dropouts under load
options rtw_core rtw_power_mgnt=0 rtw_ips_mode=0
EOF

# Apply immediately (modprobe conf takes effect on next boot/module reload)
sudo iwconfig wlp5s0 power off 2>/dev/null || true

log "WiFi power management disabled"

# ---------------------------------------------------------------------------
# 5. Verify state
# ---------------------------------------------------------------------------
log "=== Verification ==="
log "K8s hold status:"
apt-mark showhold 2>/dev/null | while read -r pkg; do log "  HOLD: $pkg"; done

log "Kubelet version: $(kubelet --version 2>/dev/null || echo 'not installed')"

if [ -f /var/run/reboot-required ]; then
    log "WARNING: Reboot required for: $(cat /var/run/reboot-required.pkgs 2>/dev/null || echo 'unknown')"
fi

log "Node setup complete."
