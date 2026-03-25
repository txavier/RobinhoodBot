#!/bin/bash
# ============================================================================
# RobinhoodBot — Monitoring Deploy Script
# ============================================================================
# Manages the Prometheus + Grafana monitoring stack on the Kubernetes cluster.
#
# Usage:
#   ./deploy-monitoring.sh install         # Full install: Prometheus + Grafana + alerts + dashboards
#   ./deploy-monitoring.sh remove          # Completely remove monitoring (Helm releases, CRDs, namespaces)
#   ./deploy-monitoring.sh reinstall       # Remove everything, then install fresh
#   ./deploy-monitoring.sh status          # Show monitoring pod/service status
#   ./deploy-monitoring.sh restart         # Restart Grafana + Prometheus pods
#   ./deploy-monitoring.sh logs-grafana    # Tail Grafana logs
#   ./deploy-monitoring.sh logs-prometheus # Tail Prometheus logs
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load secrets from secrets.env
if [[ -f "${LOCAL_DIR}/secrets.env" ]]; then
    # shellcheck disable=SC1091
    source "${LOCAL_DIR}/secrets.env"
else
    echo "ERROR: secrets.env not found. Copy secrets.env.example to secrets.env and fill in your values."
    exit 1
fi

REMOTE_SSH="${REMOTE_USER}@${REMOTE_HOST}"

NAMESPACE="monitoring"
HELM_RELEASE="kube-prometheus-stack"
MONITORING_DIR="k8s/monitoring"

# SSH ControlMaster — reuse connections
SSH_CONTROL_DIR="$(mktemp -d)"
SSH_CONTROL_PATH="${SSH_CONTROL_DIR}/ctrl-%r@%h:%p"
SSH_OPTS=(-o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
          -o ControlMaster=auto -o ControlPath="${SSH_CONTROL_PATH}" -o ControlPersist=300)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()   { echo -e "${GREEN}[monitoring]${NC} $*"; }
warn()  { echo -e "${YELLOW}[monitoring]${NC} $*"; }
err()   { echo -e "${RED}[monitoring]${NC} $*" >&2; }
header(){ echo -e "\n${CYAN}═══════════════════════════════════════════════════${NC}"; echo -e "${CYAN} $*${NC}"; echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"; }

cleanup_ssh() {
    ssh -o ControlPath="${SSH_CONTROL_PATH}" -O exit "${REMOTE_SSH}" 2>/dev/null || true
    rm -rf "${SSH_CONTROL_DIR}"
}
trap cleanup_ssh EXIT

remote() {
    ssh "${SSH_OPTS[@]}" "${REMOTE_SSH}" "$@"
}

remote_kubectl() {
    remote "kubectl $*"
}

check_connection() {
    log "Checking SSH connectivity to ${REMOTE_HOST}..."
    if ! remote "echo ok" >/dev/null 2>&1; then
        err "Cannot connect to ${REMOTE_HOST}. Check SSH access."
        exit 1
    fi
    log "Connected to ${REMOTE_HOST}"
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd_install() {
    header "Installing Monitoring Stack"
    check_connection

    # Check Helm is available
    if ! remote "which helm" >/dev/null 2>&1; then
        err "Helm is not installed on ${REMOTE_HOST}. Install it first:"
        err "  curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash"
        exit 1
    fi

    # Step 0: Sync monitoring manifests to remote
    log "Syncing monitoring manifests to ${REMOTE_HOST}..."
    remote "mkdir -p ${REMOTE_DIR}/${MONITORING_DIR}"
    rsync -az \
        -e "ssh ${SSH_OPTS[*]}" \
        "${LOCAL_DIR}/${MONITORING_DIR}/" "${REMOTE_SSH}:${REMOTE_DIR}/${MONITORING_DIR}/"

    # Step 1: Add Helm repos
    log "Adding prometheus-community Helm repo..."
    remote "helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true"
    remote "helm repo update"

    # Step 2: Create namespace
    log "Creating namespace '${NAMESPACE}'..."
    remote_kubectl "create namespace ${NAMESPACE} 2>/dev/null || true"

    # Step 3: Install kube-prometheus-stack
    log "Installing ${HELM_RELEASE} via Helm..."
    remote "helm upgrade --install ${HELM_RELEASE} prometheus-community/kube-prometheus-stack \
        --namespace ${NAMESPACE} \
        --values ${REMOTE_DIR}/${MONITORING_DIR}/prometheus-values.yaml \
        --wait --timeout 5m"

    # Step 4: Apply PodMonitors for Ray metrics
    log "Applying Ray PodMonitor..."
    remote_kubectl "apply -f ${REMOTE_DIR}/${MONITORING_DIR}/ray-pod-monitor.yaml"

    # Step 5: Apply alert rules
    log "Applying OOMKill alert rules..."
    remote_kubectl "apply -f ${REMOTE_DIR}/${MONITORING_DIR}/oomkill-alert.yaml"

    # Step 6: Import Ray Grafana dashboard
    log "Importing Ray Grafana dashboard..."
    DASHBOARD_URL="https://raw.githubusercontent.com/ray-project/kuberay/master/config/grafana/default_grafana_dashboard.json"
    remote "TEMP_FILE=\$(mktemp) && \
        if curl -sfL '${DASHBOARD_URL}' -o \"\$TEMP_FILE\" && [ -s \"\$TEMP_FILE\" ]; then \
            kubectl create configmap ray-grafana-dashboards \
                --namespace ${NAMESPACE} \
                --from-file=\"default-dashboard.json=\$TEMP_FILE\" \
                --dry-run=client -o yaml | kubectl apply -f - && \
            kubectl label configmap ray-grafana-dashboards \
                --namespace ${NAMESPACE} \
                grafana_dashboard=1 --overwrite && \
            kubectl annotate configmap ray-grafana-dashboards \
                --namespace ${NAMESPACE} \
                grafana_folder=Ray --overwrite && \
            echo 'Ray dashboard imported successfully.'; \
        else \
            echo 'WARNING: Failed to download Ray dashboard. Import manually.'; \
        fi; \
        rm -f \"\$TEMP_FILE\""

    # Step 7: Verify
    log "Waiting for pods to be ready..."
    remote_kubectl "wait --for=condition=Ready pods --all -n ${NAMESPACE} --timeout=120s" || true

    echo ""
    cmd_status

    echo ""
    log "Installation complete!"
    log ""
    log "Access Grafana:    http://<node-ip>:30080"
    log "  Login:           admin / prom-operator"
    log ""
    log "Check status:      ./deploy-monitoring.sh status"
}

cmd_remove() {
    header "Removing All Monitoring"
    check_connection

    warn "This will remove Prometheus, Grafana, alert rules, CRDs, and both monitoring namespaces."
    read -rp "Are you sure? (yes/no): " confirm
    if [[ "$confirm" != "yes" ]]; then
        log "Aborted"
        exit 0
    fi

    # Remove Helm release in monitoring namespace
    log "Removing Helm release '${HELM_RELEASE}' from '${NAMESPACE}'..."
    remote "helm uninstall ${HELM_RELEASE} -n ${NAMESPACE} 2>/dev/null || true"

    # Remove Helm release in prometheus-system (legacy install from install-monitoring.sh)
    log "Removing legacy Helm release 'prometheus' from 'prometheus-system'..."
    remote "helm uninstall prometheus -n prometheus-system 2>/dev/null || true"

    # Delete namespaces
    log "Deleting namespace '${NAMESPACE}'..."
    remote_kubectl "delete namespace ${NAMESPACE} --ignore-not-found --timeout=60s" || true

    log "Deleting namespace 'prometheus-system'..."
    remote_kubectl "delete namespace prometheus-system --ignore-not-found --timeout=60s" || true

    # Clean up CRDs left behind by kube-prometheus-stack
    log "Cleaning up Prometheus CRDs..."
    CRDS=(
        "alertmanagerconfigs.monitoring.coreos.com"
        "alertmanagers.monitoring.coreos.com"
        "podmonitors.monitoring.coreos.com"
        "probes.monitoring.coreos.com"
        "prometheusagents.monitoring.coreos.com"
        "prometheuses.monitoring.coreos.com"
        "prometheusrules.monitoring.coreos.com"
        "scrapeconfigs.monitoring.coreos.com"
        "servicemonitors.monitoring.coreos.com"
        "thanosrulers.monitoring.coreos.com"
    )
    for crd in "${CRDS[@]}"; do
        remote_kubectl "delete crd ${crd} --ignore-not-found" 2>/dev/null || true
    done

    log "All monitoring resources removed"
}

cmd_reinstall() {
    header "Reinstalling Monitoring Stack"

    # Remove (with auto-confirm)
    check_connection

    warn "This will remove and reinstall the entire monitoring stack."
    read -rp "Are you sure? (yes/no): " confirm
    if [[ "$confirm" != "yes" ]]; then
        log "Aborted"
        exit 0
    fi

    # Remove without prompting again
    log "Removing existing monitoring..."
    remote "helm uninstall ${HELM_RELEASE} -n ${NAMESPACE} 2>/dev/null || true"
    remote "helm uninstall prometheus -n prometheus-system 2>/dev/null || true"
    remote_kubectl "delete namespace ${NAMESPACE} --ignore-not-found --timeout=60s" || true
    remote_kubectl "delete namespace prometheus-system --ignore-not-found --timeout=60s" || true

    CRDS=(
        "alertmanagerconfigs.monitoring.coreos.com"
        "alertmanagers.monitoring.coreos.com"
        "podmonitors.monitoring.coreos.com"
        "probes.monitoring.coreos.com"
        "prometheusagents.monitoring.coreos.com"
        "prometheuses.monitoring.coreos.com"
        "prometheusrules.monitoring.coreos.com"
        "scrapeconfigs.monitoring.coreos.com"
        "servicemonitors.monitoring.coreos.com"
        "thanosrulers.monitoring.coreos.com"
    )
    for crd in "${CRDS[@]}"; do
        remote_kubectl "delete crd ${crd} --ignore-not-found" 2>/dev/null || true
    done

    log "Waiting for namespaces to fully terminate..."
    # Actively wait for namespaces to be fully gone (not just deleted but terminating)
    for ns in ${NAMESPACE} prometheus-system; do
        while remote_kubectl "get namespace ${ns}" >/dev/null 2>&1; do
            log "  Waiting for namespace '${ns}' to terminate..."
            sleep 5
        done
    done
    log "Namespaces fully terminated"

    # Install fresh
    cmd_install
}

cmd_status() {
    header "Monitoring Status"
    check_connection

    echo ""
    log "Pods:"
    remote_kubectl "get pods -n ${NAMESPACE} -o wide" 2>/dev/null || warn "Namespace '${NAMESPACE}' not found"

    echo ""
    log "Services:"
    remote_kubectl "get svc -n ${NAMESPACE}" 2>/dev/null || true

    echo ""
    log "Endpoints (Grafana):"
    remote_kubectl "get endpoints -n ${NAMESPACE} ${HELM_RELEASE}-grafana" 2>/dev/null || true

    echo ""
    log "PVCs:"
    remote_kubectl "get pvc -n ${NAMESPACE}" 2>/dev/null || true

    # Check for legacy prometheus-system namespace
    if remote_kubectl "get namespace prometheus-system" >/dev/null 2>&1; then
        echo ""
        warn "Legacy 'prometheus-system' namespace also exists:"
        remote_kubectl "get pods -n prometheus-system" 2>/dev/null || true
    fi
}

cmd_restart() {
    header "Restarting Monitoring Pods"
    check_connection

    log "Restarting Grafana..."
    remote_kubectl "rollout restart deployment/${HELM_RELEASE}-grafana -n ${NAMESPACE}"

    log "Restarting Prometheus Operator..."
    remote_kubectl "rollout restart deployment/${HELM_RELEASE}-operator -n ${NAMESPACE}" 2>/dev/null || true

    log "Waiting for rollouts..."
    remote_kubectl "rollout status deployment/${HELM_RELEASE}-grafana -n ${NAMESPACE} --timeout=120s" || true

    log "Monitoring pods restarted"
}

cmd_logs_grafana() {
    header "Grafana Logs (Ctrl+C to stop)"
    check_connection
    remote_kubectl "logs -f deployment/${HELM_RELEASE}-grafana -n ${NAMESPACE} -c grafana --tail=100"
}

cmd_logs_prometheus() {
    header "Prometheus Logs (Ctrl+C to stop)"
    check_connection
    remote_kubectl "logs -f statefulset/prometheus-${HELM_RELEASE}-prometheus -n ${NAMESPACE} -c prometheus --tail=100"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
CMD="${1:-}"

case "$CMD" in
    install)         cmd_install ;;
    remove)          cmd_remove ;;
    reinstall)       cmd_reinstall ;;
    status)          cmd_status ;;
    restart)         cmd_restart ;;
    logs-grafana)    cmd_logs_grafana ;;
    logs-prometheus) cmd_logs_prometheus ;;
    *)
        echo "Usage: $0 {install|remove|reinstall|status|restart|logs-grafana|logs-prometheus}"
        echo ""
        echo "Commands:"
        echo "  install         Full install: Prometheus + Grafana + alerts + dashboards"
        echo "  remove          Completely remove monitoring (Helm, CRDs, namespaces)"
        echo "  reinstall       Remove everything, then install fresh"
        echo "  status          Show monitoring pod/service status"
        echo "  restart         Restart Grafana + Prometheus pods"
        echo "  logs-grafana    Tail Grafana logs"
        echo "  logs-prometheus Tail Prometheus logs"
        exit 1
        ;;
esac
