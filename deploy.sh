#!/bin/bash
# ============================================================================
# RobinhoodBot — Kubernetes Deploy Script
# ============================================================================
# Deploys RobinhoodBot and/or the Genetic Optimizer to the Kubernetes cluster
# running on the control-plane node defined in secrets.env.
#
# Usage:
#   ./deploy.sh                 # Full deploy: build image + deploy bot
#   ./deploy.sh build           # Build Docker image on remote only
#   ./deploy.sh bot             # Deploy/restart the trading bot
#   ./deploy.sh optimizer       # Launch a genetic optimizer job
#   ./deploy.sh optimizer-stop  # Delete the optimizer job
#   ./deploy.sh kuberay-install # Install KubeRay operator via Helm
#   ./deploy.sh kuberay-remove  # Uninstall KubeRay operator
#   ./deploy.sh sync-up         # Sync local data files TO the cluster
#   ./deploy.sh sync-down       # Sync cluster data files BACK to local
#   ./deploy.sh logs            # Tail bot logs
#   ./deploy.sh logs-optimizer  # Tail optimizer logs
#   ./deploy.sh status          # Show cluster status
#   ./deploy.sh shell           # Open a shell in the bot pod
#   ./deploy.sh stop            # Scale bot to 0 replicas
#   ./deploy.sh start           # Scale bot to 1 replica
#   ./deploy.sh destroy         # Delete everything (namespace + all resources)
#   ./deploy.sh setup-ssh       # Set up SSH key auth (run once)
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load secrets from secrets.env (PII: hostnames, usernames, paths)
if [[ -f "${LOCAL_DIR}/secrets.env" ]]; then
    # shellcheck disable=SC1091
    source "${LOCAL_DIR}/secrets.env"
else
    echo "ERROR: secrets.env not found. Copy secrets.env.example to secrets.env and fill in your values."
    echo "  cp secrets.env.example secrets.env"
    exit 1
fi

REMOTE_SSH="${REMOTE_USER}@${REMOTE_HOST}"

NAMESPACE="robinhoodbot"
IMAGE_NAME="robinhoodbot:latest"
REGISTRY="192.168.87.35:5000"
REGISTRY_IMAGE="${REGISTRY}/${IMAGE_NAME}"

# SSH ControlMaster — multiplex all SSH connections over a single session.
# Authenticate once (password or key), all subsequent ssh/scp/rsync reuse it.
SSH_CONTROL_DIR="$(mktemp -d)"
SSH_CONTROL_PATH="${SSH_CONTROL_DIR}/ctrl-%r@%h:%p"
SSH_OPTS=(-o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
          -o ControlMaster=auto -o ControlPath="${SSH_CONTROL_PATH}" -o ControlPersist=300)

# Data files to sync between local and cluster
DATA_FILES=(
    "robinhoodbot/logs/tradehistory-real.json"
    "robinhoodbot/logs/tradehistory.json"
    "robinhoodbot/logs/log.json"
    "robinhoodbot/logs/console_log.json"
    "robinhoodbot/logs/buy_reasons.json"
    "robinhoodbot/logs/ai_changelog.json"
    "robinhoodbot/logs/ai_suggested_config_changelog.json"
    "robinhoodbot/genetic_optimization_intraday_result.json"
    "robinhoodbot/genetic_optimization_intraday_result.checkpoint.json"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log()   { echo -e "${GREEN}[deploy]${NC} $*"; }
warn()  { echo -e "${YELLOW}[deploy]${NC} $*"; }
err()   { echo -e "${RED}[deploy]${NC} $*" >&2; }
header(){ echo -e "\n${CYAN}═══════════════════════════════════════════════════${NC}"; echo -e "${CYAN} $*${NC}"; echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"; }

# Clean up ControlMaster socket on exit
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
        err "Run: ./deploy.sh setup-ssh"
        exit 1
    fi
    log "Connected to ${REMOTE_HOST}"
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd_setup_ssh() {
    header "Setting up SSH key authentication"
    if [[ ! -f ~/.ssh/id_ed25519 ]]; then
        log "Generating SSH key pair..."
        ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "robinhoodbot-deploy"
    else
        log "SSH key already exists at ~/.ssh/id_ed25519"
    fi
    log "Copying public key to ${REMOTE_HOST}..."
    log "You will be prompted for the password ONE TIME."
    ssh-copy-id -i ~/.ssh/id_ed25519.pub "${REMOTE_SSH}"
    log "Testing passwordless connection..."
    if remote "echo ok" >/dev/null 2>&1; then
        log "SSH key auth is working!"
    else
        err "SSH key auth failed. You may need to manually authorize."
        exit 1
    fi
}

cmd_build() {
    header "Building Docker image on ${REMOTE_HOST}"
    check_connection

    # Ensure remote directory exists
    remote "mkdir -p ${REMOTE_DIR}"

    # Sync project files to the remote machine (excluding sensitive/large files)
    log "Syncing project files to ${REMOTE_HOST}:${REMOTE_DIR}..."
    rsync -az --delete \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='.venv' \
        --exclude='venv' \
        --exclude='*.pyc' \
        --exclude='robinhoodbot/config.py' \
        --exclude='robinhoodbot/logs/' \
        --exclude='robinhoodbot/genetic_optimization_intraday_result.json' \
        --exclude='robinhoodbot/genetic_optimization_intraday_result.checkpoint.json' \
        --exclude='robinhoodbot/.yfinance_cache' \
        --exclude='robinhoodbot/archive/' \
        -e "ssh ${SSH_OPTS[*]}" \
        "${LOCAL_DIR}/" "${REMOTE_SSH}:${REMOTE_DIR}/"

    # Build the Docker image on the remote node
    log "Building Docker image ${IMAGE_NAME}..."
    remote "cd ${REMOTE_DIR} && docker build -t ${IMAGE_NAME} ."

    log "Image built successfully on ${REMOTE_HOST}"

    # Push to local registry so all K8s nodes can pull it.
    # Use localhost:5000 for the push (Docker trusts localhost as insecure by default).
    # K8s manifests reference 192.168.87.35:5000 so workers can pull.
    log "Pushing image to local registry (${REGISTRY})..."
    remote "docker tag ${IMAGE_NAME} localhost:5000/${IMAGE_NAME} && docker push localhost:5000/${IMAGE_NAME}"

    log "Image available to all Kubernetes nodes via ${REGISTRY_IMAGE}"
}

cmd_deploy_bot() {
    header "Deploying RobinhoodBot to Kubernetes"
    check_connection

    # Create namespace
    log "Creating namespace '${NAMESPACE}'..."
    remote_kubectl "apply -f ${REMOTE_DIR}/k8s/namespace.yaml"

    # Apply cluster-wide resources (priority classes, node setup)
    log "Applying cluster-wide resources..."
    remote_kubectl "apply -f ${REMOTE_DIR}/k8s/priority-classes.yaml"
    remote_kubectl "apply -f ${REMOTE_DIR}/k8s/node-setup-daemonset.yaml"

    # Create/update config secret
    log "Creating config secret..."
    if [[ -f "${LOCAL_DIR}/robinhoodbot/config.py" ]]; then
        # Copy config.py to remote temporarily, create secret, remove it
        scp "${SSH_OPTS[@]}" \
            "${LOCAL_DIR}/robinhoodbot/config.py" \
            "${REMOTE_SSH}:/tmp/robinhoodbot-config.py"
        remote_kubectl "create secret generic robinhoodbot-config \
            --from-file=config.py=/tmp/robinhoodbot-config.py \
            -n ${NAMESPACE} \
            --dry-run=client -o yaml | kubectl apply -f -"
        remote "rm -f /tmp/robinhoodbot-config.py"
    else
        err "robinhoodbot/config.py not found! Copy config.py.sample and configure it."
        exit 1
    fi

    # Create/update credentials secret (RH_PASSWORD from secrets.env)
    if [[ -n "${RH_PASSWORD:-}" ]]; then
        log "Creating credentials secret..."
        remote_kubectl "create secret generic robinhoodbot-secrets \
            --from-literal=rh-password='${RH_PASSWORD}' \
            -n ${NAMESPACE} \
            --dry-run=client -o yaml | kubectl apply -f -"
    else
        warn "RH_PASSWORD not set in secrets.env — bot will prompt for password (will fail in container)."
    fi

    # Create PVCs
    log "Creating PersistentVolumeClaims..."
    remote_kubectl "apply -f ${REMOTE_DIR}/k8s/robinhoodbot-data-longhorn-pvc.yaml"
    remote_kubectl "apply -f ${REMOTE_DIR}/k8s/optimizer-cache-pvc.yaml"
    remote_kubectl "apply -f ${REMOTE_DIR}/k8s/optimizer-data-pvc.yaml"

    # Deploy
    log "Applying bot deployment..."
    remote_kubectl "apply -f ${REMOTE_DIR}/k8s/deployment.yaml"

    # Force restart so the pod picks up the new image (tag is always :latest)
    log "Restarting bot pod..."
    remote_kubectl "rollout restart deployment/robinhoodbot -n ${NAMESPACE}"

    # Wait for rollout
    log "Waiting for deployment rollout..."
    remote_kubectl "rollout status deployment/robinhoodbot -n ${NAMESPACE} --timeout=120s" || true

    log "Bot deployed! Use: ./deploy.sh logs"
}

cmd_deploy_optimizer() {
    header "Launching Genetic Optimizer (Ray Cluster)"
    check_connection

    # Ensure PVCs exist
    remote_kubectl "apply -f ${REMOTE_DIR}/k8s/optimizer-cache-pvc.yaml"
    remote_kubectl "apply -f ${REMOTE_DIR}/k8s/optimizer-data-pvc.yaml"

    # Deploy the Ray cluster (head + workers across nodes)
    log "Creating Ray cluster (1 head + 4 workers)..."
    remote_kubectl "apply -f ${REMOTE_DIR}/k8s/ray-cluster.yaml"

    # Wait for Ray head to be ready
    log "Waiting for Ray head pod..."
    remote_kubectl "wait --for=condition=Ready pod -l ray-role=head -n ${NAMESPACE} --timeout=180s" || true

    # Wait for workers
    log "Waiting for Ray worker pods..."
    sleep 5
    remote_kubectl "get pods -n ${NAMESPACE} -l app=robinhoodbot-optimizer -o wide"

    # Delete previous optimizer job if it exists
    remote_kubectl "delete job robinhoodbot-optimizer -n ${NAMESPACE} --ignore-not-found"

    # Launch the optimizer job (connects to Ray cluster via RAY_ADDRESS)
    log "Starting optimizer job..."
    remote_kubectl "apply -f ${REMOTE_DIR}/k8s/optimizer-job.yaml"

    log "Optimizer job launched on Ray cluster!"
    log "  Watch progress:   ./deploy.sh logs-optimizer"
    log "  Ray cluster:      ./deploy.sh ray-status"
    log "  Stop everything:  ./deploy.sh optimizer-stop"

    # Show initial status
    sleep 3
    remote_kubectl "get pods -n ${NAMESPACE} -l app=robinhoodbot-optimizer -o wide"
}

cmd_optimizer_stop() {
    header "Stopping Genetic Optimizer"
    check_connection
    remote_kubectl "delete job robinhoodbot-optimizer -n ${NAMESPACE} --ignore-not-found"
    remote_kubectl "delete raycluster optimizer-ray -n ${NAMESPACE} --ignore-not-found"
    log "Optimizer job and Ray cluster deleted"
}

cmd_kuberay_install() {
    header "Installing KubeRay Operator (Helm)"
    check_connection

    # Check if helm is available on the remote
    if ! remote "which helm" >/dev/null 2>&1; then
        err "Helm is not installed on ${REMOTE_HOST}. Install it first:"
        err "  curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash"
        exit 1
    fi

    # Add the KubeRay repo if not already added
    log "Adding KubeRay Helm repo..."
    remote "helm repo add kuberay https://ray-project.github.io/kuberay-helm/ 2>/dev/null || true"
    remote "helm repo update kuberay"

    # Install the operator
    log "Installing KubeRay operator..."
    remote "helm install kuberay-operator kuberay/kuberay-operator --namespace kuberay-system --create-namespace"

    # Wait for operator pod to be ready
    log "Waiting for KubeRay operator pod..."
    remote_kubectl "wait --for=condition=Ready pod -l app.kubernetes.io/name=kuberay-operator -n kuberay-system --timeout=120s" || true

    log "KubeRay operator installed!"
    remote_kubectl "get pods -n kuberay-system"
}

cmd_kuberay_remove() {
    header "Removing KubeRay Operator"
    check_connection

    # Stop optimizer first if running
    log "Cleaning up optimizer resources..."
    remote_kubectl "delete job robinhoodbot-optimizer -n ${NAMESPACE} --ignore-not-found"
    remote_kubectl "delete raycluster optimizer-ray -n ${NAMESPACE} --ignore-not-found"

    # Check if helm is available
    if remote "which helm" >/dev/null 2>&1; then
        log "Uninstalling KubeRay Helm release..."
        remote "helm uninstall kuberay-operator --namespace kuberay-system" 2>/dev/null || true
    fi

    # Delete the namespace
    log "Deleting kuberay-system namespace..."
    remote_kubectl "delete namespace kuberay-system --ignore-not-found"

    log "KubeRay operator removed"
}

cmd_sync_up() {
    header "Syncing local data TO cluster"
    check_connection

    # Get the bot pod name
    POD=$(remote_kubectl "get pod -n ${NAMESPACE} -l app=robinhoodbot -o jsonpath='{.items[0].metadata.name}'" 2>/dev/null || echo "")

    if [[ -z "$POD" || "$POD" == "''" ]]; then
        warn "No bot pod running. Syncing files to remote host filesystem instead."
        remote "mkdir -p ${REMOTE_DIR}/robinhoodbot/logs"
        for f in "${DATA_FILES[@]}"; do
            if [[ -f "${LOCAL_DIR}/${f}" ]]; then
                log "  Uploading ${f}..."
                scp "${SSH_OPTS[@]}" \
                    "${LOCAL_DIR}/${f}" "${REMOTE_SSH}:${REMOTE_DIR}/${f}" 2>/dev/null || warn "  Skipped ${f}"
            fi
        done
    else
        log "Copying data files into pod ${POD}..."
        for f in "${DATA_FILES[@]}"; do
            if [[ -f "${LOCAL_DIR}/${f}" ]]; then
                BASENAME=$(basename "$f")
                log "  Uploading ${BASENAME}..."
                # Copy to remote host first, then kubectl cp into pod
                scp "${SSH_OPTS[@]}" \
                    "${LOCAL_DIR}/${f}" "${REMOTE_SSH}:/tmp/${BASENAME}" 2>/dev/null || continue
                remote_kubectl "cp /tmp/${BASENAME} ${NAMESPACE}/${POD}:/app/data/${BASENAME}" 2>/dev/null || warn "  Failed to copy ${BASENAME} into pod"
                remote "rm -f /tmp/${BASENAME}"
            fi
        done
    fi
    log "Sync complete"
}

cmd_sync_down() {
    header "Syncing cluster data BACK to local"
    check_connection

    POD=$(remote_kubectl "get pod -n ${NAMESPACE} -l app=robinhoodbot -o jsonpath='{.items[0].metadata.name}'" 2>/dev/null || echo "")

    if [[ -z "$POD" || "$POD" == "''" ]]; then
        warn "No bot pod running. Trying to pull from remote filesystem..."
        for f in "${DATA_FILES[@]}"; do
            BASENAME=$(basename "$f")
            log "  Downloading ${f}..."
            scp "${SSH_OPTS[@]}" \
                "${REMOTE_SSH}:${REMOTE_DIR}/${f}" "${LOCAL_DIR}/${f}" 2>/dev/null || warn "  Skipped ${f}"
        done
    else
        log "Copying data files from pod ${POD}..."
        for f in "${DATA_FILES[@]}"; do
            BASENAME=$(basename "$f")
            log "  Downloading ${BASENAME}..."
            remote_kubectl "cp ${NAMESPACE}/${POD}:/app/data/${BASENAME} /tmp/${BASENAME}" 2>/dev/null || { warn "  Skipped ${BASENAME}"; continue; }
            scp "${SSH_OPTS[@]}" \
                "${REMOTE_SSH}:/tmp/${BASENAME}" "${LOCAL_DIR}/${f}" 2>/dev/null || warn "  Failed to download ${BASENAME}"
            remote "rm -f /tmp/${BASENAME}"
        done
    fi
    log "Sync complete"
}

cmd_logs() {
    header "Bot Logs (Ctrl+C to stop)"
    check_connection
    remote_kubectl "logs -f deployment/robinhoodbot -n ${NAMESPACE} --tail=100"
}

cmd_logs_optimizer() {
    header "Optimizer Logs (Ctrl+C to stop)"
    check_connection
    remote_kubectl "logs -f job/robinhoodbot-optimizer -n ${NAMESPACE} --tail=100"
}

cmd_status() {
    header "Cluster Status"
    check_connection

    echo ""
    log "Nodes:"
    remote_kubectl "get nodes -o wide"

    echo ""
    log "Namespace '${NAMESPACE}' resources:"
    remote_kubectl "get all -n ${NAMESPACE}" 2>/dev/null || warn "Namespace not found"

    echo ""
    log "PVCs:"
    remote_kubectl "get pvc -n ${NAMESPACE}" 2>/dev/null || true

    echo ""
    log "Secrets:"
    remote_kubectl "get secrets -n ${NAMESPACE}" 2>/dev/null || true

    echo ""
    log "Pod details:"
    remote_kubectl "get pods -n ${NAMESPACE} -o wide" 2>/dev/null || true

    echo ""
    log "Ray clusters:"
    remote_kubectl "get raycluster -n ${NAMESPACE}" 2>/dev/null || true
}

cmd_ray_status() {
    header "Ray Cluster Status"
    check_connection

    log "Ray cluster:"
    remote_kubectl "get raycluster -n ${NAMESPACE} -o wide" 2>/dev/null || warn "No Ray cluster found"

    echo ""
    log "Ray pods:"
    remote_kubectl "get pods -n ${NAMESPACE} -l app=robinhoodbot-optimizer -o wide" 2>/dev/null || true

    echo ""
    log "Ray head resources:"
    HEAD_POD=$(remote_kubectl "get pod -n ${NAMESPACE} -l ray-role=head -o jsonpath='{.items[0].metadata.name}'" 2>/dev/null || echo "")
    if [[ -n "$HEAD_POD" && "$HEAD_POD" != "''" ]]; then
        remote_kubectl "exec ${HEAD_POD} -n ${NAMESPACE} -- ray status" 2>/dev/null || warn "Could not get Ray status"
    else
        warn "No Ray head pod found"
    fi
}

cmd_shell() {
    header "Opening shell in bot pod"
    check_connection
    POD=$(remote_kubectl "get pod -n ${NAMESPACE} -l app=robinhoodbot -o jsonpath='{.items[0].metadata.name}'" 2>/dev/null || echo "")
    if [[ -z "$POD" || "$POD" == "''" ]]; then
        err "No bot pod running."
        exit 1
    fi
    log "Connecting to pod ${POD}..."
    ssh -t "${SSH_OPTS[@]}" "${REMOTE_SSH}" \
        "kubectl exec -it ${POD} -n ${NAMESPACE} -- bash"
}

cmd_stop() {
    header "Stopping bot (scale to 0)"
    check_connection
    remote_kubectl "scale deployment robinhoodbot --replicas=0 -n ${NAMESPACE}"
    log "Bot stopped"
}

cmd_start() {
    header "Starting bot (scale to 1)"
    check_connection
    remote_kubectl "scale deployment robinhoodbot --replicas=1 -n ${NAMESPACE}"
    log "Bot started. Use: ./deploy.sh logs"
}

cmd_destroy() {
    header "Destroying all RobinhoodBot resources"
    check_connection

    warn "This will delete the '${NAMESPACE}' namespace and ALL resources within it."
    read -rp "Are you sure? (yes/no): " confirm
    if [[ "$confirm" != "yes" ]]; then
        log "Aborted"
        exit 0
    fi

    remote_kubectl "delete namespace ${NAMESPACE} --ignore-not-found"
    log "Namespace '${NAMESPACE}' deleted"
}

cmd_update_config() {
    header "Updating config secret"
    check_connection

    if [[ ! -f "${LOCAL_DIR}/robinhoodbot/config.py" ]]; then
        err "robinhoodbot/config.py not found!"
        exit 1
    fi

    scp "${SSH_OPTS[@]}" \
        "${LOCAL_DIR}/robinhoodbot/config.py" \
        "${REMOTE_SSH}:/tmp/robinhoodbot-config.py"
    remote_kubectl "create secret generic robinhoodbot-config \
        --from-file=config.py=/tmp/robinhoodbot-config.py \
        -n ${NAMESPACE} \
        --dry-run=client -o yaml | kubectl apply -f -"
    remote "rm -f /tmp/robinhoodbot-config.py"

    log "Config secret updated. Restarting bot..."
    remote_kubectl "rollout restart deployment/robinhoodbot -n ${NAMESPACE}"
    log "Done. Use: ./deploy.sh logs"
}

cmd_full_deploy() {
    cmd_build
    cmd_deploy_bot
    echo ""
    log "Full deployment complete!"
    log ""
    log "Next steps:"
    log "  ./deploy.sh status           # Check cluster status"
    log "  ./deploy.sh logs             # View bot logs"
    log "  ./deploy.sh sync-up          # Push local data to cluster"
    log "  ./deploy.sh optimizer        # Launch genetic optimizer"
    log "  ./deploy.sh update-config    # Push config changes"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
CMD="${1:-}"

case "$CMD" in
    build)          cmd_build ;;
    bot)            cmd_deploy_bot ;;
    optimizer)      cmd_deploy_optimizer ;;
    optimizer-stop) cmd_optimizer_stop ;;
    ray-status)      cmd_ray_status ;;
    kuberay-install) cmd_kuberay_install ;;
    kuberay-remove)  cmd_kuberay_remove ;;
    sync-up)         cmd_sync_up ;;
    sync-down)      cmd_sync_down ;;
    logs)           cmd_logs ;;
    logs-optimizer) cmd_logs_optimizer ;;
    status)         cmd_status ;;
    shell)          cmd_shell ;;
    stop)           cmd_stop ;;
    start)          cmd_start ;;
    stop)           cmd_stop ;;
    destroy)        cmd_destroy ;;
    setup-ssh)      cmd_setup_ssh ;;
    update-config)  cmd_update_config ;;
    "")             cmd_full_deploy ;;
    *)
        echo "Usage: $0 {build|bot|optimizer|optimizer-stop|ray-status|kuberay-install|kuberay-remove|sync-up|sync-down|logs|logs-optimizer|status|shell|start|stop|destroy|setup-ssh|update-config}"
        echo ""
        echo "Commands:"
        echo "  (none)          Full deploy: build image + deploy bot"
        echo "  build           Build Docker image and push to local registry"
        echo "  bot             Deploy/restart the trading bot"
        echo "  optimizer       Launch optimizer with Ray cluster (multi-node)"
        echo "  optimizer-stop  Delete optimizer job and Ray cluster"
        echo "  ray-status      Show Ray cluster status and resources"
        echo "  kuberay-install Install KubeRay operator via Helm"
        echo "  kuberay-remove  Uninstall KubeRay operator and namespace"
        echo "  sync-up         Sync local data files TO the cluster"
        echo "  sync-down       Sync cluster data files BACK to local"
        echo "  logs            Tail bot logs"
        echo "  logs-optimizer  Tail optimizer logs"
        echo "  status          Show cluster/pod status"
        echo "  shell           Open a shell in the bot pod"
        echo "  start           Scale bot to 1 replica"
        echo "  stop            Scale bot to 0 replicas"
        echo "  destroy         Delete everything (namespace + resources)"
        echo "  setup-ssh       Set up SSH key auth (run once)"
        echo "  update-config   Push config.py changes and restart bot"
        exit 1
        ;;
esac
