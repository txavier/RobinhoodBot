#!/bin/bash
# ============================================================================
# kernel-upgrade.sh — Rolling Kernel Upgrade for K8s Cluster
# ============================================================================
# Upgrades the Linux kernel on all (or specified) cluster nodes with proper
# cordon/drain/reboot/uncordon sequencing to maintain cluster availability.
#
# Two-phase approach:
#   Phase 1: Install kernel on all nodes (no reboot, no disruption)
#   Phase 2: Rolling reboot — one node at a time with cordon+drain
#
# Usage:
#   ./kernel-upgrade.sh                    # Upgrade all worker nodes
#   ./kernel-upgrade.sh --all              # Include control-plane node
#   ./kernel-upgrade.sh --nodes "d7e327 d7e2a5"  # Specific nodes only
#   ./kernel-upgrade.sh --install-only     # Install kernel, skip reboot
#   ./kernel-upgrade.sh --reboot-only      # Skip install, just rolling reboot
#   ./kernel-upgrade.sh --dry-run          # Show what would happen
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEBUG_IMAGE="ubuntu"
DRAIN_TIMEOUT=120          # seconds to wait for drain
READY_TIMEOUT=300          # seconds to wait for node Ready after reboot
SETTLE_TIME=30             # seconds between nodes for pods to reschedule
BLACKLIST_LINES=(
    '"linux-image-*";'
    '"linux-headers-*";'
    '"linux-modules-*";'
    '"linux-base";'
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
INCLUDE_CONTROL_PLANE=false
SPECIFIC_NODES=""
INSTALL_ONLY=false
REBOOT_ONLY=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)              INCLUDE_CONTROL_PLANE=true; shift ;;
        --nodes)            SPECIFIC_NODES="$2"; shift 2 ;;
        --install-only)     INSTALL_ONLY=true; shift ;;
        --reboot-only)      REBOOT_ONLY=true; shift ;;
        --dry-run)          DRY_RUN=true; shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo "[kernel-upgrade] $(date '+%H:%M:%S') $*"; }
warn() { echo "[kernel-upgrade] $(date '+%H:%M:%S') WARNING: $*" >&2; }
die()  { echo "[kernel-upgrade] $(date '+%H:%M:%S') ERROR: $*" >&2; exit 1; }

# Run a command on a node, wait for completion, print output
exec_on_node() {
    local node="$1"
    local cmd="$2"
    local max_wait="${3:-120}"

    # Base64-encode the command to avoid all quoting/escaping issues
    local b64cmd
    b64cmd=$(printf '%s' "$cmd" | base64 -w0)

    # Create the debug pod — decode and pipe into bash on the host
    local create_output
    create_output=$(kubectl debug "node/${node}" --image="${DEBUG_IMAGE}" \
        -- bash -c "echo ${b64cmd} | base64 -d | chroot /host bash" 2>&1)

    # Extract pod name from "Creating debugging pod <name>"
    local pod_name
    pod_name=$(echo "$create_output" | grep -oP 'pod \K\S+' | head -1)
    if [[ -z "$pod_name" ]]; then
        warn "Could not determine debug pod name for ${node}"
        echo "$create_output"
        return 1
    fi

    # Wait for pod to complete
    local elapsed=0
    while [[ $elapsed -lt $max_wait ]]; do
        local phase
        phase=$(kubectl get pod "$pod_name" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
        case "$phase" in
            Succeeded|Failed)
                kubectl logs "$pod_name" 2>/dev/null
                kubectl delete pod "$pod_name" --force --grace-period=0 >/dev/null 2>&1 || true
                [[ "$phase" == "Succeeded" ]] && return 0 || return 1
                ;;
        esac
        sleep 3
        elapsed=$((elapsed + 3))
    done
    warn "Timed out after ${max_wait}s waiting for pod ${pod_name} on ${node}"
    kubectl delete pod "$pod_name" --force --grace-period=0 >/dev/null 2>&1 || true
    return 1
}

wait_for_ready() {
    local node="$1"
    local deadline=$((SECONDS + READY_TIMEOUT))

    log "  Waiting for ${node} to become Ready..."
    while [[ $SECONDS -lt $deadline ]]; do
        local status
        status=$(kubectl get node "$node" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "Unknown")
        if [[ "$status" == "True" ]]; then
            log "  ${node} is Ready"
            return 0
        fi
        sleep 5
    done
    warn "${node} did not become Ready within ${READY_TIMEOUT}s"
    return 1
}

# ---------------------------------------------------------------------------
# Discover nodes
# ---------------------------------------------------------------------------
if [[ -n "$SPECIFIC_NODES" ]]; then
    NODES=()
    for partial in $SPECIFIC_NODES; do
        # Allow partial names like "d7e327" → "k8s-node-d7e327"
        full=$(kubectl get nodes --no-headers | awk '{print $1}' | grep "$partial" || true)
        if [[ -z "$full" ]]; then
            die "No node matching '${partial}'"
        fi
        NODES+=("$full")
    done
else
    readarray -t NODES < <(kubectl get nodes --no-headers | awk '{print $1}')
    if [[ "$INCLUDE_CONTROL_PLANE" == "false" ]]; then
        NODES_FILTERED=()
        for n in "${NODES[@]}"; do
            # Check if node has the control-plane role label (key exists, value may be empty)
            if kubectl get node "$n" -o jsonpath='{.metadata.labels}' 2>/dev/null | grep -q 'node-role.kubernetes.io/control-plane'; then
                log "Skipping control-plane node: ${n} (use --all to include)"
            else
                NODES_FILTERED+=("$n")
            fi
        done
        NODES=("${NODES_FILTERED[@]}")
    fi
fi

if [[ ${#NODES[@]} -eq 0 ]]; then
    die "No nodes selected for upgrade"
fi

log "Target nodes: ${NODES[*]}"
log "Options: install_only=${INSTALL_ONLY}, reboot_only=${REBOOT_ONLY}, dry_run=${DRY_RUN}"

# Show current kernel versions
log ""
log "Current kernel versions:"
for node in "${NODES[@]}"; do
    ver=$(kubectl get node "$node" -o jsonpath='{.status.nodeInfo.kernelVersion}')
    log "  ${node}: ${ver}"
done
log ""

if [[ "$DRY_RUN" == "true" ]]; then
    log "[DRY RUN] Would upgrade kernel on: ${NODES[*]}"
    log "[DRY RUN] Phase 1: Remove blacklist → apt update → install kernel → restore blacklist"
    log "[DRY RUN] Phase 2: For each node: cordon → drain → reboot → wait Ready → uncordon"
    exit 0
fi

# =========================================================================
# Phase 1: Install kernel on all nodes (no reboot, no disruption)
# =========================================================================
if [[ "$REBOOT_ONLY" == "false" ]]; then
    log "═══════════════════════════════════════════════════════════"
    log "Phase 1: Installing kernel updates on all nodes"
    log "═══════════════════════════════════════════════════════════"

    for node in "${NODES[@]}"; do
        log ""
        log "── ${node}: Removing kernel blacklist ──"
        exec_on_node "$node" \
            'sed -i "/linux-image/d; /linux-headers/d; /linux-modules/d; /linux-base/d" /etc/apt/apt.conf.d/50unattended-upgrades && echo "Blacklist removed"' \
            30

        log "── ${node}: Running apt update + kernel install ──"
        exec_on_node "$node" \
            'export DEBIAN_FRONTEND=noninteractive && apt-get update -qq && apt-get install -y -qq linux-image-generic linux-headers-generic && echo "Kernel installed"' \
            300

        log "── ${node}: Restoring kernel blacklist ──"
        # Insert blacklist lines back after "kubectl"; line
        read -r -d '' bl_insert <<'BLCMD' || true
sed -i '/"kubectl";/a\    "linux-image-*";\n    "linux-headers-*";\n    "linux-modules-*";\n    "linux-base";' /etc/apt/apt.conf.d/50unattended-upgrades && echo "Blacklist restored"
BLCMD
        exec_on_node "$node" "$bl_insert" 30

        log "── ${node}: Kernel install complete ──"
    done

    log ""
    log "Phase 1 complete — kernel packages installed on all nodes"
fi

# =========================================================================
# Phase 2: Rolling reboot — one node at a time
# =========================================================================
if [[ "$INSTALL_ONLY" == "false" ]]; then
    log ""
    log "═══════════════════════════════════════════════════════════"
    log "Phase 2: Rolling reboot (cordon → drain → reboot → uncordon)"
    log "═══════════════════════════════════════════════════════════"

    for node in "${NODES[@]}"; do
        log ""
        log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log "  Upgrading ${node}"
        log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # Cordon
        log "  Cordoning ${node}..."
        kubectl cordon "$node"

        # Drain
        log "  Draining ${node} (timeout ${DRAIN_TIMEOUT}s)..."
        kubectl drain "$node" \
            --ignore-daemonsets \
            --delete-emptydir-data \
            --force \
            --timeout="${DRAIN_TIMEOUT}s" \
            2>&1 | tail -3 || warn "  Drain timed out — reboot will handle remaining pods"

        # Reboot — use systemd-run to schedule reboot that survives container exit
        log "  Rebooting ${node}..."
        exec_on_node "$node" 'systemd-run --on-active=3 /sbin/reboot && echo "Reboot scheduled"' 15 || true

        # Give the reboot time to initiate before polling
        sleep 15

        # Wait for the node to go NotReady (confirms reboot started)
        log "  Waiting for ${node} to go NotReady..."
        reboot_start=$SECONDS
        while [[ $((SECONDS - reboot_start)) -lt 120 ]]; do
            status=$(kubectl get node "$node" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "Unknown")
            if [[ "$status" != "True" ]]; then
                log "  ${node} is NotReady (reboot in progress)"
                break
            fi
            sleep 5
        done

        # Wait for Ready
        if ! wait_for_ready "$node"; then
            warn "Node ${node} didn't come back! Stopping to avoid further disruption."
            warn "Remaining nodes NOT upgraded. Fix ${node} and re-run with:"
            warn "  ./kernel-upgrade.sh --reboot-only --nodes \"remaining-nodes\""
            exit 1
        fi

        # Uncordon
        log "  Uncordoning ${node}..."
        kubectl uncordon "$node"

        # Verify new kernel
        new_ver=$(kubectl get node "$node" -o jsonpath='{.status.nodeInfo.kernelVersion}')
        log "  ${node} now running kernel: ${new_ver}"

        # Let pods settle before next node
        if [[ "$node" != "${NODES[-1]}" ]]; then
            log "  Waiting ${SETTLE_TIME}s for pods to settle..."
            sleep "$SETTLE_TIME"
        fi
    done

    log ""
    log "═══════════════════════════════════════════════════════════"
    log "Phase 2 complete — all nodes rebooted"
    log "═══════════════════════════════════════════════════════════"
fi

# =========================================================================
# Summary
# =========================================================================
log ""
log "Final kernel versions:"
for node in "${NODES[@]}"; do
    ver=$(kubectl get node "$node" -o jsonpath='{.status.nodeInfo.kernelVersion}')
    log "  ${node}: ${ver}"
done
log ""
log "Kernel upgrade complete."
