#!/bin/bash
# Monitor RobinhoodBot logs in Kubernetes with auto-reconnect.
# Connects via SSH to the control-plane node to run kubectl.
# Usage: ./monitor.sh [--history N]
#   --history N  Show last N lines on first connect (default: 50)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load connection info from secrets.env
if [[ -f "${SCRIPT_DIR}/secrets.env" ]]; then
    source "${SCRIPT_DIR}/secrets.env"
else
    echo "ERROR: secrets.env not found."
    exit 1
fi

REMOTE_SSH="${REMOTE_USER}@${REMOTE_HOST}"
NAMESPACE="robinhoodbot"

TAIL_LINES=50
if [[ "$1" == "--history" && -n "$2" ]]; then
    TAIL_LINES="$2"
fi

echo "Monitoring robinhoodbot logs (Ctrl+C to stop)..."
FIRST=true
while true; do
    if $FIRST; then
        ssh -o ConnectTimeout=10 "${REMOTE_SSH}" \
            "kubectl logs -f deployment/robinhoodbot -n ${NAMESPACE} --tail=${TAIL_LINES}"
        FIRST=false
    else
        ssh -o ConnectTimeout=10 "${REMOTE_SSH}" \
            "kubectl logs -f deployment/robinhoodbot -n ${NAMESPACE} --tail=0"
    fi
    echo "--- Connection lost. Reconnecting... ---"
    sleep 1
done
