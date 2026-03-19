#!/bin/bash
# Monitor RobinhoodBot logs in Kubernetes with auto-reconnect.
# Usage: ./monitor.sh [--history N]
#   --history N  Show last N lines on first connect (default: 50)

TAIL_LINES=50
if [[ "$1" == "--history" && -n "$2" ]]; then
    TAIL_LINES="$2"
fi

echo "Monitoring robinhoodbot logs (Ctrl+C to stop)..."
FIRST=true
while true; do
    if $FIRST; then
        kubectl logs -f deployment/robinhoodbot -n robinhoodbot --tail="$TAIL_LINES"
        FIRST=false
    else
        kubectl logs -f deployment/robinhoodbot -n robinhoodbot --tail=0
    fi
    echo "--- Connection lost. Reconnecting... ---"
    sleep 1
done
