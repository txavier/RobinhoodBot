#!/bin/bash
# =============================================================================
# Robust Optimizer Runner
# =============================================================================
# Runs the genetic optimizer with protections against:
#   - Terminal SIGTERM (VS Code, GNOME Terminal, SSH disconnect)
#   - stdout pipe breaks (tee dying)
#   - Python output buffering
#
# Usage:
#   ./run_optimizer.sh [optimizer args...]
#
# Examples:
#   # 50-stock overnight run with built-in logging:
#   ./run_optimizer.sh --symbols AAPL,MSFT,NVDA,GOOGL,AMZN --generations 20 --population 30 --resume --real-data
#
#   # Custom log file:
#   LOG_FILE=/tmp/my_run.log ./run_optimizer.sh --symbols AAPL,MSFT --generations 10 --resume --real-data
#
# The script:
#   1. Uses setsid to create a new session (immune to terminal SIGTERM)
#   2. Uses --log-file for reliable file logging (no tee pipe to break)
#   3. Uses PYTHONUNBUFFERED=1 for real-time stdout
#   4. Logs start/end times and PID for monitoring
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Default log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_FILE:-/tmp/optimizer_${TIMESTAMP}.log}"

echo "============================================="
echo "  Genetic Optimizer Runner"
echo "============================================="
echo "  Log file:    $LOG_FILE"
echo "  Started:     $(date)"
echo "  Arguments:   $@"
echo "============================================="

# Activate venv if it exists (check common locations)
for venv_path in "$SCRIPT_DIR/../.venv" "$SCRIPT_DIR/.venv" "$HOME/dev/.venv"; do
    if [ -f "$venv_path/bin/activate" ]; then
        source "$venv_path/bin/activate"
        break
    fi
done

# Run with setsid (new session = immune to terminal signals)
# PYTHONUNBUFFERED=1 ensures real-time stdout
# --log-file writes directly to file (no tee needed)
echo "  PID tracking: see $LOG_FILE"
echo ""

PYTHONUNBUFFERED=1 setsid python3 genetic_optimizer_intraday.py \
    --log-file "$LOG_FILE" \
    --resume \
    "$@" &

OPTIMIZER_PID=$!
echo "  Optimizer PID: $OPTIMIZER_PID (session leader, immune to terminal SIGTERM)"
echo "  Monitor with:  tail -f $LOG_FILE"
echo "  Check status:  ps -p $OPTIMIZER_PID -o pid,etime,pcpu,pmem"
echo "  Stop cleanly:  kill $OPTIMIZER_PID  (saves checkpoint)"
echo "  Force stop:    kill -9 $OPTIMIZER_PID"
echo ""

# Wait for it (in foreground, so Ctrl+C works but terminal close won't kill it)
wait $OPTIMIZER_PID 2>/dev/null
EXIT_CODE=$?

echo ""
echo "============================================="
echo "  Optimizer finished at: $(date)"
echo "  Exit code: $EXIT_CODE"
echo "  Full log:  $LOG_FILE"
echo "============================================="

exit $EXIT_CODE
