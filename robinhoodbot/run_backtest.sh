#!/bin/bash
# Run backtest with sample data
#
# Usage:
#   ./run_backtest.sh                    # Generate sample data and run backtest
#   ./run_backtest.sh --no-generate      # Run backtest with existing sample data
#   ./run_backtest.sh --symbols AAPL,MSFT --days 180

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
SYMBOLS="AAPL,MSFT,GOOGL,AMZN,NVDA"
DAYS=365
CAPITAL=10000
SAMPLE_DIR="sample_data"
GENERATE=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-generate)
            GENERATE=false
            shift
            ;;
        --symbols)
            SYMBOLS="$2"
            shift 2
            ;;
        --days)
            DAYS="$2"
            shift 2
            ;;
        --capital)
            CAPITAL="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "=============================================="
echo "RobinhoodBot Backtesting System"
echo "=============================================="
echo ""

# Generate sample data if needed
if [ "$GENERATE" = true ]; then
    echo "Step 1: Generating sample data..."
    python sample_data_generator.py \
        --symbols "$SYMBOLS" \
        --days "$DAYS" \
        --output "$SAMPLE_DIR" \
        --scenario mixed \
        --seed 42
    echo ""
fi

# Run backtest
echo "Step 2: Running backtest..."
python backtest.py \
    --symbols "$SYMBOLS" \
    --days "$DAYS" \
    --capital "$CAPITAL" \
    --sample-data "$SAMPLE_DIR" \
    --output backtest_result.json

echo ""
echo "=============================================="
echo "Backtest complete! Results saved to backtest_result.json"
echo "=============================================="
