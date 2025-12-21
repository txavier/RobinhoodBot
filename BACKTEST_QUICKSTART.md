# Backtesting Quick Start Guide

## Installation

No additional installation needed! The backtesting system uses the same dependencies as RobinhoodBot.

## 5-Minute Quick Start

### 1. Run Your First Backtest

```bash
cd robinhoodbot
python run_backtest.py
```

This tests 10 popular tech stocks over the past year with $10,000 starting capital.

### 2. View Results

Check these files:
- `backtest_report.txt` - Detailed performance report
- `backtest_equity.csv` - Portfolio value over time
- `backtest_trades.csv` - All trades executed

### 3. Generate Charts

```python
python backtest_viz.py
```

This creates:
- `equity_curve.png` - Portfolio growth chart
- `trade_analysis.png` - Win/loss analysis
- `returns_distribution.png` - Profit/loss distribution

## Common Use Cases

### Test Specific Stocks

```bash
python run_backtest.py AAPL MSFT GOOGL
```

### Test Specific Time Period

```bash
python run_backtest.py --start 2024-01-01 --end 2024-06-30
```

### Run Interactive Examples

```bash
python backtest_examples.py
```

Choose from:
1. Simple backtest
2. Custom date range
3. Parameter tuning
4. Sector comparison
5. Walk-forward testing
6. Visualization

## Understanding Results

### Key Metrics

- **Total Return %**: How much your portfolio grew
- **Win Rate %**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns (higher is better, >1 is good)
- **Max Drawdown %**: Worst peak-to-trough decline

### Good vs Bad Performance

✅ **Good Signs:**
- Positive total return
- Win rate > 50%
- Sharpe ratio > 1.0
- Max drawdown < 20%
- Consistent profits across periods

❌ **Warning Signs:**
- Negative total return
- Win rate < 40%
- Sharpe ratio < 0
- Max drawdown > 30%
- Large variation in quarterly returns

## Next Steps

1. **Read Full Documentation**: See [BACKTESTING.md](BACKTESTING.md)
2. **Tune Parameters**: Adjust SMA periods and profit thresholds
3. **Test Different Stocks**: Try various sectors and market caps
4. **Compare Periods**: Test bull vs bear markets
5. **Walk-Forward Test**: Validate across multiple time windows

## Troubleshooting

### "No data available"
- Check internet connection
- Verify symbol is valid
- Try different date range
- Clear cache: delete `backtest_cache.json`

### "Insufficient funds"
- Increase `initial_cash`
- Reduce number of symbols
- Check position sizing

### Slow Performance
- Data is cached after first run
- Reduce date range
- Test fewer symbols

## Tips

1. **Start Small**: Test with 3-5 stocks first
2. **Use Cache**: Data is cached to speed up subsequent runs
3. **Test Multiple Scenarios**: Bull, bear, and sideways markets
4. **Compare to Benchmark**: How does it compare to buy-and-hold?
5. **Don't Overfit**: Parameters that work in backtest may not work live

## Getting Help

- Read [BACKTESTING.md](BACKTESTING.md) for detailed documentation
- Check `backtest_examples.py` for code examples
- Review generated `backtest_report.txt` for detailed analysis

Happy backtesting!
