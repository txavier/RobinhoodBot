# RobinhoodBot Backtesting System

A comprehensive backtesting framework for testing the RobinhoodBot trading strategy against historical data.

## Features

- **Simulated Portfolio**: Track virtual cash and positions without risking real capital
- **Historical Data Caching**: Automatically cache historical price data to reduce API calls
- **Performance Metrics**: Calculate returns, Sharpe ratio, win rate, drawdown, and more
- **Trade Tracking**: Record all trades with buy/sell reasons
- **Visualization**: Generate equity curves, trade analysis charts, and performance plots
- **Flexible Configuration**: Test any symbols, date ranges, and starting capital

## Quick Start

### 1. Run a Default Backtest

Test 10 popular tech stocks over the past year with $10,000 starting capital:

```bash
cd robinhoodbot
python run_backtest.py
```

### 2. Test Custom Symbols

```bash
python run_backtest.py AAPL MSFT GOOGL AMZN TSLA
```

### 3. Custom Date Range

```bash
python run_backtest.py --start 2024-01-01 --end 2024-12-31
```

### 4. Programmatic Usage

```python
from run_backtest import run_simple_backtest, run_custom_backtest

# Simple backtest
metrics = run_simple_backtest(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2024-01-01',
    end_date='2024-12-31',
    initial_cash=10000.0
)

# Custom backtest with more control
from run_backtest import StrategyBacktest

backtest = StrategyBacktest(
    symbols=['AAPL', 'MSFT'],
    start_date='2024-01-01',
    end_date='2024-12-31',
    initial_cash=20000.0
)

# Customize strategy parameters
backtest.n1_buy = 15  # Short-term SMA
backtest.n2_buy = 40  # Long-term SMA
backtest.take_profit_threshold = 3.0  # 3% profit threshold

metrics = backtest.run()
```

## Output Files

After running a backtest, the following files are generated:

1. **backtest_report.txt** - Detailed text report with:
   - Portfolio performance (returns, drawdown, Sharpe ratio)
   - Trading statistics (win rate, average profit/loss)
   - Complete trade history

2. **backtest_equity.csv** - Portfolio value over time:
   - Date, total value, cash, positions value

3. **backtest_trades.csv** - All trades executed:
   - Date, type (BUY/SELL), symbol, shares, price, profit, reasons

4. **backtest_cache.json** - Cached historical price data

## Visualization

Generate charts from backtest results:

```python
from backtest_viz import generate_all_plots

# Generate all visualization plots
generate_all_plots(show_plots=True)
```

Or generate individual plots:

```python
from backtest_viz import plot_equity_curve, plot_trade_analysis, plot_returns_distribution

plot_equity_curve(show_plot=True)
plot_trade_analysis(show_plot=True)
plot_returns_distribution(show_plot=True)
```

Generated plots:
- **equity_curve.png** - Portfolio value and cash/invested breakdown
- **trade_analysis.png** - P/L per trade, cumulative profit, win/loss pie chart, sell reasons
- **returns_distribution.png** - Histogram of returns in dollars and percentages

## Performance Metrics Explained

### Portfolio Metrics
- **Total Return**: Overall gain/loss in dollars and percentage
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Final Cash**: Uninvested cash remaining
- **Open Positions**: Number of stocks still held

### Trading Metrics
- **Win Rate**: Percentage of profitable trades
- **Average Win**: Average profit on winning trades
- **Average Loss**: Average loss on losing trades
- **Avg Profit/Trade**: Average profit across all trades

## Strategy Logic

The backtest uses the same strategy from main.py:

### Buy Signals (Golden Cross)
- Short-term SMA (default 20) crosses above long-term SMA (default 50)
- Price is still rising after the cross
- Price is higher than 5 hours ago (in live trading)

### Sell Signals
- **Death Cross**: Short-term SMA crosses below long-term SMA
- **Take Profit**: Position gain exceeds threshold (default 2.15%)
- **Sudden Drop**: Rapid price decline (requires intraday data)
- **Profit Before EOD**: Take profits before end of day (requires time-of-day data)

## Limitations

1. **Data Granularity**: Backtests use daily data by default. Intraday signals (sudden drops, time-of-day checks) are approximated.

2. **Look-Ahead Bias**: The system tries to prevent look-ahead bias, but some indicators may use future data unintentionally.

3. **Slippage**: Assumes perfect execution at closing prices. Real trades may have worse fills.

4. **Market Conditions**: Past performance doesn't guarantee future results. Market regime changes can affect strategy performance.

5. **Data Availability**: Limited to historical data available from Robinhood API (typically 5 years max).

## Advanced Usage

### Clear Cache

If you want fresh data:

```python
from backtest import HistoricalDataCache

cache = HistoricalDataCache()
cache.clear_cache()
```

### Custom Portfolio Simulator

```python
from backtest import SimulatedPortfolio

portfolio = SimulatedPortfolio(initial_cash=50000, commission=1.0)

# Execute trades
portfolio.buy('AAPL', 10, 150.0, '2024-01-15', reason='golden_cross')
portfolio.sell('AAPL', 160.0, '2024-02-15', reason='take_profit')

# Get results
print(portfolio.trade_history)
```

### Custom Metrics

```python
from backtest import BacktestEngine

engine = BacktestEngine(initial_cash=10000)
# ... run backtest ...

metrics = engine.calculate_performance_metrics()
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
```

## Tips for Better Backtesting

1. **Test Multiple Periods**: Run backtests across different market conditions (bull, bear, sideways)

2. **Parameter Optimization**: Try different SMA periods, profit thresholds, etc.

3. **Diversification**: Test with different stock universes (tech, healthcare, value stocks)

4. **Walk-Forward Testing**: Use rolling windows to simulate real-world conditions

5. **Compare to Benchmark**: Compare results to buy-and-hold SPY or individual stocks

## Example Output

```
================================================================================
BACKTEST RESULTS
================================================================================

PORTFOLIO PERFORMANCE:
  Initial Value:        $10,000.00
  Final Value:          $12,345.67
  Total Return:         $2,345.67 (23.46%)
  Max Drawdown:         -8.32%
  Sharpe Ratio:         1.45
  Final Cash:           $2,100.50
  Open Positions:       3

TRADING STATISTICS:
  Total Trades:         45
  Buy Orders:           23
  Sell Orders:          22
  Winning Trades:       15
  Losing Trades:        7
  Win Rate:             68.18%
  Avg Win:              $215.30
  Avg Loss:             -$95.20
  Avg Profit/Trade:     $106.62
```

## Troubleshooting

### "No data available for symbol"
- Check if symbol is valid
- Try a different date range
- Clear cache and retry

### "Insufficient funds"
- Reduce number of symbols
- Increase initial_cash
- Adjust position sizing

### API Rate Limits
- Use cached data when possible
- Add delays between requests
- Test with fewer symbols first

## Contributing

To add new features:
1. Strategy modifications go in `run_backtest.py`
2. Core engine features go in `backtest.py`
3. Visualization features go in `backtest_viz.py`

## License

Same license as RobinhoodBot main project.
