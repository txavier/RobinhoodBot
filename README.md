# RobinhoodBot
Trading bot for Robinhood accounts

For more info:
https://medium.com/@kev.guo123/building-a-robinhood-stock-trading-bot-8ee1b040ec6a


5/1/19: Since Robinhood has updated it's API, you now have to enter a 2 factor authentication code whenever you run the script. To do this, go to the Robinhood mobile app and enable two factor authentication in your settings. You will now receive an SMS code when you run the script, which you have to enter into the script.



This project supports Python 3.7+


To Install:

```bash
git clone https://github.com/txavier/RobinhoodBot.git
cd RobinhoodBot/
pip install -r requirements.txt
cp config.py.sample config.py # add auth info and watchlist name to monitor after copying
```

To Run:
In RobinHood create a watchlist named "Exclusion".  This will be the watchlist that you will use to tell the bot to ignore the stock tickers contained within.

```python
cd RobinboodBot/robinhoodbot (If outside of root directory)
python3 main.py
```

To loop: 1 once an hour

```python
cd RobinboodBot/robinhoodbot (If outside of root directory)
./run.sh # uses bash
```

# Changes to robin_stocks library
 - Added @cache to def get_name_by_symbol(symbol):
 - Added @cache to def get_name_by_url(url):
 - Added @cache to def get_symbol_by_url(url):
 - Added None Parameter to order
Afterwards, be sure to run /> pip install . 

# VENV
## In the event you want to use Venv instead of anaconda, activate it with the follwing command.
venv:
/home/theo/dev/.venv/bin/activate
~/> source dev/.venv/bin/activate


# Backtesting

The bot includes a comprehensive backtesting module to test the trading strategy against historical data before risking real money.

## Quick Start

```bash
cd robinhoodbot/

# Generate sample data and run backtest
./run_backtest.sh

# Or run with custom parameters
./run_backtest.sh --symbols AAPL,MSFT,GOOGL --days 180 --capital 20000
```

## Backtest Module

Run backtests directly with the Python script:

```bash
# Basic backtest with default symbols (AAPL,MSFT,GOOGL,AMZN,NVDA)
python backtest.py

# Specify symbols and use sample data (no API required)
python backtest.py --symbols AAPL,MSFT --sample-data sample_data/

# Backtest with explicit date range
python backtest.py --symbols AAPL,MSFT,GOOGL --start 2025-01-01 --end 2025-12-31

# Backtest last 200 days with custom capital
python backtest.py --symbols AAPL --days 200 --capital 25000

# Custom strategy parameters (shorter SMAs, tighter stop-loss)
python backtest.py --symbols AAPL --short-sma 10 --long-sma 30 --stop-loss 3 --take-profit 1.0

# Quiet mode - suppress trade-by-trade output
python backtest.py --symbols AAPL,MSFT --sample-data sample_data/ --quiet

# Save results to custom file
python backtest.py --symbols AAPL --output my_backtest_results.json

# Full example with all options
python backtest.py \
    --symbols AAPL,MSFT,GOOGL \
    --start 2025-01-01 \
    --end 2025-10-31 \
    --capital 10000 \
    --short-sma 20 \
    --long-sma 50 \
    --stop-loss 5 \
    --take-profit 0.7 \
    --sample-data sample_data/ \
    --output backtest_result.json
```

### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--symbols` | `-s` | Comma-separated stock symbols to test | AAPL,MSFT,GOOGL,AMZN,NVDA |
| `--start` | `-st` | Start date in YYYY-MM-DD format | Calculated from --days |
| `--end` | `-e` | End date in YYYY-MM-DD format | Today |
| `--days` | `-d` | Number of days to backtest (used if --start not specified) | 365 |
| `--capital` | `-c` | Initial capital in dollars | 10000 |
| `--short-sma` | | Short-term Simple Moving Average period | 20 |
| `--long-sma` | | Long-term Simple Moving Average period | 50 |
| `--stop-loss` | | Stop loss percentage (sells if loss exceeds this %) | 5 (from config) |
| `--take-profit` | | Take profit percentage (sells if gain exceeds this %) | 0.7 (from config) |
| `--sample-data` | | Directory containing sample data files (bypasses API) | None |
| `--no-api` | | Disable API calls, use only cached/sample data | False |
| `--output` | `-o` | Output JSON file for results | backtest_result.json |
| `--quiet` | `-q` | Suppress verbose trade-by-trade output | False |

## Sample Data Generator

Generate synthetic stock data for testing without requiring a Robinhood login:

```bash
# Generate mixed scenario data (bullish, bearish, golden cross patterns)
python sample_data_generator.py --symbols AAPL,MSFT,GOOGL --days 365 --scenario mixed

# Generate golden cross scenarios (good for testing buy signals)
python sample_data_generator.py --symbols TEST1,TEST2 --scenario golden_cross

# Generate death cross scenarios (good for testing sell signals)
python sample_data_generator.py --symbols TEST1,TEST2 --scenario death_cross

# Use a seed for reproducible results
python sample_data_generator.py --symbols AAPL --seed 42
```

### Data Scenarios

| Scenario | Description |
|----------|-------------|
| `random` | Random trends (bullish, bearish, sideways, volatile) |
| `golden_cross` | Data designed to trigger golden cross buy signals |
| `death_cross` | Data designed to trigger death cross sell signals |
| `mixed` | Combination of all scenarios |

## Backtest Results

Results are saved to JSON and include:

- **Financial Summary**: Initial/final capital, total return
- **Performance Metrics**: Max drawdown, Sharpe ratio
- **Trading Statistics**: Win rate, average win/loss, profit factor
- **Trade History**: Complete list of all trades with timestamps and reasons

Example output:
```
============================================================
BACKTEST RESULTS
============================================================

📅 Period: 2024-01-01 to 2024-12-31
📈 Symbols: AAPL, MSFT, GOOGL

💰 FINANCIAL SUMMARY
   Initial Capital:  $    10,000.00
   Final Capital:    $    11,234.56
   Total Return:     $    +1,234.56 (+12.35%)

📊 PERFORMANCE METRICS
   Max Drawdown:     $       567.89 (5.67%)
   Sharpe Ratio:           0.8234

🎯 TRADING STATISTICS
   Total Trades:               24
   Winning Trades:             15
   Losing Trades:               9
   Win Rate:                62.50%
   Avg Win:          $       123.45
   Avg Loss:         $        67.89
   Profit Factor:           2.73
```

## Using Backtest Results

The backtest module uses the same trading logic as the main bot:
- Golden Cross (SMA 20/50) for buy signals
- Death Cross for sell signals
- Stop-loss protection
- Take-profit targets
- Sudden drop detection

Review backtest results to:
1. Validate strategy parameters before live trading
2. Understand expected win rates and drawdowns
3. Test different SMA periods and risk settings
4. Evaluate performance across different market conditions

# Genetic Algorithm Optimizer

The bot includes a genetic algorithm that automatically evolves trading parameters to maximize profits and minimize losses.

## Quick Start

```bash
cd robinhoodbot/

# Generate sample data first
python sample_data_generator.py --symbols AAPL,MSFT,GOOGL --days 200 --scenario mixed --seed 42

# Run genetic optimization with sample data
python genetic_optimizer.py --symbols AAPL,MSFT,GOOGL --sample-data sample_data/ --generations 15 --population 20
```

## How It Works

The genetic algorithm treats each set of trading parameters as a "chromosome" and evolves them over generations:

1. **Initialization**: Creates a population of random parameter combinations
2. **Evaluation**: Runs backtests to calculate fitness (profit, Sharpe ratio, win rate, etc.)
3. **Selection**: Tournament selection picks the fittest individuals for breeding
4. **Crossover**: Parents exchange parameters to create offspring
5. **Mutation**: Random changes introduce genetic diversity
6. **Elitism**: Top performers are preserved unchanged
7. **Repeat**: Process continues for specified number of generations

## Parameters Being Optimized

| Parameter | Range | Description |
|-----------|-------|-------------|
| `short_sma` | 5-50 | Short-term Simple Moving Average period |
| `long_sma` | 20-200 | Long-term Simple Moving Average period |
| `golden_cross_days` | 1-10 | Days to look back for golden cross signals |
| `stop_loss_pct` | 1-15% | Stop loss percentage |
| `take_profit_pct` | 0.3-5% | Take profit percentage |
| `position_size_pct` | 5-30% | Maximum position size as % of portfolio |

## Fitness Function

The fitness score is a weighted combination of multiple metrics:

| Metric | Weight | Description |
|--------|--------|-------------|
| Total Return | 30% | Overall profit/loss percentage |
| Sharpe Ratio | 25% | Risk-adjusted return |
| Win Rate | 15% | Percentage of winning trades |
| Profit Factor | 15% | Gross profits / gross losses |
| Max Drawdown | 15% | Penalty for large drawdowns |

## Command Line Options

```bash
# Basic usage with defaults
python genetic_optimizer.py --symbols AAPL,MSFT

# Full optimization with all options
python genetic_optimizer.py \
    --symbols AAPL,MSFT,GOOGL \
    --start 2025-01-01 \
    --end 2025-12-31 \
    --capital 10000 \
    --population 30 \
    --generations 25 \
    --mutation-rate 0.15 \
    --crossover-rate 0.7 \
    --sample-data sample_data/ \
    --output my_optimization.json \
    --seed 42

# Quick test run
python genetic_optimizer.py --symbols AAPL --generations 5 --population 10 --sample-data sample_data/
```

### Options Reference

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--symbols` | `-s` | Comma-separated stock symbols | AAPL,MSFT,GOOGL |
| `--start` | `-st` | Start date (YYYY-MM-DD) | Calculated from --days |
| `--end` | `-e` | End date (YYYY-MM-DD) | Today |
| `--days` | `-d` | Days to backtest | 200 |
| `--capital` | `-c` | Initial capital | 10000 |
| `--population` | `-p` | Population size (more = better exploration, slower) | 20 |
| `--generations` | `-g` | Number of generations (more = better convergence) | 15 |
| `--mutation-rate` | `-m` | Probability of mutation (0.0-1.0) | 0.15 |
| `--crossover-rate` | | Probability of crossover (0.0-1.0) | 0.7 |
| `--sample-data` | | Directory with sample data files | None |
| `--output` | `-o` | Output JSON file | genetic_optimization_result.json |
| `--seed` | | Random seed for reproducibility | None |
| `--quiet` | `-q` | Suppress verbose output | False |

## Example Output

```
======================================================================
GENETIC ALGORITHM OPTIMIZER
======================================================================
Symbols: AAPL, MSFT, GOOGL
Period: 2025-01-01 to 2025-12-31
Population: 20
Generations: 15
======================================================================

--- Generation 1/15 ---
  Evaluating 1/20: SMA(20/50) GC:3d SL:5.0% TP:0.7% Pos:15% | Fit:0.4521
  ...

--- Generation 15/15 ---
  Best:  SMA(12/35) GC:4d SL:3.5% TP:1.2% Pos:18% | Fit:0.8234

======================================================================
BEST CONFIGURATION FOUND
======================================================================

# Add these to your config.py:
# SMA Settings
short_sma = 12
long_sma = 35
golden_cross_buy_days = 4

# Risk Management
stop_loss_percent = 3.5
take_profit_percent = 1.2

# Position Sizing
purchase_limit_percentage = 18.0

# Performance Metrics:
# Total Return: +8.45%
# Win Rate: 72.5%
# Sharpe Ratio: 1.2341
# Max Drawdown: 3.21%
# Profit Factor: 2.85
```

## Tips for Best Results

1. **More Data = Better**: Use at least 200 days of historical data
2. **Larger Population**: 30-50 individuals explores more possibilities
3. **More Generations**: 20-30 generations allows better convergence
4. **Multiple Runs**: Run several times with different seeds and compare
5. **Diverse Symbols**: Test across different stocks/sectors
6. **Sample Data First**: Generate good sample data with various scenarios before optimizing

