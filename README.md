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

# Intraday Backtesting (Day Trading Simulation)

The `backtest_intraday.py` module provides **accurate day trading simulation** that matches the actual main.py trading logic. Unlike the basic backtest which uses daily data, this module simulates hourly trading with all the same conditions the real bot uses.

## Why Intraday Backtesting?

The main bot (`main.py`) is a **day trading** application that:
- Uses **hourly SMA crossovers** (not daily)
- Makes **multiple trades per day**
- Has **aggressive take-profit targets** (0.70% default)
- Includes many **conditional filters** before buying/selling

The basic `backtest.py` uses daily data and produces only 3-5 trades over months - not representative of actual day trading performance. The intraday backtester generates hourly data and applies all the same filters as main.py.

## Quick Start

```bash
cd robinhoodbot/

# Basic intraday backtest (60 days, 5 symbols)
python backtest_intraday.py --symbols AAPL,MSFT,GOOGL,NVDA,TSLA --days 60

# Reproducible test with seed
python backtest_intraday.py --symbols AAPL,MSFT --days 30 --seed 42

# Custom parameters
python backtest_intraday.py --symbols AAPL --days 90 --capital 25000 --take-profit 1.0 --stop-loss 3
```

## Main.py Matching Features

The intraday backtester accurately simulates these main.py conditions:

### Buy Filters (all must pass)
| Filter | Description | Config Flag |
|--------|-------------|-------------|
| **Market Uptrend** | 2 of 3 indices (SPY, DIA, NDAQ) must be up today | `--no-market-filter` |
| **No Major Downtrend** | Not in weekly market downtrend | `--no-market-filter` |
| **EOD Filter** | No buying after 1:30pm (inflection protection) | `--no-eod-filter` |
| **Price > 5hr Ago** | Price must be higher than 5 hours ago | `--no-price-5hr-check` |
| **Slope Ordering** | Prioritize stocks with positive momentum (slope > 0.0008) | `--no-slope-ordering` |
| **Price Cap** | Stock price must be under $2,100 | `--no-price-cap` |
| **Day Trade Limit** | Respect pattern day trader limits | Built-in |

### Sell Triggers (in priority order)
| Trigger | Description |
|---------|-------------|
| **Stop Loss** | Sell if loss exceeds stop_loss_percent (default 5%) |
| **Take Profit** | Sell if gain exceeds take_profit_percent (default 0.70%) |
| **Sudden Drop** | Sell if 10% drop in 2hr OR 15% drop in 1hr |
| **Profit Before EOD** | Sell profitable positions after 1:30pm |
| **Death Cross** | Sell when short SMA crosses below long SMA |

### Dynamic SMA Adjustment
The backtest simulates main.py's dynamic SMA periods:
- **Default**: n1=20, n2=50
- **In Downtrend**: n1=14 (more conservative)
- **After Take Profit + Traded Today**: n1=5, n2=7 (more aggressive re-entry)

## Command Line Options

```bash
# Full options example
python backtest_intraday.py \
    --symbols AAPL,MSFT,GOOGL,NVDA,TSLA \
    --days 60 \
    --capital 10000 \
    --short-sma 20 \
    --long-sma 50 \
    --take-profit 0.7 \
    --stop-loss 5 \
    --max-positions 5 \
    --seed 42 \
    --output intraday_results.json

# Simple mode (disable ALL filters - like old backtest)
python backtest_intraday.py --symbols AAPL,MSFT --days 30 --simple-mode

# Disable specific filters
python backtest_intraday.py --symbols AAPL --no-market-filter --no-eod-filter
```

### Options Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--symbols` | Comma-separated stock symbols | AAPL,MSFT,GOOGL |
| `--days` | Trading days to simulate | 30 |
| `--capital` | Initial capital | 10000 |
| `--short-sma` | Short SMA period (hours) | 20 |
| `--long-sma` | Long SMA period (hours) | 50 |
| `--take-profit` | Take profit % | 0.70 (from config) |
| `--stop-loss` | Stop loss % | 5 (from config) |
| `--golden-cross-hours` | Hours to look back for cross | 24 |
| `--max-positions` | Max concurrent positions | 5 |
| `--close-eod` | Close all positions at end of day | False |
| `--seed` | Random seed for reproducibility | None |
| `--output` | Output JSON file | None |
| `--quiet` | Suppress trade-by-trade output | False |

### Filter Toggle Options

| Option | Description |
|--------|-------------|
| `--simple-mode` | Disable ALL main.py filters |
| `--no-market-filter` | Disable market trend filtering |
| `--no-eod-filter` | Disable no-buy-after-1:30pm rule |
| `--no-profit-before-eod` | Disable profit_before_eod sell |
| `--no-price-5hr-check` | Disable price > 5hr ago check |
| `--no-dynamic-sma` | Disable dynamic SMA adjustment |
| `--no-slope-ordering` | Disable slope-based ordering |
| `--no-price-cap` | Disable price cap filter |
| `--price-cap` | Set custom price cap (default: 2100) |

## Example Output

```
======================================================================
INTRADAY BACKTESTING - ACCURATE MAIN.PY SIMULATION
======================================================================
Symbols: AAPL, MSFT, GOOGL, NVDA, TSLA
Trading Days: 60
Initial Capital: $10,000.00
Strategy: Hourly SMA(20/50)
Take Profit: 0.7%
Stop Loss: 5%

⚙️  MAIN.PY MATCHING FEATURES:
   Market Filter: ON
   EOD Filter (no buy after 1:30pm): ON
   Profit Before EOD Sell: ON
   Price > 5hr Ago Check: ON
   Dynamic SMA: ON
   Slope Ordering: ON
   Price Cap ($2100): ON
======================================================================

  🟢 BUY  AAPL: 20sh @ $99.79 [golden_cross, slope=0.2561] Cost: $1995.80
  ✅ SELL AAPL: 20sh @ $100.82 [take_profit] P/L: $+20.60 (+1.03%)
  📊 Day 2025-10-10 [uptrend]: 6 trades, P/L: $+6.53, Portfolio: $10,006.53
  ...

======================================================================
BACKTEST RESULTS (ACCURATE MAIN.PY SIMULATION)
======================================================================

📅 Period: 2025-10-07 to 2026-02-25
📈 Symbols: AAPL, MSFT, GOOGL, NVDA, TSLA

💰 FINANCIAL SUMMARY
   Initial Capital:  $    10,000.00
   Final Capital:    $    10,262.91
   Total Return:     $      +262.91 (+2.63%)

📊 PERFORMANCE METRICS
   Max Drawdown:     $        87.46 (0.87%)
   Sharpe Ratio:             3.0699

🎯 TRADING STATISTICS
   Trading Days:                102
   Total Trades:                142
   Trades/Day:                 0.70
   Winning Trades:               48
   Losing Trades:                23
   Win Rate:                 67.61%
   Avg Win:          $        11.31
   Avg Loss:         $        12.17
   Profit Factor:              1.94

🚫 REJECTED BUY SIGNALS (main.py filters)
   Total Rejected:             2972
   Market Downtrend:            772
   Major Downtrend:              55
   After 1:30pm EOD:            414
   Price < 5hr Ago:               8
   Low Slope:                  1642
   Day Trade Limit:              81

🔧 MAIN.PY MATCHING FEATURES
   market_filter: ✅ ON
   eod_filter: ✅ ON
   profit_before_eod: ✅ ON
   price_5hr_check: ✅ ON
   dynamic_sma: ✅ ON
   slope_ordering: ✅ ON
   price_cap: ✅ ON
======================================================================
```

## Comparing Accurate vs Simple Mode

Run both modes to see the impact of main.py's filters:

```bash
# Accurate mode (all filters ON)
python backtest_intraday.py --symbols AAPL,MSFT,GOOGL,NVDA,TSLA --days 60 --seed 42

# Simple mode (all filters OFF)
python backtest_intraday.py --symbols AAPL,MSFT,GOOGL,NVDA,TSLA --days 60 --seed 42 --simple-mode
```

### Typical Comparison Results

| Metric | Accurate (filters ON) | Simple (filters OFF) |
|--------|----------------------|---------------------|
| Total Return | +2.63% | +3.69% |
| Max Drawdown | 0.87% | 2.72% |
| Sharpe Ratio | 3.07 | 2.00 |
| Profit Factor | 1.94 | 1.33 |
| Avg Loss | $12.17 | $35.44 |
| Rejected Buys | 2,972 | 254 |

**Key Insight**: The filters produce **lower raw returns** but **much better risk-adjusted returns** with 70% less drawdown and 50% better profit factor.

## When to Use Each Backtest

| Use Case | Module |
|----------|--------|
| Quick strategy validation | `backtest.py` (daily) |
| Accurate day trading simulation | `backtest_intraday.py` |
| Daily parameter optimization | `genetic_optimizer.py` |
| **Intraday parameter optimization** | **`genetic_optimizer_intraday.py`** |
| Testing buy/sell logic | `backtest_intraday.py` |
| Understanding filter impact | `backtest_intraday.py --simple-mode` vs default |

---

# Intraday Genetic Optimizer (Day Trading Parameter Evolution)

The `genetic_optimizer_intraday.py` module uses genetic algorithms to evolve trading parameters using **hourly data** - matching actual day trading performance. This is the recommended optimizer for the RobinhoodBot since the main bot is a day trading application.

## Why Intraday Optimization?

| Optimizer | Data Type | Trades | Best For |
|-----------|-----------|--------|----------|
| `genetic_optimizer.py` | Daily | 3-5 per 200 days | Swing trading |
| **`genetic_optimizer_intraday.py`** | **Hourly** | **100+ per 60 days** | **Day trading (main.py)** |

Since `main.py` uses **hourly SMA crossovers** and makes **multiple trades per day**, the intraday optimizer produces parameters that actually match real-world bot performance.

## Quick Start

```bash
cd robinhoodbot/

# Basic intraday optimization (recommended)
python genetic_optimizer_intraday.py --symbols AAPL,MSFT,GOOGL --generations 15 --population 20

# Full optimization with reproducibility
python genetic_optimizer_intraday.py \
    --symbols AAPL,MSFT,GOOGL,NVDA,TSLA \
    --days 60 \
    --capital 10000 \
    --population 25 \
    --generations 20 \
    --seed 42 \
    --output my_intraday_optimization.json

# Quick test run
python genetic_optimizer_intraday.py --symbols AAPL --generations 5 --population 10 --seed 42
```

## Parameters Being Optimized

The intraday optimizer evolves these parameters (all designed for hourly trading):

| Parameter | Range | Description |
|-----------|-------|-------------|
| `short_sma` | 5-50 hours | Short-term SMA (~1-7 trading days) |
| `long_sma` | 20-100 hours | Long-term SMA (~3-14 trading days) |
| `golden_cross_hours` | 7-72 hours | Lookback for golden cross signals |
| `stop_loss_pct` | 1-15% | Stop loss percentage |
| `take_profit_pct` | 0.3-3.0% | Take profit percentage (tighter for day trading) |
| `position_size_pct` | 5-30% | Maximum position size |
| `slope_threshold` | 0.0001-0.002 | Slope filter for `order_symbols_by_slope` |
| `price_cap_value` | $500-$5000 | Maximum stock price to consider |

### Advanced: Filter Optimization

Use `--optimize-filters` to also evolve which main.py filters to enable/disable:

```bash
python genetic_optimizer_intraday.py --symbols AAPL,MSFT --optimize-filters --generations 20
```

This will evolve whether each filter (market trend, EOD, slope ordering, etc.) should be ON or OFF for optimal performance.

## Fitness Function

The intraday fitness function is tuned for day trading (emphasizes risk-adjusted returns):

| Metric | Weight | Description |
|--------|--------|-------------|
| Total Return | 25% | Overall profit/loss percentage |
| **Sharpe Ratio** | **30%** | Risk-adjusted return (higher weight for day trading) |
| Win Rate | 15% | Percentage of winning trades |
| Profit Factor | 15% | Gross profits / gross losses |
| Max Drawdown | 10% | Penalty for large drawdowns |
| Trades/Day | 5% | Bonus for ~1 trade/day activity level |

Additional penalties:
- No trades: -2.0 fitness
- Fewer than 5 trades: -0.5 fitness
- Bonus for good win/loss ratio

## Command Line Options

```bash
# Full options reference
python genetic_optimizer_intraday.py \
    --symbols AAPL,MSFT,GOOGL \
    --days 60 \
    --capital 10000 \
    --population 25 \
    --generations 20 \
    --mutation-rate 0.15 \
    --crossover-rate 0.7 \
    --optimize-filters \
    --output intraday_optimization.json \
    --seed 42
```

### Options Reference

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--symbols` | `-s` | Comma-separated stock symbols | AAPL,MSFT,GOOGL |
| `--days` | `-d` | Trading days to simulate | 60 |
| `--capital` | `-c` | Initial capital | 10000 |
| `--population` | `-p` | Population size | 20 |
| `--generations` | `-g` | Number of generations | 15 |
| `--mutation-rate` | `-m` | Mutation probability (0.0-1.0) | 0.15 |
| `--crossover-rate` | | Crossover probability (0.0-1.0) | 0.7 |
| `--optimize-filters` | | Also optimize filter on/off settings | False |
| `--output` | `-o` | Output JSON file | genetic_optimization_intraday_result.json |
| `--seed` | | Random seed for reproducibility | None |
| `--quiet` | `-q` | Suppress verbose output | False |

## Example Output

```
======================================================================
INTRADAY GENETIC ALGORITHM OPTIMIZER
======================================================================
Symbols: AAPL, MSFT, GOOGL
Trading Days: 60
Initial Capital: $10,000.00
Population: 20
Generations: 15
Mutation Rate: 0.15
Crossover Rate: 0.7
Optimize Filters: False
======================================================================

--- Generation 1/15 ---
  Evaluating 1/20: SMA(20/50) GC:24h SL:5.0% TP:0.70% Slope:0.0008 | Fit:0.3521
  ...

--- Generation 15/15 ---
  Best:  SMA(15/40) GC:18h SL:3.0% TP:0.85% Slope:0.0012 | Fit:0.6234

======================================================================
EVOLUTION SUMMARY
======================================================================
 Gen |   Best Fit |    Avg Fit |  Return % |   Sharpe | Win % | Tr/Day
----------------------------------------------------------------------------
   1 |     0.3521 |     0.1823 |     +1.24 |     1.52 |  62.0 |   0.85
   ...
  15 |     0.6234 |     0.4521 |     +3.85 |     2.41 |  71.2 |   0.92
----------------------------------------------------------------------------
Improvement: Fitness +0.2713 | Return +2.61%

======================================================================
BEST INTRADAY CONFIGURATION FOUND
======================================================================

# Add these to your config.py:
# Note: SMA values are in HOURS (for hourly data)

# SMA Settings (Hourly)
# short_sma_hours = 15  # ~2.1 trading days
# long_sma_hours = 40   # ~5.7 trading days
golden_cross_buy_days = 2  # 18 hours

# Risk Management
stop_loss_percent = 3.0
take_profit_percent = 0.85

# Position Sizing
purchase_limit_percentage = 18.0

# Main.py Specific Settings
# slope_threshold = 0.0012  # For order_symbols_by_slope
price_cap = 1800

# Performance Metrics:
# Total Return: +3.85%
# Win Rate: 71.2%
# Sharpe Ratio: 2.4100
# Max Drawdown: 1.52%
# Profit Factor: 2.31
# Total Trades: 98
# Trades/Day: 0.92
# Avg Win: $14.21
# Avg Loss: $11.82
# Fitness Score: 0.6234
======================================================================
```

## Tips for Best Results

1. **Use 60+ Trading Days**: More data captures various market conditions
2. **Population 20-30**: Good balance of exploration vs computation time
3. **Generations 15-25**: Allows proper convergence
4. **Multiple Runs**: Run with different seeds and compare results
5. **Diverse Symbols**: Include different sectors (tech, finance, energy)
6. **Start Without Filter Optimization**: Get good base parameters first, then try `--optimize-filters`
7. **Validate Results**: Run `backtest_intraday.py` with the optimized parameters to confirm

