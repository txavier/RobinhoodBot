# RobinhoodBot
Trading bot for Robinhood accounts

For more info:
https://medium.com/@kev.guo123/building-a-robinhood-stock-trading-bot-8ee1b040ec6a


5/1/19: Since Robinhood has updated it's API, you now have to enter a 2 factor authentication code whenever you run the script. To do this, go to the Robinhood mobile app and enable two factor authentication in your settings. You will now receive an SMS code when you run the script, which you have to enter into the script.



This project supports Python 3.7+

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/txavier/RobinhoodBot.git
cd RobinhoodBot

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
cd robinhoodbot
pip install -r requirements.txt

# 4. Configure your credentials
cp config.py.sample config.py
# Edit config.py with your Robinhood credentials:
#   - rh_username: Your Robinhood email
#   - RH_DEVICE_TOKEN: Your 16-character device token
#   - rh_email/rh_mail_password: For SMS notifications (optional)

# 5. Setup Robinhood
# - Enable 2FA in Robinhood app (Settings > Security > Two-Factor Authentication)
# - Create a watchlist named "Default" with stocks to monitor
# - Create a watchlist named "Exclusion" for stocks to ignore

# 6. Run the bot
python main.py          # Single run
./run.sh               # Continuous loop (runs every 7 minutes)
```

---

## Installation (Detailed)

```bash
git clone https://github.com/txavier/RobinhoodBot.git
cd RobinhoodBot/
pip install -r requirements.txt
cp config.py.sample config.py # add auth info and watchlist name to monitor after copying
```

## Running the Bot

In RobinHood create a watchlist named "Exclusion".  This will be the watchlist that you will use to tell the bot to ignore the stock tickers contained within.

```bash
cd RobinhoodBot/robinhoodbot
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
/home/[Your home directory]/dev/.venv/bin/activate
~/> source dev/.venv/bin/activate


# Backtesting & Parameter Optimization

The bot includes intraday backtesting and genetic optimization modules that accurately simulate day trading using hourly data - matching the actual main.py trading logic.

# Intraday Backtesting (Day Trading Simulation)

The `backtest_intraday.py` module provides **accurate day trading simulation** that matches the actual main.py trading logic. Unlike the basic backtest which uses daily data, this module simulates hourly trading with all the same conditions the real bot uses.

## Why Intraday Backtesting?

The main bot (`main.py`) is a **day trading** application that:
- Uses **hourly SMA crossovers** (not daily)
- Makes **multiple trades per day**
- Has **aggressive take-profit targets** (1.88% optimized)
- Includes many **conditional filters** before buying/selling

The intraday backtester generates hourly data and applies all the same filters as main.py, providing accurate day trading simulation.

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

---

# Intraday Genetic Optimizer (Day Trading Parameter Evolution)

The `genetic_optimizer_intraday.py` module uses genetic algorithms to evolve trading parameters using **hourly data** - matching actual day trading performance.

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

# Use parallel processing for faster optimization (auto-detects CPU cores)
python genetic_optimizer_intraday.py --symbols AAPL,MSFT,GOOGL,NVDA,TSLA,META,AMZN -w 0
```

## Parallel Processing

The optimizer supports multiprocessing to speed up fitness evaluation:

| Option | Description |
|--------|-------------|
| `-w 0` or `--workers 0` | Auto-detect (uses cpu_count - 1 workers) |
| `-w 1` or `--workers 1` | Sequential execution (no parallelism, useful for debugging) |
| `-w N` or `--workers N` | Use exactly N parallel workers |

Example with 8 workers:
```bash
python genetic_optimizer_intraday.py --symbols AAPL,MSFT,GOOGL -w 8
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
| `--max-positions` | | Maximum concurrent positions | 5 |
| `--validate-real` | | Validate results against tradehistory-real.json | False |
| `--output` | `-o` | Output JSON file | genetic_optimization_intraday_result.json |
| `--seed` | | Random seed for reproducibility | None |
| `--quiet` | `-q` | Suppress verbose output | False |
| `--workers` | `-w` | Number of parallel workers (0 = auto, uses cpu_count-1) | 0 |
| `--use-ray` | | Use Ray for distributed computing (local or Kubernetes) | False |
| `--disable-ray-mem-monitor` | | Disable Ray memory monitor (fixes cgroup v2 crashes) | False |

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

