"""
Backtesting Configuration

Customize these settings for your backtesting needs
"""

# Default backtest parameters
DEFAULT_INITIAL_CASH = 10000.0
DEFAULT_COMMISSION = 0.0  # Robinhood has no commission

# Strategy parameters
STRATEGY_PARAMS = {
    # Golden Cross / Death Cross parameters
    'n1_buy': 20,      # Short-term SMA for buy signals
    'n2_buy': 50,      # Long-term SMA for buy signals
    'n1_sell': 20,     # Short-term SMA for sell signals
    'n2_sell': 50,     # Long-term SMA for sell signals
    
    # Profit taking
    'take_profit_threshold': 2.15,  # Sell when profit exceeds this %
    
    # Cross validation
    'cross_lookback_days': 3,   # Days to look back for golden cross
    'death_cross_days': 10,     # Days to look back for death cross
}

# Data settings
DATA_SETTINGS = {
    'default_interval': 'day',
    'default_span': '5year',
    'cache_enabled': True,
    'cache_file': 'robinhoodbot/backtest_cache.json'
}

# Output settings
OUTPUT_SETTINGS = {
    'report_file': 'robinhoodbot/backtest_report.txt',
    'equity_csv': 'robinhoodbot/backtest_equity.csv',
    'trades_csv': 'robinhoodbot/backtest_trades.csv',
    'equity_plot': 'robinhoodbot/equity_curve.png',
    'analysis_plot': 'robinhoodbot/trade_analysis.png',
    'returns_plot': 'robinhoodbot/returns_distribution.png'
}

# Default symbol lists for testing
SYMBOL_LISTS = {
    'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'NFLX', 'ADBE'],
    'large_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BRK.B', 'JNJ', 'JPM', 'V', 'PG', 'NVDA'],
    'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK', 'LLY', 'DHR', 'AMGN'],
    'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'SPGI'],
    'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL'],
    'consumer': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'CMG'],
    'industrial': ['BA', 'HON', 'UNP', 'UPS', 'CAT', 'GE', 'LMT', 'MMM', 'DE', 'RTX'],
    'utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'ES', 'PEG'],
}

# Preset backtest configurations
PRESET_CONFIGS = {
    'conservative': {
        'n1_buy': 30,
        'n2_buy': 90,
        'n1_sell': 20,
        'n2_sell': 50,
        'take_profit_threshold': 5.0
    },
    'moderate': {
        'n1_buy': 20,
        'n2_buy': 50,
        'n1_sell': 20,
        'n2_sell': 50,
        'take_profit_threshold': 2.15
    },
    'aggressive': {
        'n1_buy': 10,
        'n2_buy': 30,
        'n1_sell': 10,
        'n2_sell': 30,
        'take_profit_threshold': 1.5
    },
    'fast': {
        'n1_buy': 5,
        'n2_buy': 15,
        'n1_sell': 5,
        'n2_sell': 15,
        'take_profit_threshold': 1.0
    }
}

# Benchmark symbols for comparison
BENCHMARKS = {
    'SP500': 'SPY',
    'NASDAQ': 'QQQ',
    'DOW': 'DIA',
    'RUSSELL2000': 'IWM',
    'TOTAL_MARKET': 'VTI'
}
