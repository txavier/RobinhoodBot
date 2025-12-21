"""
Backtest Runner for RobinhoodBot

This script runs backtests using the existing trading strategy logic
against historical data.

IMPORTANT LIMITATIONS:
- The live bot runs every 15 minutes (see run.sh)
- This backtest uses DAILY closing prices and executes once per day
- Real bot may execute trades at different intraday prices
- Intraday signals (sudden_drop, profit_before_eod) are simplified/omitted
- Use for strategy validation and parameter tuning, not exact timing prediction
"""

import sys
import pandas as pd
from datetime import datetime, timedelta
from backtest import BacktestEngine
from main import golden_cross, sudden_drop, take_profit, profit_before_eod, purchase_limiter
from config import *
import robin_stocks.robinhood as rr


class StrategyBacktest:
    """
    Backtesting runner that uses the actual trading strategy from main.py
    
    Note: Simulates using daily candles, while live bot runs every 15 minutes.
    Golden cross signals are based on daily SMAs so this is reasonably accurate,
    but trade execution timing may differ from live trading.
    """
    
    def __init__(self, symbols, start_date, end_date, initial_cash=10000.0):
        """
        Initialize strategy backtest
        
        Args:
            symbols: List of stock symbols to test
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_cash: Starting cash amount
        """
        self.engine = BacktestEngine(initial_cash=initial_cash)
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        
        # Strategy parameters (from main.py defaults)
        self.n1_buy = 20  # Short-term SMA for buying
        self.n2_buy = 50  # Long-term SMA for buying
        self.n1_sell = 20  # Short-term SMA for selling
        self.n2_sell = 50  # Long-term SMA for selling
        self.take_profit_threshold = 2.15  # % profit threshold
        
        # Config parameters
        self.use_price_cap = use_price_cap
        self.price_cap = price_cap
        self.use_purchase_limit_percentage = use_purchase_limit_percentage
        self.purchase_limit_percentage = purchase_limit_percentage
        
    def run(self):
        """Execute the backtest"""
        print("=" * 80)
        print("ROBINHOODBOT BACKTESTING")
        print("=" * 80)
        print(f"Symbols:       {', '.join(self.symbols)}")
        print(f"Date Range:    {self.start_date} to {self.end_date}")
        print(f"Initial Cash:  ${self.engine.portfolio.initial_cash:,.2f}")
        print("=" * 80)
        print("Config Parameters:")
        print(f"  Price Cap:          ${self.price_cap if self.use_price_cap else 'None'}")
        print(f"  Purchase Limit:     {self.purchase_limit_percentage}% of equity" if self.use_purchase_limit_percentage else "  Purchase Limit:     None")
        print("=" * 80)
        
        # Load historical data
        print("\n[1/3] Loading historical data...")
        self.engine.load_historical_data(self.symbols, interval='day', span='5year')
        
        # Get sorted list of dates
        all_dates = sorted(self.engine.data_by_date.keys())
        test_dates = [d for d in all_dates if self.start_date <= d <= self.end_date]
        
        if not test_dates:
            print(f"ERROR: No data available in date range {self.start_date} to {self.end_date}")
            return None
        
        print(f"Testing period: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")
        
        # Run backtest
        print("\n[2/3] Running backtest...")
        self._run_backtest(test_dates)
        
        # Generate report
        print("\n[3/3] Generating results...")
        metrics = self.engine.generate_report()
        self.engine.export_results()
        
        return metrics
    
    def _run_backtest(self, test_dates):
        """Run the backtest simulation"""
        portfolio_symbols = []  # Symbols currently in portfolio
        
        for date_idx, current_date in enumerate(test_dates):
            current_prices = self.engine.get_prices_for_date(current_date)
            
            if not current_prices:
                continue
            
            # Record portfolio value
            self.engine.portfolio.record_equity(current_date, current_prices)
            
            # Check for sells (existing positions)
            sells = []
            sell_reasons = {}
            
            for symbol in list(portfolio_symbols):
                if symbol not in current_prices:
                    continue
                
                current_price = current_prices[symbol]
                holdings_data = self.engine.portfolio.get_holdings_data(current_prices)
                
                # Check sell conditions
                try:
                    # Death cross check
                    cross = golden_cross(symbol, n1=self.n1_sell, n2=self.n2_sell, days=10, direction="below")
                    is_death_cross = cross[0] == -1
                    
                    # Sudden drop check (simplified - would need intraday data for accurate simulation)
                    is_sudden_drop = False  # sudden_drop requires hourly data
                    
                    # Take profit check
                    is_take_profit = False
                    if symbol in holdings_data:
                        avg_cost = float(holdings_data[symbol]['average_buy_price'])
                        profit_pct = ((current_price - avg_cost) / avg_cost) * 100
                        is_take_profit = profit_pct >= self.take_profit_threshold
                    
                    # Profit before EOD check (simplified)
                    is_profit_before_eod = False  # Would need time-of-day data
                    
                    # Determine sell reason
                    sell_reason = None
                    if is_death_cross:
                        sell_reason = "death_cross"
                    elif is_sudden_drop:
                        sell_reason = "sudden_drop"
                    elif is_take_profit:
                        sell_reason = "take_profit"
                    elif is_profit_before_eod:
                        sell_reason = "profit_before_eod"
                    
                    if sell_reason:
                        sells.append(symbol)
                        sell_reasons[symbol] = sell_reason
                
                except Exception as e:
                    # Skip if error calculating indicators
                    pass
            
            # Execute sells
            for symbol in sells:
                price = current_prices[symbol]
                reason = sell_reasons.get(symbol, "unknown")
                success, msg = self.engine.portfolio.sell(symbol, price, current_date, reason)
                if success:
                    portfolio_symbols.remove(symbol)
            
            # Check for buys (symbols not in portfolio)
            potential_buys = []
            buy_reasons = {}
            
            available_symbols = [s for s in self.symbols if s in current_prices and s not in portfolio_symbols]
            
            for symbol in available_symbols:
                current_price = current_prices[symbol]
                
                # Apply price cap filter
                if self.use_price_cap and current_price > self.price_cap:
                    continue
                
                try:
                    # Golden cross check
                    cross = golden_cross(symbol, n1=self.n1_buy, n2=self.n2_buy, days=3, direction="above")
                    
                    if cross[0] == 1:
                        # Additional filters from main.py
                        # Note: cross[3] (5-hour-ago price) may not be meaningful with daily data
                        # We simplify to just check if price is rising from cross point
                        if float(cross[2]) > float(cross[1]):  # Current price > price at cross
                            potential_buys.append(symbol)
                            buy_reasons[symbol] = "golden_cross"
                            if verbose and date_idx % 30 == 0:  # Log occasionally
                                print(f"    Golden cross detected for {symbol} at ${cross[2]}")
                
                except Exception as e:
                    # Skip if error calculating indicators
                    if verbose and "list index out of range" not in str(e):
                        pass  # Silently skip common errors
            
            # Execute buys
            if potential_buys:
                # Calculate position sizing (simplified from main.py buy_holdings logic)
                equity = self.engine.portfolio.get_portfolio_value(current_prices)
                cash = self.engine.portfolio.cash
                num_positions = len(portfolio_symbols)
                
                for symbol in potential_buys:
                    price = current_prices[symbol]
                    
                    # Simple position sizing: divide available cash by number of potential buys
                    position_size = cash / (len(potential_buys) + num_positions + 1)
                    shares = int(position_size / price)
                    
                    # Apply purchase limit from config
                    if self.use_purchase_limit_percentage:
                        shares = purchase_limiter(shares, price, equity)
                    
                    if shares > 0:
                        reason = buy_reasons.get(symbol, "golden_cross")
                        success, msg = self.engine.portfolio.buy(symbol, shares, price, current_date, reason)
                        if success:
                            portfolio_symbols.append(symbol)
                            # Update cash after successful buy
                            cash = self.engine.portfolio.cash
            
            # Progress update
            if (date_idx + 1) % 30 == 0 or date_idx == len(test_dates) - 1:
                progress = ((date_idx + 1) / len(test_dates)) * 100
                value = self.engine.portfolio.get_portfolio_value(current_prices)
                print(f"  Progress: {progress:5.1f}% | Date: {current_date} | Portfolio: ${value:,.2f} | Positions: {len(portfolio_symbols)}")


def run_simple_backtest(symbols=None, start_date=None, end_date=None, initial_cash=None):
    """
    Run a simple backtest with default parameters
    
    Args:
        symbols: List of symbols (default: common stocks)
        start_date: Start date YYYY-MM-DD (default: 1 year ago)
        end_date: End date YYYY-MM-DD (default: today)
        initial_cash: Starting capital (default: uses 'investing' from config.py)
    """
    # Use investing amount from config if not specified
    if initial_cash is None:
        initial_cash = investing
    
    # Default symbols if none provided
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'AMD', 'NFLX', 'DIS']
    
    # Default dates if none provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        start = datetime.now() - timedelta(days=365)
        start_date = start.strftime('%Y-%m-%d')
    
    # Run backtest
    backtest = StrategyBacktest(symbols, start_date, end_date, initial_cash)
    metrics = backtest.run()
    
    return metrics


def run_custom_backtest(symbols, start_date, end_date, initial_cash=10000.0):
    """
    Run a backtest with custom parameters
    
    Args:
        symbols: List of stock symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_cash: Starting cash amount
    """
    backtest = StrategyBacktest(symbols, start_date, end_date, initial_cash)
    metrics = backtest.run()
    return metrics


if __name__ == "__main__":
    """Run backtest from command line"""
    
    # Example usage
    print("\nRobinhoodBot Backtesting System")
    print("=" * 80)
    print("\nExample 1: Default backtest (10 tech stocks, 1 year, $10,000)")
    print("  python run_backtest.py")
    print("\nExample 2: Custom symbols")
    print("  python run_backtest.py AAPL MSFT GOOGL")
    print("\nExample 3: Custom date range")
    print("  python run_backtest.py --start 2024-01-01 --end 2024-12-31")
    print("\n" + "=" * 80)
    
    # Parse command line arguments
    if len(sys.argv) > 1 and '--start' not in sys.argv and '--help' not in sys.argv:
        # Custom symbols provided
        symbols = sys.argv[1:]
        print(f"\nRunning backtest with custom symbols: {', '.join(symbols)}")
        run_simple_backtest(symbols=symbols)
    elif '--start' in sys.argv:
        # Custom date range
        start_idx = sys.argv.index('--start') + 1
        end_idx = sys.argv.index('--end') + 1 if '--end' in sys.argv else None
        
        start_date = sys.argv[start_idx]
        end_date = sys.argv[end_idx] if end_idx else None
        
        symbols = [arg for arg in sys.argv[1:] if not arg.startswith('--') and arg not in [start_date, end_date]]
        if not symbols:
            symbols = None
        
        print(f"\nRunning backtest from {start_date} to {end_date or 'today'}")
        run_simple_backtest(symbols=symbols, start_date=start_date, end_date=end_date)
    else:
        # Default backtest
        print("\nRunning default backtest...")
        run_simple_backtest()
