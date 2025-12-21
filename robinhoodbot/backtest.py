"""
Backtesting Engine for RobinhoodBot

This module provides comprehensive backtesting capabilities to test trading strategies
against historical data without risking real capital.
"""

import pandas as pd
import numpy as np
import json
import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
try:
    from robin_stocks_adapter import rsa
except ImportError:
    # If robin_stocks is not installed, we can still define the classes
    # but historical data fetching won't work
    rsa = None


class SimulatedPortfolio:
    """Manages a simulated trading portfolio for backtesting"""
    
    def __init__(self, initial_cash: float = 10000.0, commission: float = 0.0):
        """
        Initialize simulated portfolio
        
        Args:
            initial_cash: Starting cash amount
            commission: Commission per trade (default 0 for Robinhood)
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission = commission
        self.positions = {}  # {symbol: {'shares': float, 'avg_cost': float, 'bought_at': str, 'buy_reason': str}}
        self.trade_history = []
        self.equity_curve = []
        self.current_date = None
        
    def buy(self, symbol: str, shares: int, price: float, date: str, reason: str = ""):
        """Execute a simulated buy order"""
        cost = shares * price + self.commission
        
        if cost > self.cash:
            return False, f"Insufficient funds: ${self.cash:.2f} < ${cost:.2f}"
        
        self.cash -= cost
        
        if symbol in self.positions:
            # Average down
            old_shares = self.positions[symbol]['shares']
            old_cost = self.positions[symbol]['avg_cost']
            new_shares = old_shares + shares
            new_avg_cost = ((old_shares * old_cost) + (shares * price)) / new_shares
            self.positions[symbol]['shares'] = new_shares
            self.positions[symbol]['avg_cost'] = new_avg_cost
        else:
            self.positions[symbol] = {
                'shares': shares,
                'avg_cost': price,
                'bought_at': date,
                'buy_reason': reason
            }
        
        self.trade_history.append({
            'date': date,
            'type': 'BUY',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'commission': self.commission,
            'reason': reason,
            'cash_after': self.cash
        })
        
        return True, f"Bought {shares} shares of {symbol} at ${price:.2f}"
    
    def sell(self, symbol: str, price: float, date: str, reason: str = ""):
        """Execute a simulated sell order"""
        if symbol not in self.positions:
            return False, f"No position in {symbol}"
        
        shares = self.positions[symbol]['shares']
        avg_cost = self.positions[symbol]['avg_cost']
        proceeds = (shares * price) - self.commission
        profit = proceeds - (shares * avg_cost)
        profit_pct = (profit / (shares * avg_cost)) * 100
        
        self.cash += proceeds
        buy_reason = self.positions[symbol].get('buy_reason', '')
        bought_at = self.positions[symbol].get('bought_at', '')
        
        self.trade_history.append({
            'date': date,
            'type': 'SELL',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'commission': self.commission,
            'avg_cost': avg_cost,
            'profit': profit,
            'profit_pct': profit_pct,
            'buy_reason': buy_reason,
            'sell_reason': reason,
            'bought_at': bought_at,
            'cash_after': self.cash
        })
        
        del self.positions[symbol]
        
        return True, f"Sold {shares} shares of {symbol} at ${price:.2f} (P/L: ${profit:.2f}, {profit_pct:.2f}%)"
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(
            pos['shares'] * current_prices.get(symbol, pos['avg_cost'])
            for symbol, pos in self.positions.items()
        )
        return self.cash + positions_value
    
    def record_equity(self, date: str, current_prices: Dict[str, float]):
        """Record portfolio value for equity curve"""
        total_value = self.get_portfolio_value(current_prices)
        self.equity_curve.append({
            'date': date,
            'value': total_value,
            'cash': self.cash,
            'positions_value': total_value - self.cash
        })
        
    def get_holdings_data(self, current_prices: Dict[str, float]) -> Dict:
        """Get holdings in the same format as get_modified_holdings()"""
        holdings = {}
        for symbol, pos in self.positions.items():
            current_price = current_prices.get(symbol, pos['avg_cost'])
            equity = pos['shares'] * current_price
            equity_change = equity - (pos['shares'] * pos['avg_cost'])
            percent_change = (equity_change / (pos['shares'] * pos['avg_cost'])) * 100
            
            holdings[symbol] = {
                'price': str(current_price),
                'quantity': str(pos['shares']),
                'average_buy_price': str(pos['avg_cost']),
                'equity': str(equity),
                'percent_change': str(percent_change),
                'equity_change': str(equity_change),
                'bought_at': pos.get('bought_at', ''),
                'buy_reason': pos.get('buy_reason', '')
            }
        return holdings


class HistoricalDataCache:
    """Cache for historical price data"""
    
    def __init__(self, cache_file: str = "backtest_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cache from file"""
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_cache(self):
        """Save cache to file"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=4)
    
    def get_historical_data(self, symbol: str, interval: str = 'day', span: str = 'year') -> List[Dict]:
        """Get historical data with caching"""
        cache_key = f"{symbol}_{interval}_{span}"
        
        if cache_key in self.cache:
            print(f"Using cached data for {symbol}")
            return self.cache[cache_key]
        
        print(f"Fetching historical data for {symbol} ({interval}, {span})...")
        try:
            data = rsa.get_stock_historicals(symbol, interval=interval, span=span, bounds='regular')
            if data:
                self.cache[cache_key] = data
                self._save_cache()
            return data if data else []
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return []
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache = {}
        self._save_cache()


class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, initial_cash: float = 10000.0, commission: float = 0.0):
        """
        Initialize backtest engine
        
        Args:
            initial_cash: Starting cash amount
            commission: Commission per trade
        """
        self.portfolio = SimulatedPortfolio(initial_cash, commission)
        self.cache = HistoricalDataCache()
        self.data_by_date = defaultdict(dict)  # {date: {symbol: price}}
        self.results = None
        
    def load_historical_data(self, symbols: List[str], interval: str = 'day', span: str = 'year'):
        """Load historical data for multiple symbols"""
        print(f"\nLoading historical data for {len(symbols)} symbols...")
        
        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] Loading {symbol}...")
            data = self.cache.get_historical_data(symbol, interval, span)
            
            if not data:
                print(f"Warning: No data for {symbol}")
                continue
            
            for item in data:
                date = item['begins_at'][:10]  # Extract date (YYYY-MM-DD)
                price = float(item['close_price'])
                self.data_by_date[date][symbol] = price
        
        print(f"\nLoaded data for {len(self.data_by_date)} trading days")
        
    def get_available_symbols_for_date(self, date: str) -> List[str]:
        """Get list of symbols with data available for a given date"""
        return list(self.data_by_date.get(date, {}).keys())
    
    def get_price(self, symbol: str, date: str) -> Optional[float]:
        """Get price for a symbol on a specific date"""
        return self.data_by_date.get(date, {}).get(symbol)
    
    def get_prices_for_date(self, date: str) -> Dict[str, float]:
        """Get all prices for a specific date"""
        return self.data_by_date.get(date, {})
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.portfolio.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.portfolio.equity_curve)
        trades_df = pd.DataFrame(self.portfolio.trade_history)
        
        # Basic metrics
        initial_value = self.portfolio.initial_cash
        final_value = equity_df['value'].iloc[-1]
        total_return = ((final_value - initial_value) / initial_value) * 100
        
        # Trade metrics
        sell_trades = trades_df[trades_df['type'] == 'SELL']
        winning_trades = sell_trades[sell_trades['profit'] > 0]
        losing_trades = sell_trades[sell_trades['profit'] < 0]
        
        win_rate = len(winning_trades) / len(sell_trades) * 100 if len(sell_trades) > 0 else 0
        avg_win = winning_trades['profit'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['profit'].mean() if len(losing_trades) > 0 else 0
        
        # Drawdown
        equity_df['peak'] = equity_df['value'].cummax()
        equity_df['drawdown'] = (equity_df['value'] - equity_df['peak']) / equity_df['peak'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # Sharpe ratio (simplified, assuming daily returns)
        equity_df['returns'] = equity_df['value'].pct_change()
        sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std()) * np.sqrt(252) if equity_df['returns'].std() > 0 else 0
        
        metrics = {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return_pct': total_return,
            'total_return_dollar': final_value - initial_value,
            'total_trades': len(trades_df),
            'buy_trades': len(trades_df[trades_df['type'] == 'BUY']),
            'sell_trades': len(sell_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_profit_per_trade': sell_trades['profit'].mean() if len(sell_trades) > 0 else 0,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_days': len(equity_df),
            'final_cash': self.portfolio.cash,
            'open_positions': len(self.portfolio.positions)
        }
        
        return metrics
    
    def generate_report(self, output_file: str = "backtest_report.txt"):
        """Generate detailed backtest report"""
        metrics = self.calculate_performance_metrics()
        
        report = []
        report.append("=" * 80)
        report.append("BACKTEST RESULTS")
        report.append("=" * 80)
        report.append("")
        
        report.append("PORTFOLIO PERFORMANCE:")
        report.append(f"  Initial Value:        ${metrics['initial_value']:,.2f}")
        report.append(f"  Final Value:          ${metrics['final_value']:,.2f}")
        report.append(f"  Total Return:         ${metrics['total_return_dollar']:,.2f} ({metrics['total_return_pct']:.2f}%)")
        report.append(f"  Max Drawdown:         {metrics['max_drawdown_pct']:.2f}%")
        report.append(f"  Sharpe Ratio:         {metrics['sharpe_ratio']:.2f}")
        report.append(f"  Final Cash:           ${metrics['final_cash']:,.2f}")
        report.append(f"  Open Positions:       {metrics['open_positions']}")
        report.append("")
        
        report.append("TRADING STATISTICS:")
        report.append(f"  Total Trades:         {metrics['total_trades']}")
        report.append(f"  Buy Orders:           {metrics['buy_trades']}")
        report.append(f"  Sell Orders:          {metrics['sell_trades']}")
        report.append(f"  Winning Trades:       {metrics['winning_trades']}")
        report.append(f"  Losing Trades:        {metrics['losing_trades']}")
        report.append(f"  Win Rate:             {metrics['win_rate']:.2f}%")
        report.append(f"  Avg Win:              ${metrics['avg_win']:,.2f}")
        report.append(f"  Avg Loss:             ${metrics['avg_loss']:,.2f}")
        report.append(f"  Avg Profit/Trade:     ${metrics['avg_profit_per_trade']:,.2f}")
        report.append("")
        
        report.append("=" * 80)
        report.append("TRADE HISTORY:")
        report.append("=" * 80)
        
        for trade in self.portfolio.trade_history:
            if trade['type'] == 'BUY':
                report.append(f"{trade['date']} | BUY  | {trade['symbol']:6} | {trade['shares']:4.0f} shares @ ${trade['price']:8.2f} | Reason: {trade['reason']}")
            else:
                report.append(f"{trade['date']} | SELL | {trade['symbol']:6} | {trade['shares']:4.0f} shares @ ${trade['price']:8.2f} | P/L: ${trade['profit']:8.2f} ({trade['profit_pct']:6.2f}%) | Reason: {trade['sell_reason']}")
        
        report_text = "\n".join(report)
        
        # Print to console
        print("\n" + report_text)
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(f"\nReport saved to {output_file}")
        
        return metrics
    
    def export_results(self, equity_file: str = "backtest_equity.csv",
                      trades_file: str = "backtest_trades.csv"):
        """Export results to CSV files"""
        # Export equity curve
        if self.portfolio.equity_curve:
            equity_df = pd.DataFrame(self.portfolio.equity_curve)
            equity_df.to_csv(equity_file, index=False)
            print(f"Equity curve saved to {equity_file}")
        
        # Export trades
        if self.portfolio.trade_history:
            trades_df = pd.DataFrame(self.portfolio.trade_history)
            trades_df.to_csv(trades_file, index=False)
            print(f"Trade history saved to {trades_file}")
