#!/usr/bin/env python3
"""
Backtesting Module for RobinhoodBot

This module allows you to test the trading strategy against historical data
to evaluate performance before risking real money.

Usage:
    python backtest.py --symbols AAPL,MSFT,GOOGL --start 2024-01-01 --end 2024-12-31
    python backtest.py --watchlist Default --days 365
    python backtest.py --portfolio --days 180
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np
import ta as t
from scipy.stats import linregress

# Import config settings
try:
    from config import (
        stop_loss_percent, take_profit_percent, use_stop_loss,
        golden_cross_buy_days, price_cap, use_price_cap,
        min_volume, min_market_cap, purchase_limit_percentage,
        use_purchase_limit_percentage, investing, version
    )
except ImportError:
    # Default values if config not available
    stop_loss_percent = 5
    take_profit_percent = 0.70
    use_stop_loss = True
    golden_cross_buy_days = 3
    price_cap = 2100
    use_price_cap = True
    min_volume = 1000000
    min_market_cap = 400000
    purchase_limit_percentage = 15
    use_purchase_limit_percentage = True
    investing = 10000
    version = "backtest"


class TradeType(Enum):
    BUY = "buy"
    SELL = "sell"


class SellReason(Enum):
    DEATH_CROSS = "death_cross"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    SUDDEN_DROP = "sudden_drop"
    END_OF_BACKTEST = "end_of_backtest"


class BuyReason(Enum):
    GOLDEN_CROSS = "golden_cross"


@dataclass
class Trade:
    """Represents a single trade"""
    symbol: str
    trade_type: TradeType
    price: float
    shares: float
    timestamp: datetime
    reason: str
    total_value: float = 0.0
    
    def __post_init__(self):
        self.total_value = self.price * self.shares


@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    shares: float
    avg_buy_price: float
    buy_timestamp: datetime
    total_cost: float = 0.0
    
    def __post_init__(self):
        self.total_cost = self.shares * self.avg_buy_price
    
    def current_value(self, current_price: float) -> float:
        return self.shares * current_price
    
    def profit_loss(self, current_price: float) -> float:
        return self.current_value(current_price) - self.total_cost
    
    def profit_loss_pct(self, current_price: float) -> float:
        if self.total_cost == 0:
            return 0
        return ((current_price - self.avg_buy_price) / self.avg_buy_price) * 100


@dataclass
class BacktestResult:
    """Contains all results from a backtest run"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_return: float
    total_return_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    symbols_tested: List[str] = field(default_factory=list)
    config_used: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'start_date': str(self.start_date),
            'end_date': str(self.end_date),
            'initial_capital': self.initial_capital,
            'final_capital': round(self.final_capital, 2),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_return': round(self.total_return, 2),
            'total_return_pct': round(self.total_return_pct, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'max_drawdown_pct': round(self.max_drawdown_pct, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 4),
            'win_rate': round(self.win_rate, 2),
            'avg_win': round(self.avg_win, 2),
            'avg_loss': round(self.avg_loss, 2),
            'profit_factor': round(self.profit_factor, 2),
            'symbols_tested': self.symbols_tested,
            'config_used': self.config_used,
            'trades': [
                {
                    'symbol': t.symbol,
                    'type': t.trade_type.value,
                    'price': round(t.price, 2),
                    'shares': round(t.shares, 4),
                    'total_value': round(t.total_value, 2),
                    'timestamp': str(t.timestamp),
                    'reason': t.reason
                }
                for t in self.trades
            ]
        }


class HistoricalDataProvider:
    """
    Provides historical stock data for backtesting.
    Can use cached data, sample data, or fetch from robin_stocks API.
    """
    
    def __init__(self, cache_dir: str = "backtest_cache", sample_data_dir: str = None):
        self.cache_dir = cache_dir
        self.sample_data_dir = sample_data_dir
        self._ensure_cache_dir()
        self._data_cache: Dict[str, pd.DataFrame] = {}
    
    def _ensure_cache_dir(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = 'day'
    ) -> Optional[pd.DataFrame]:
        """
        Get historical price data for a symbol.
        Returns DataFrame with columns: date, open, high, low, close, volume
        """
        cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}_{interval}"
        
        # Check memory cache first
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        # Check sample data directory
        if self.sample_data_dir:
            sample_file = os.path.join(self.sample_data_dir, f"{symbol}_historical.json")
            if os.path.exists(sample_file):
                try:
                    df = pd.read_json(sample_file)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                    self._data_cache[cache_key] = df
                    return df
                except Exception as e:
                    print(f"Error loading sample data for {symbol}: {e}")
        
        # Check file cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                df = pd.read_json(cache_file)
                df['date'] = pd.to_datetime(df['date'])
                self._data_cache[cache_key] = df
                return df
            except Exception:
                pass
        
        # Fetch from API
        df = self._fetch_from_api(symbol, start_date, end_date, interval)
        
        if df is not None and not df.empty:
            # Cache to file
            df.to_json(cache_file)
            self._data_cache[cache_key] = df
        
        return df
    
    def _fetch_from_api(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data from robin_stocks API"""
        try:
            import robin_stocks.robinhood as rr
            
            # Determine span based on date range
            days_diff = (end_date - start_date).days
            if days_diff <= 7:
                span = 'week'
            elif days_diff <= 30:
                span = 'month'
            elif days_diff <= 90:
                span = '3month'
            elif days_diff <= 365:
                span = 'year'
            else:
                span = '5year'
            
            # Adjust interval based on span
            if interval == 'hour' and span in ['year', '5year']:
                interval = 'day'
            
            historicals = rr.get_stock_historicals(
                symbol, interval=interval, span=span, bounds='regular'
            )
            
            if not historicals:
                return None
            
            # Convert to DataFrame
            data = []
            for h in historicals:
                data.append({
                    'date': pd.to_datetime(h['begins_at']),
                    'open': float(h['open_price']),
                    'high': float(h['high_price']),
                    'low': float(h['low_price']),
                    'close': float(h['close_price']),
                    'volume': int(h['volume'])
                })
            
            df = pd.DataFrame(data)
            
            # Filter by date range
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            return df.reset_index(drop=True)
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_hourly_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Get hourly data for more granular backtesting"""
        return self.get_historical_data(symbol, start_date, end_date, interval='hour')
    
    def load_from_csv(self, filepath: str, symbol: str) -> Optional[pd.DataFrame]:
        """Load historical data from a CSV file"""
        try:
            df = pd.read_csv(filepath)
            # Standardize column names
            df.columns = df.columns.str.lower()
            if 'date' not in df.columns and 'datetime' in df.columns:
                df['date'] = df['datetime']
            df['date'] = pd.to_datetime(df['date'])
            
            # Cache it
            cache_key = f"{symbol}_csv"
            self._data_cache[cache_key] = df
            
            return df
        except Exception as e:
            print(f"Error loading CSV {filepath}: {e}")
            return None


class TradingStrategy:
    """
    Implements the golden cross/death cross trading strategy
    matching the logic in main.py
    """
    
    def __init__(
        self,
        short_sma: int = 20,
        long_sma: int = 50,
        golden_cross_days: int = 3,
        stop_loss_pct: float = 5.0,
        take_profit_pct: float = 0.70,
        use_stop_loss: bool = True
    ):
        self.short_sma = short_sma
        self.long_sma = long_sma
        self.golden_cross_days = golden_cross_days
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.use_stop_loss = use_stop_loss
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate SMA indicators on the dataframe"""
        df = df.copy()
        df['sma_short'] = t.volatility.bollinger_mavg(
            df['close'], window=self.short_sma, fillna=False
        )
        df['sma_long'] = t.volatility.bollinger_mavg(
            df['close'], window=self.long_sma, fillna=False
        )
        return df
    
    def check_golden_cross(
        self,
        df: pd.DataFrame,
        current_idx: int,
        lookback_days: int = None
    ) -> Tuple[bool, Optional[float]]:
        """
        Check if a golden cross occurred within lookback_days.
        Returns (is_golden_cross, cross_price)
        """
        if lookback_days is None:
            lookback_days = self.golden_cross_days
        
        if current_idx < self.long_sma + lookback_days:
            return False, None
        
        # Get the actual row indices from the dataframe
        df_indices = df.index.tolist()
        if current_idx >= len(df_indices):
            return False, None
        
        # Look for the cross
        start_idx = max(0, current_idx - lookback_days)
        for i in range(start_idx, current_idx):
            if i == 0 or i >= len(df_indices):
                continue
            
            prev_row = df.iloc[i - 1]
            curr_row = df.iloc[i]
            
            prev_short = prev_row['sma_short']
            prev_long = prev_row['sma_long']
            curr_short = curr_row['sma_short']
            curr_long = curr_row['sma_long']
            
            # Skip NaN values
            if pd.isna(prev_short) or pd.isna(prev_long) or pd.isna(curr_short) or pd.isna(curr_long):
                continue
            
            # Golden cross: short SMA crosses above long SMA
            if prev_short <= prev_long and curr_short > curr_long:
                cross_price = curr_row['close']
                # Verify price is still rising
                current_price = df.iloc[current_idx]['close']
                if current_price > cross_price:
                    return True, cross_price
        
        return False, None
    
    def check_death_cross(
        self,
        df: pd.DataFrame,
        current_idx: int,
        lookback_days: int = 10
    ) -> Tuple[bool, Optional[float]]:
        """
        Check if a death cross occurred within lookback_days.
        Returns (is_death_cross, cross_price)
        """
        if current_idx < self.long_sma + lookback_days:
            return False, None
        
        # Get the actual row indices from the dataframe
        df_indices = df.index.tolist()
        if current_idx >= len(df_indices):
            return False, None
        
        start_idx = max(0, current_idx - lookback_days)
        for i in range(start_idx, current_idx):
            if i == 0 or i >= len(df_indices):
                continue
            
            prev_row = df.iloc[i - 1]
            curr_row = df.iloc[i]
            
            prev_short = prev_row['sma_short']
            prev_long = prev_row['sma_long']
            curr_short = curr_row['sma_short']
            curr_long = curr_row['sma_long']
            
            if pd.isna(prev_short) or pd.isna(prev_long) or pd.isna(curr_short) or pd.isna(curr_long):
                continue
            
            # Death cross: short SMA crosses below long SMA
            if prev_short >= prev_long and curr_short < curr_long:
                return True, curr_row['close']
        
        return False, None
    
    def check_stop_loss(
        self,
        position: Position,
        current_price: float
    ) -> bool:
        """Check if stop loss has been triggered"""
        if not self.use_stop_loss:
            return False
        
        pct_change = position.profit_loss_pct(current_price)
        return pct_change <= -self.stop_loss_pct
    
    def check_take_profit(
        self,
        position: Position,
        current_price: float
    ) -> bool:
        """Check if take profit has been triggered"""
        pct_change = position.profit_loss_pct(current_price)
        return pct_change >= self.take_profit_pct
    
    def check_sudden_drop(
        self,
        df: pd.DataFrame,
        current_idx: int,
        pct_threshold_1hr: float = 15.0,
        pct_threshold_2hr: float = 10.0
    ) -> bool:
        """Check for sudden price drops"""
        if current_idx < 2:
            return False
        
        current_price = df.iloc[current_idx]['close']
        
        # 1 period ago
        if current_idx >= 1:
            price_1_ago = df.iloc[current_idx - 1]['close']
            pct_change_1 = ((current_price - price_1_ago) / price_1_ago) * 100
            if pct_change_1 <= -pct_threshold_1hr:
                return True
        
        # 2 periods ago
        if current_idx >= 2:
            price_2_ago = df.iloc[current_idx - 2]['close']
            pct_change_2 = ((current_price - price_2_ago) / price_2_ago) * 100
            if pct_change_2 <= -pct_threshold_2hr:
                return True
        
        return False


class Backtester:
    """
    Main backtesting engine that simulates trading over historical data
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.0,  # Robinhood has no commission
        strategy: Optional[TradingStrategy] = None,
        data_provider: Optional[HistoricalDataProvider] = None
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.strategy = strategy or TradingStrategy(
            stop_loss_pct=stop_loss_percent,
            take_profit_pct=take_profit_percent,
            use_stop_loss=use_stop_loss,
            golden_cross_days=golden_cross_buy_days
        )
        self.data_provider = data_provider or HistoricalDataProvider()
        
        # State
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
    
    def reset(self):
        """Reset backtester state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
    
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        portfolio_value: float
    ) -> float:
        """Calculate how many shares to buy based on position sizing rules"""
        # Use purchase limit percentage
        if use_purchase_limit_percentage:
            max_position = portfolio_value / purchase_limit_percentage
        else:
            max_position = self.cash
        
        # Don't exceed available cash
        max_position = min(max_position, self.cash)
        
        # Calculate shares
        shares = int(max_position / price)
        
        return max(0, shares)
    
    def execute_buy(
        self,
        symbol: str,
        price: float,
        shares: float,
        timestamp: datetime,
        reason: str
    ) -> Optional[Trade]:
        """Execute a buy order"""
        total_cost = price * shares + self.commission
        
        if total_cost > self.cash or shares <= 0:
            return None
        
        self.cash -= total_cost
        
        # Add or update position
        if symbol in self.positions:
            # Average up/down
            existing = self.positions[symbol]
            total_shares = existing.shares + shares
            total_cost_basis = existing.total_cost + (price * shares)
            avg_price = total_cost_basis / total_shares
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=total_shares,
                avg_buy_price=avg_price,
                buy_timestamp=existing.buy_timestamp
            )
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=shares,
                avg_buy_price=price,
                buy_timestamp=timestamp
            )
        
        trade = Trade(
            symbol=symbol,
            trade_type=TradeType.BUY,
            price=price,
            shares=shares,
            timestamp=timestamp,
            reason=reason
        )
        self.trades.append(trade)
        
        return trade
    
    def execute_sell(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        reason: str
    ) -> Optional[Trade]:
        """Execute a sell order (sells entire position)"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        shares = position.shares
        total_value = price * shares - self.commission
        
        self.cash += total_value
        del self.positions[symbol]
        
        trade = Trade(
            symbol=symbol,
            trade_type=TradeType.SELL,
            price=price,
            shares=shares,
            timestamp=timestamp,
            reason=reason
        )
        self.trades.append(trade)
        
        return trade
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(
            pos.current_value(current_prices.get(symbol, pos.avg_buy_price))
            for symbol, pos in self.positions.items()
        )
        return self.cash + positions_value
    
    def run(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        verbose: bool = True
    ) -> BacktestResult:
        """
        Run backtest on specified symbols and date range
        """
        self.reset()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"BACKTESTING ROBINHOODBOT STRATEGY")
            print(f"{'='*60}")
            print(f"Period: {start_date.date()} to {end_date.date()}")
            print(f"Symbols: {', '.join(symbols)}")
            print(f"Initial Capital: ${self.initial_capital:,.2f}")
            print(f"Strategy: SMA({self.strategy.short_sma}/{self.strategy.long_sma})")
            print(f"Stop Loss: {self.strategy.stop_loss_pct}%")
            print(f"Take Profit: {self.strategy.take_profit_pct}%")
            print(f"{'='*60}\n")
        
        # Fetch all historical data
        symbol_data: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            if verbose:
                print(f"Fetching data for {symbol}...")
            df = self.data_provider.get_historical_data(
                symbol, start_date, end_date, interval='day'
            )
            if df is not None and not df.empty:
                df = self.strategy.calculate_indicators(df)
                symbol_data[symbol] = df
        
        if not symbol_data:
            print("No data available for backtesting")
            return self._generate_result(start_date, end_date, symbols)
        
        # Get all unique dates across all symbols
        all_dates = set()
        for df in symbol_data.values():
            all_dates.update(df['date'].tolist())
        all_dates = sorted(all_dates)
        
        if verbose:
            print(f"\nSimulating {len(all_dates)} trading days...\n")
        
        # Watchlist simulation - start with all symbols
        watchlist = set(symbols)
        
        # Main simulation loop
        for date in all_dates:
            current_prices = {}
            
            # Get current prices for all symbols
            for symbol, df in symbol_data.items():
                date_rows = df[df['date'] == date]
                if not date_rows.empty:
                    current_prices[symbol] = date_rows.iloc[0]['close']
            
            # Check existing positions for sell signals
            positions_to_check = list(self.positions.keys())
            for symbol in positions_to_check:
                if symbol not in current_prices:
                    continue
                
                price = current_prices[symbol]
                position = self.positions[symbol]
                df = symbol_data.get(symbol)
                
                if df is None:
                    continue
                
                date_idx = df[df['date'] == date].index
                if len(date_idx) == 0:
                    continue
                current_idx = date_idx[0]
                
                # Check sell conditions
                sell_reason = None
                
                # 1. Check stop loss
                if self.strategy.check_stop_loss(position, price):
                    sell_reason = SellReason.STOP_LOSS.value
                
                # 2. Check take profit
                elif self.strategy.check_take_profit(position, price):
                    sell_reason = SellReason.TAKE_PROFIT.value
                
                # 3. Check death cross
                elif self.strategy.check_death_cross(df, current_idx)[0]:
                    sell_reason = SellReason.DEATH_CROSS.value
                
                # 4. Check sudden drop
                elif self.strategy.check_sudden_drop(df, current_idx):
                    sell_reason = SellReason.SUDDEN_DROP.value
                
                if sell_reason:
                    trade = self.execute_sell(symbol, price, date, sell_reason)
                    if trade and verbose:
                        pnl = (price - position.avg_buy_price) * position.shares
                        pnl_pct = ((price - position.avg_buy_price) / position.avg_buy_price) * 100
                        print(f"  SELL {symbol}: {position.shares:.0f} shares @ ${price:.2f} "
                              f"[{sell_reason}] P/L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
                    watchlist.add(symbol)  # Add back to watchlist
            
            # Check watchlist for buy signals
            for symbol in list(watchlist):
                if symbol in self.positions:
                    continue
                
                if symbol not in current_prices:
                    continue
                
                price = current_prices[symbol]
                df = symbol_data.get(symbol)
                
                if df is None:
                    continue
                
                # Check price cap
                if use_price_cap and price > price_cap:
                    continue
                
                date_idx = df[df['date'] == date].index
                if len(date_idx) == 0:
                    continue
                current_idx = date_idx[0]
                
                # Check golden cross
                is_golden, cross_price = self.strategy.check_golden_cross(df, current_idx)
                
                if is_golden and cross_price:
                    # Verify price is rising
                    if price > cross_price:
                        portfolio_value = self.get_portfolio_value(current_prices)
                        shares = self.calculate_position_size(symbol, price, portfolio_value)
                        
                        if shares > 0:
                            trade = self.execute_buy(
                                symbol, price, shares, date,
                                BuyReason.GOLDEN_CROSS.value
                            )
                            if trade and verbose:
                                print(f"  BUY  {symbol}: {shares:.0f} shares @ ${price:.2f} "
                                      f"[{BuyReason.GOLDEN_CROSS.value}] Cost: ${trade.total_value:.2f}")
                            watchlist.discard(symbol)
            
            # Record equity
            portfolio_value = self.get_portfolio_value(current_prices)
            self.equity_curve.append((date, portfolio_value))
        
        # Close remaining positions at end of backtest
        if self.positions and symbol_data:
            if verbose:
                print("\nClosing remaining positions at end of backtest...")
            
            final_prices = {}
            for symbol, df in symbol_data.items():
                if not df.empty:
                    final_prices[symbol] = df.iloc[-1]['close']
            
            for symbol in list(self.positions.keys()):
                if symbol in final_prices:
                    self.execute_sell(
                        symbol, final_prices[symbol],
                        end_date, SellReason.END_OF_BACKTEST.value
                    )
        
        return self._generate_result(start_date, end_date, symbols)
    
    def _generate_result(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: List[str]
    ) -> BacktestResult:
        """Generate backtest result summary"""
        
        # Calculate metrics
        final_capital = self.cash
        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Win/Loss analysis
        winning_trades = 0
        losing_trades = 0
        total_wins = 0.0
        total_losses = 0.0
        
        # Match buys with sells
        buy_prices: Dict[str, float] = {}
        for trade in self.trades:
            if trade.trade_type == TradeType.BUY:
                buy_prices[trade.symbol] = trade.price
            elif trade.trade_type == TradeType.SELL:
                if trade.symbol in buy_prices:
                    buy_price = buy_prices[trade.symbol]
                    pnl = (trade.price - buy_price) * trade.shares
                    if pnl > 0:
                        winning_trades += 1
                        total_wins += pnl
                    else:
                        losing_trades += 1
                        total_losses += abs(pnl)
                    del buy_prices[trade.symbol]
        
        total_trades = winning_trades + losing_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_win = total_wins / winning_trades if winning_trades > 0 else 0
        avg_loss = total_losses / losing_trades if losing_trades > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Calculate max drawdown
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        peak = self.initial_capital
        
        for _, equity in self.equity_curve:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
        
        # Calculate daily returns for Sharpe ratio
        daily_returns = []
        for i in range(1, len(self.equity_curve)):
            prev_equity = self.equity_curve[i-1][1]
            curr_equity = self.equity_curve[i][1]
            if prev_equity > 0:
                daily_return = (curr_equity - prev_equity) / prev_equity
                daily_returns.append(daily_return)
        
        # Sharpe ratio (assuming 252 trading days, 0% risk-free rate)
        if daily_returns:
            avg_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            sharpe_ratio = (avg_return * np.sqrt(252)) / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        config_used = {
            'short_sma': self.strategy.short_sma,
            'long_sma': self.strategy.long_sma,
            'golden_cross_days': self.strategy.golden_cross_days,
            'stop_loss_pct': self.strategy.stop_loss_pct,
            'take_profit_pct': self.strategy.take_profit_pct,
            'use_stop_loss': self.strategy.use_stop_loss,
            'price_cap': price_cap if use_price_cap else None,
            'purchase_limit_pct': purchase_limit_percentage if use_purchase_limit_percentage else None
        }
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_return=total_return,
            total_return_pct=total_return_pct,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            trades=self.trades,
            equity_curve=self.equity_curve,
            daily_returns=daily_returns,
            symbols_tested=symbols,
            config_used=config_used
        )


def print_result(result: BacktestResult):
    """Print backtest results in a formatted manner"""
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS")
    print(f"{'='*60}")
    
    print(f"\n📅 Period: {result.start_date.date()} to {result.end_date.date()}")
    print(f"📈 Symbols: {', '.join(result.symbols_tested)}")
    
    print(f"\n💰 FINANCIAL SUMMARY")
    print(f"   Initial Capital:  ${result.initial_capital:>12,.2f}")
    print(f"   Final Capital:    ${result.final_capital:>12,.2f}")
    print(f"   Total Return:     ${result.total_return:>+12,.2f} ({result.total_return_pct:+.2f}%)")
    
    print(f"\n📊 PERFORMANCE METRICS")
    print(f"   Max Drawdown:     ${result.max_drawdown:>12,.2f} ({result.max_drawdown_pct:.2f}%)")
    print(f"   Sharpe Ratio:     {result.sharpe_ratio:>12.4f}")
    
    print(f"\n🎯 TRADING STATISTICS")
    print(f"   Total Trades:     {result.total_trades:>12}")
    print(f"   Winning Trades:   {result.winning_trades:>12}")
    print(f"   Losing Trades:    {result.losing_trades:>12}")
    print(f"   Win Rate:         {result.win_rate:>12.2f}%")
    print(f"   Avg Win:          ${result.avg_win:>12,.2f}")
    print(f"   Avg Loss:         ${result.avg_loss:>12,.2f}")
    print(f"   Profit Factor:    {result.profit_factor:>12.2f}")
    
    print(f"\n⚙️  STRATEGY CONFIGURATION")
    for key, value in result.config_used.items():
        print(f"   {key}: {value}")
    
    print(f"\n{'='*60}\n")


def save_result(result: BacktestResult, filepath: str = "backtest_result.json"):
    """Save backtest results to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"Results saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description='Backtest RobinhoodBot trading strategy'
    )
    
    parser.add_argument(
        '--symbols', '-s',
        type=str,
        help='Comma-separated list of symbols to test (e.g., AAPL,MSFT,GOOGL)'
    )
    
    parser.add_argument(
        '--start', '-st',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end', '-e',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=365,
        help='Number of days to backtest (default: 365)'
    )
    
    parser.add_argument(
        '--capital', '-c',
        type=float,
        default=10000.0,
        help='Initial capital (default: 10000)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='backtest_result.json',
        help='Output file for results (default: backtest_result.json)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )
    
    parser.add_argument(
        '--short-sma',
        type=int,
        default=20,
        help='Short SMA period (default: 20)'
    )
    
    parser.add_argument(
        '--long-sma',
        type=int,
        default=50,
        help='Long SMA period (default: 50)'
    )
    
    parser.add_argument(
        '--stop-loss',
        type=float,
        default=stop_loss_percent,
        help=f'Stop loss percentage (default: {stop_loss_percent})'
    )
    
    parser.add_argument(
        '--take-profit',
        type=float,
        default=take_profit_percent,
        help=f'Take profit percentage (default: {take_profit_percent})'
    )
    
    parser.add_argument(
        '--sample-data',
        type=str,
        help='Directory containing sample data files (from sample_data_generator.py)'
    )
    
    parser.add_argument(
        '--no-api',
        action='store_true',
        help='Disable API calls (use only cached/sample data)'
    )
    
    args = parser.parse_args()
    
    # Determine symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        # Default test symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        print(f"No symbols specified, using defaults: {', '.join(symbols)}")
    
    # Determine date range
    if args.end:
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    else:
        end_date = datetime.now()
    
    if args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
    else:
        start_date = end_date - timedelta(days=args.days)
    
    # Create strategy
    strategy = TradingStrategy(
        short_sma=args.short_sma,
        long_sma=args.long_sma,
        golden_cross_days=golden_cross_buy_days,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        use_stop_loss=use_stop_loss
    )
    
    # Create data provider
    data_provider = HistoricalDataProvider(
        sample_data_dir=args.sample_data
    )
    
    # Create backtester
    backtester = Backtester(
        initial_capital=args.capital,
        strategy=strategy,
        data_provider=data_provider
    )
    
    # Run backtest
    result = backtester.run(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        verbose=not args.quiet
    )
    
    # Print and save results
    print_result(result)
    save_result(result, args.output)


if __name__ == '__main__':
    main()
