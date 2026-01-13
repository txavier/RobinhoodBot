#!/usr/bin/env python3
"""
Intraday Backtesting Module for RobinhoodBot

This module simulates day trading with hourly data, ACCURATELY matching the real app's
behavior from main.py including:
- SMA crossovers on hourly candles (default 20/50)
- Dynamic SMA adjustment based on market conditions
- Market trend filtering (uptrend/downtrend)
- Profit before EOD selling
- No buying after 1:30pm
- Price > 5 hours ago check
- Slope ordering for buy priority
- Correct sudden drop thresholds (10%/2hr, 15%/1hr)

Usage:
    python backtest_intraday.py --symbols AAPL,MSFT,GOOGL --days 30
    python backtest_intraday.py --sample-data sample_data/ --days 60
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from scipy.stats import linregress

import pandas as pd
import numpy as np
import ta as t

# Import config settings
try:
    from config import (
        stop_loss_percent, take_profit_percent, use_stop_loss,
        golden_cross_buy_days, price_cap, use_price_cap,
        min_volume, min_market_cap, purchase_limit_percentage,
        use_purchase_limit_percentage, investing, version
    )
except ImportError:
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


class MarketCondition(Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    MAJOR_DOWNTREND = "major_downtrend"
    NEUTRAL = "neutral"


class SellReason(Enum):
    DEATH_CROSS = "death_cross"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    SUDDEN_DROP = "sudden_drop"
    PROFIT_BEFORE_EOD = "profit_before_eod"  # NEW: matches main.py
    END_OF_DAY = "end_of_day"
    END_OF_BACKTEST = "end_of_backtest"


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
    market_condition: str = ""  # NEW: track market state at trade time
    
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
    traded_today: bool = False  # NEW: for dynamic SMA adjustment
    took_profit_today: bool = False  # NEW: for dynamic SMA adjustment
    
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
class DailyStats:
    """Track daily trading statistics"""
    date: datetime
    trades_count: int = 0
    buys: int = 0
    sells: int = 0
    profit_loss: float = 0.0
    start_equity: float = 0.0
    end_equity: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    market_condition: str = ""  # NEW: track market trend for the day


def generate_market_data(
    start_date: datetime,
    days: int,
    seed: int = None
) -> pd.DataFrame:
    """
    Generate synthetic market index data (like SPY/DIA/NDAQ).
    
    This simulates the overall market conditions that main.py checks via:
    - is_market_in_uptrend(): Today's close > today's open for 2+ of 3 indices
    - is_market_in_major_downtrend(): Week open > current close for 2+ of 3 indices
    
    Returns DataFrame with daily market conditions.
    """
    if seed is not None:
        np.random.seed(seed + 999)  # Different seed for market data
    
    data = []
    current_date = start_date
    
    # Track weekly opens for major downtrend detection
    week_start_price = 100.0
    market_price = 100.0
    
    for day in range(days + 60):  # Extra days for warmup
        # Skip weekends
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        
        # Reset week start on Monday
        if current_date.weekday() == 0:
            week_start_price = market_price
        
        # Market tends to be more trendy - periods of up and down
        cycle_period = 15  # 15 day cycles
        cycle_day = day % cycle_period
        
        if cycle_day < 4:
            # Market uptrend period
            daily_return = np.random.normal(0.005, 0.01)
            intraday_bias = 0.6  # 60% chance of up day
        elif cycle_day < 8:
            # Consolidation
            daily_return = np.random.normal(0.0, 0.008)
            intraday_bias = 0.5
        elif cycle_day < 12:
            # Market downtrend period  
            daily_return = np.random.normal(-0.004, 0.012)
            intraday_bias = 0.35
        else:
            # Recovery
            daily_return = np.random.normal(0.002, 0.009)
            intraday_bias = 0.55
        
        open_price = market_price
        close_price = market_price * (1 + daily_return)
        
        # Simulate intraday: is close > open? (for is_market_in_uptrend)
        intraday_up = np.random.random() < intraday_bias
        if intraday_up:
            # Force close > open
            if close_price < open_price:
                close_price = open_price * (1 + abs(daily_return))
        else:
            # Force close < open  
            if close_price > open_price:
                close_price = open_price * (1 - abs(daily_return))
        
        # is_market_in_uptrend: today's close > today's open
        is_uptrend = close_price > open_price
        
        # is_market_in_major_downtrend: week open > current close
        is_major_downtrend = week_start_price > close_price * 1.02  # 2% down from week start
        
        data.append({
            'date': current_date.date(),
            'market_open': round(open_price, 2),
            'market_close': round(close_price, 2),
            'week_start': round(week_start_price, 2),
            'is_uptrend': is_uptrend,
            'is_major_downtrend': is_major_downtrend,
            'market_condition': (
                MarketCondition.MAJOR_DOWNTREND.value if is_major_downtrend else
                MarketCondition.UPTREND.value if is_uptrend else
                MarketCondition.DOWNTREND.value
            )
        })
        
        market_price = close_price
        current_date += timedelta(days=1)
    
    return pd.DataFrame(data)


def generate_intraday_data(
    symbol: str,
    start_date: datetime,
    days: int,
    initial_price: float = 100.0,
    hourly_volatility: float = 0.005,  # 0.5% per hour typical
    daily_drift: float = 0.0001,
    avg_hourly_volume: int = 100000,
    seed: int = None
) -> pd.DataFrame:
    """
    Generate synthetic intraday (hourly) stock data.
    
    Trading hours: 9:30 AM - 4:00 PM = 7 hours per day (we'll use 7 hourly candles)
    
    Returns DataFrame with: datetime, open, high, low, close, volume
    """
    if seed is not None:
        np.random.seed(seed)
    
    data = []
    current_price = initial_price
    current_date = start_date
    
    trading_hours = [9, 10, 11, 12, 13, 14, 15]  # 9am to 3pm (close at 4pm)
    
    for day in range(days):
        # Skip weekends
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        
        # Daily trend (some days up, some down)
        daily_trend = np.random.normal(daily_drift, 0.005)
        
        for hour in trading_hours:
            timestamp = current_date.replace(hour=hour, minute=30, second=0)
            
            # Hourly price movement
            hourly_return = np.random.normal(daily_trend / len(trading_hours), hourly_volatility)
            
            open_price = current_price
            close_price = open_price * (1 + hourly_return)
            
            # Generate high/low
            intra_hour_vol = abs(hourly_return) * 0.5
            high = max(open_price, close_price) * (1 + np.random.uniform(0, intra_hour_vol))
            low = min(open_price, close_price) * (1 - np.random.uniform(0, intra_hour_vol))
            
            # Volume varies by hour (higher at open and close)
            hour_idx = trading_hours.index(hour)
            if hour_idx == 0 or hour_idx == len(trading_hours) - 1:
                volume_mult = np.random.uniform(1.5, 2.5)
            else:
                volume_mult = np.random.uniform(0.7, 1.3)
            volume = int(avg_hourly_volume * volume_mult)
            
            data.append({
                'datetime': timestamp,
                'date': current_date.date(),
                'hour': hour,
                'open': round(max(0.01, open_price), 2),
                'high': round(max(0.01, high), 2),
                'low': round(max(0.01, low), 2),
                'close': round(max(0.01, close_price), 2),
                'volume': volume
            })
            
            current_price = close_price
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(data)


def generate_golden_cross_intraday(
    symbol: str,
    start_date: datetime,
    days: int,
    initial_price: float = 100.0,
    seed: int = None
) -> pd.DataFrame:
    """
    Generate intraday data that creates golden cross opportunities.
    Pattern: slight downtrend -> consolidation -> uptrend with cross -> pullback -> repeat
    """
    if seed is not None:
        np.random.seed(seed)
    
    all_data = []
    current_price = initial_price
    current_date = start_date
    
    trading_hours = [9, 10, 11, 12, 13, 14, 15]
    days_generated = 0
    cycle_day = 0
    
    while days_generated < days:
        # Skip weekends
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        
        # Cycle: 5 days down, 3 days flat, 7 days up, 5 days profit-taking
        cycle_length = 20
        cycle_day = days_generated % cycle_length
        
        if cycle_day < 5:
            # Downtrend phase
            daily_trend = np.random.uniform(-0.008, -0.002)
            hourly_vol = 0.004
        elif cycle_day < 8:
            # Consolidation
            daily_trend = np.random.uniform(-0.002, 0.002)
            hourly_vol = 0.003
        elif cycle_day < 15:
            # Uptrend (golden cross forms here)
            daily_trend = np.random.uniform(0.003, 0.012)
            hourly_vol = 0.005
        else:
            # Profit-taking / mild pullback
            daily_trend = np.random.uniform(-0.004, 0.002)
            hourly_vol = 0.004
        
        for hour in trading_hours:
            timestamp = current_date.replace(hour=hour, minute=30, second=0)
            
            hourly_return = np.random.normal(daily_trend / len(trading_hours), hourly_vol)
            
            open_price = current_price
            close_price = open_price * (1 + hourly_return)
            
            intra_vol = abs(hourly_return) * 0.5
            high = max(open_price, close_price) * (1 + np.random.uniform(0, intra_vol))
            low = min(open_price, close_price) * (1 - np.random.uniform(0, intra_vol))
            
            hour_idx = trading_hours.index(hour)
            if hour_idx == 0 or hour_idx == len(trading_hours) - 1:
                volume = int(150000 * np.random.uniform(1.5, 2.5))
            else:
                volume = int(150000 * np.random.uniform(0.7, 1.3))
            
            all_data.append({
                'datetime': timestamp,
                'date': current_date.date(),
                'hour': hour,
                'open': round(max(0.01, open_price), 2),
                'high': round(max(0.01, high), 2),
                'low': round(max(0.01, low), 2),
                'close': round(max(0.01, close_price), 2),
                'volume': volume
            })
            
            current_price = close_price
        
        current_date += timedelta(days=1)
        days_generated += 1
    
    return pd.DataFrame(all_data)


class IntradayTradingStrategy:
    """
    Day trading strategy using hourly SMA crossovers.
    
    ACCURATELY matches main.py behavior:
    - Uses SMA(20/50) on hourly data (not daily!)
    - Dynamic SMA adjustment: n1=14 in downtrend, n1=5/n2=7 if take_profit+traded_today
    - Market trend filtering (must be uptrend to buy)
    - No buying after 1:30pm (EOD inflection protection)
    - Price must be > price 5 hours ago
    - profit_before_eod: sell profitable positions after 1:30pm
    - Sudden drop: 10% in 2hr OR 15% in 1hr (matching main.py)
    """
    
    def __init__(
        self,
        short_sma: int = 20,  # 20 hours ≈ ~3 trading days
        long_sma: int = 50,   # 50 hours ≈ ~7 trading days
        golden_cross_hours: int = 24,  # Look back 24 hours for cross (matches golden_cross_buy_days * 7hrs)
        stop_loss_pct: float = 5.0,
        take_profit_pct: float = 0.70,
        use_stop_loss: bool = True,
        close_at_eod: bool = False,
        # NEW: Main.py matching parameters
        use_market_filter: bool = True,  # Require market uptrend to buy
        use_eod_filter: bool = True,  # No buying after 1:30pm
        use_profit_before_eod: bool = True,  # Sell profitable positions after 1:30pm
        use_price_5hr_check: bool = True,  # Price must be > 5hr ago
        use_dynamic_sma: bool = True,  # Adjust SMA based on conditions
        use_slope_ordering: bool = True,  # Prioritize by price slope
        use_price_cap: bool = True,  # Max price filter
        price_cap_value: float = 2100.0,
        slope_threshold: float = 0.0008  # Min slope to consider (from main.py order_symbols_by_slope)
    ):
        self.short_sma = short_sma
        self.long_sma = long_sma
        self.golden_cross_hours = golden_cross_hours
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.use_stop_loss = use_stop_loss
        self.close_at_eod = close_at_eod
        
        # Main.py matching features
        self.use_market_filter = use_market_filter
        self.use_eod_filter = use_eod_filter
        self.use_profit_before_eod = use_profit_before_eod
        self.use_price_5hr_check = use_price_5hr_check
        self.use_dynamic_sma = use_dynamic_sma
        self.use_slope_ordering = use_slope_ordering
        self.use_price_cap = use_price_cap
        self.price_cap_value = price_cap_value
        self.slope_threshold = slope_threshold
    
    def get_dynamic_sma_periods(
        self,
        market_in_downtrend: bool,
        took_profit_today: bool,
        traded_today: bool
    ) -> Tuple[int, int]:
        """
        Get dynamic SMA periods based on market conditions.
        
        Matches main.py logic:
        - Default: n1=20, n2=50
        - If market in downtrend (not uptrend): n1=14
        - If took_profit AND traded_today: n1=5, n2=7 (more aggressive)
        """
        if not self.use_dynamic_sma:
            return self.short_sma, self.long_sma
        
        n1, n2 = self.short_sma, self.long_sma
        
        if took_profit_today and traded_today:
            # More aggressive after taking profit
            n1, n2 = 5, 7
        elif market_in_downtrend:
            # More conservative in downtrend
            n1 = 14
        
        return n1, n2
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate SMA indicators on hourly data"""
        df = df.copy()
        df['sma_short'] = df['close'].rolling(window=self.short_sma, min_periods=1).mean()
        df['sma_long'] = df['close'].rolling(window=self.long_sma, min_periods=1).mean()
        df['sma_diff'] = df['sma_short'] - df['sma_long']
        df['sma_diff_prev'] = df['sma_diff'].shift(1)
        
        # Also calculate alternative SMA for downtrend (n1=14)
        df['sma_short_14'] = df['close'].rolling(window=14, min_periods=1).mean()
        
        # And aggressive SMA for post-profit (n1=5, n2=7)
        df['sma_short_5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['sma_long_7'] = df['close'].rolling(window=7, min_periods=1).mean()
        
        return df
    
    def calculate_slope(self, df: pd.DataFrame, current_idx: int, lookback_hours: int = 7) -> float:
        """
        Calculate price slope for the current trading day.
        
        Matches main.py order_symbols_by_slope() which uses linregress on 5min data.
        We simulate with hourly data for the current day.
        """
        if current_idx < lookback_hours:
            return 0.0
        
        start_idx = max(0, current_idx - lookback_hours)
        prices = df.iloc[start_idx:current_idx + 1]['close'].values
        
        if len(prices) < 2:
            return 0.0
        
        x = np.arange(len(prices))
        try:
            result = linregress(x, prices)
            return result.slope
        except:
            return 0.0
    
    def is_eod(self, timestamp: datetime) -> bool:
        """
        Check if it's End of Day (EOD) - after 1:30pm.
        
        Matches main.py is_eod():
        - Returns True if time >= 13:30 and < 16:00
        - No buying after 1:30pm due to EOD price inflection
        """
        return 13 <= timestamp.hour < 16 and (timestamp.hour > 13 or timestamp.minute >= 30)
    
    def check_profit_before_eod(
        self,
        position: Position,
        current_price: float,
        timestamp: datetime
    ) -> bool:
        """
        Check if we should sell for profit before EOD.
        
        Matches main.py profit_before_eod():
        - If it's EOD time (after 1:30pm) AND position is profitable, sell
        - This captures gains before EOD price fluctuations
        """
        if not self.use_profit_before_eod:
            return False
        
        if not self.is_eod(timestamp):
            return False
        
        # Check if position is profitable
        pnl_pct = position.profit_loss_pct(current_price)
        return pnl_pct > 0
    
    def check_price_higher_than_5hr_ago(
        self,
        df: pd.DataFrame,
        current_idx: int
    ) -> bool:
        """
        Check if current price > price 5 hours ago.
        
        Matches main.py buy condition: cross[2] > cross[3]
        where cross[3] is price 5 hours ago
        """
        if not self.use_price_5hr_check:
            return True
        
        if current_idx < 5:
            return False
        
        current_price = df.iloc[current_idx]['close']
        price_5hr_ago = df.iloc[current_idx - 5]['close']
        
        return current_price > price_5hr_ago
    
    def check_golden_cross(
        self,
        df: pd.DataFrame,
        current_idx: int,
        lookback_hours: int = None,
        n1: int = None,
        n2: int = None
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        """
        Check if a golden cross occurred within lookback_hours.
        Golden cross = short SMA crosses above long SMA
        
        Returns:
            (is_golden_cross, cross_price, price_5hr_ago)
        """
        if lookback_hours is None:
            lookback_hours = self.golden_cross_hours
        
        if n1 is None:
            n1 = self.short_sma
        if n2 is None:
            n2 = self.long_sma
        
        if current_idx < max(n1, n2):
            return False, None, None
        
        if current_idx >= len(df):
            return False, None, None
        
        # Get SMA columns based on periods
        if n1 == 14:
            sma_short_col = 'sma_short_14'
            sma_long_col = 'sma_long'
        elif n1 == 5 and n2 == 7:
            sma_short_col = 'sma_short_5'
            sma_long_col = 'sma_long_7'
        else:
            sma_short_col = 'sma_short'
            sma_long_col = 'sma_long'
        
        # Current state: short must be above long
        current_row = df.iloc[current_idx]
        if pd.isna(current_row.get(sma_short_col)) or pd.isna(current_row.get(sma_long_col)):
            return False, None, None
        
        if current_row[sma_short_col] <= current_row[sma_long_col]:
            return False, None, None
        
        # Look for the cross point
        start_idx = max(0, current_idx - lookback_hours)
        for i in range(start_idx, current_idx):
            if i == 0:
                continue
            
            prev_row = df.iloc[i - 1]
            curr_row = df.iloc[i]
            
            if pd.isna(prev_row.get(sma_short_col)) or pd.isna(prev_row.get(sma_long_col)):
                continue
            if pd.isna(curr_row.get(sma_short_col)) or pd.isna(curr_row.get(sma_long_col)):
                continue
            
            # Cross: prev short <= prev long AND curr short > curr long
            if prev_row[sma_short_col] <= prev_row[sma_long_col] and curr_row[sma_short_col] > curr_row[sma_long_col]:
                cross_price = curr_row['close']
                current_price = df.iloc[current_idx]['close']
                
                # Get price 5 hours ago
                price_5hr_ago = df.iloc[max(0, current_idx - 5)]['close'] if current_idx >= 5 else current_price
                
                # Only buy if price is still rising (current > cross price)
                if current_price >= cross_price:
                    return True, cross_price, price_5hr_ago
        
        return False, None, None
    
    def check_death_cross(
        self,
        df: pd.DataFrame,
        current_idx: int,
        lookback_hours: int = 24,
        n1: int = None,
        n2: int = None
    ) -> Tuple[bool, Optional[float]]:
        """Check if a death cross occurred (short SMA crosses below long SMA)"""
        if n1 is None:
            n1 = self.short_sma
        if n2 is None:
            n2 = self.long_sma
            
        if current_idx < max(n1, n2):
            return False, None
        
        if current_idx >= len(df):
            return False, None
        
        # Get SMA columns based on periods
        if n1 == 14:
            sma_short_col = 'sma_short_14'
            sma_long_col = 'sma_long'
        elif n1 == 5 and n2 == 7:
            sma_short_col = 'sma_short_5'
            sma_long_col = 'sma_long_7'
        else:
            sma_short_col = 'sma_short'
            sma_long_col = 'sma_long'
        
        start_idx = max(0, current_idx - lookback_hours)
        for i in range(start_idx, current_idx):
            if i == 0:
                continue
            
            prev_row = df.iloc[i - 1]
            curr_row = df.iloc[i]
            
            if pd.isna(prev_row.get(sma_short_col)) or pd.isna(prev_row.get(sma_long_col)):
                continue
            if pd.isna(curr_row.get(sma_short_col)) or pd.isna(curr_row.get(sma_long_col)):
                continue
            
            # Death cross: prev short >= prev long AND curr short < curr long
            if prev_row[sma_short_col] >= prev_row[sma_long_col] and curr_row[sma_short_col] < curr_row[sma_long_col]:
                return True, curr_row['close']
        
        return False, None
    
    def check_stop_loss(self, position: Position, current_price: float) -> bool:
        """Check if stop loss triggered"""
        if not self.use_stop_loss:
            return False
        pct_change = position.profit_loss_pct(current_price)
        return pct_change <= -self.stop_loss_pct
    
    def check_take_profit(self, position: Position, current_price: float) -> bool:
        """Check if take profit triggered"""
        pct_change = position.profit_loss_pct(current_price)
        return pct_change >= self.take_profit_pct
    
    def check_sudden_drop(
        self,
        df: pd.DataFrame,
        current_idx: int,
        pct_threshold_1hr: float = 15.0,  # FIXED: main.py uses 15% in 1hr
        pct_threshold_2hr: float = 10.0   # FIXED: main.py uses 10% in 2hr
    ) -> bool:
        """
        Check for sudden price drops.
        
        Matches main.py sudden_drop():
        - 10% drop in 2 hours OR
        - 15% drop in 1 hour
        """
        if current_idx < 2 or current_idx >= len(df):
            return False
        
        current_price = df.iloc[current_idx]['close']
        
        # 1 hour ago - check for 15% drop
        if current_idx >= 1:
            price_1_ago = df.iloc[current_idx - 1]['close']
            pct_change_1 = ((current_price - price_1_ago) / price_1_ago) * 100
            if pct_change_1 <= -pct_threshold_1hr:
                return True
        
        # 2 hours ago - check for 10% drop
        if current_idx >= 2:
            price_2_ago = df.iloc[current_idx - 2]['close']
            pct_change_2 = ((current_price - price_2_ago) / price_2_ago) * 100
            if pct_change_2 <= -pct_threshold_2hr:
                return True
        
        return False
class IntradayBacktester:
    """
    Intraday backtesting engine for day trading simulation.
    
    ACCURATELY simulates main.py behavior:
    - Market trend filtering (uptrend required for buys)
    - Dynamic SMA adjustment
    - Profit before EOD selling
    - No buying after 1:30pm
    - Slope-based symbol ordering
    - Price cap filtering
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_positions: int = 5,
        max_position_pct: float = 20.0,  # Max % of portfolio per position
        strategy: Optional[IntradayTradingStrategy] = None,
        day_trade_limit: int = 3  # Pattern day trader limit
    ):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct
        self.strategy = strategy or IntradayTradingStrategy()
        self.day_trade_limit = day_trade_limit
        
        # State
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_stats: List[DailyStats] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        # NEW: Track daily state for dynamic SMA
        self.day_trades_today: int = 0
        self.took_profit_today: bool = False
        self.symbols_traded_today: set = set()
    
    def reset(self):
        """Reset backtester state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.daily_stats = []
        self.equity_curve = []
        self.day_trades_today = 0
        self.took_profit_today = False
        self.symbols_traded_today = set()
    
    def reset_daily_state(self):
        """Reset state at start of each trading day"""
        self.day_trades_today = 0
        self.took_profit_today = False
        self.symbols_traded_today = set()
        
        # Reset position daily flags
        for pos in self.positions.values():
            pos.traded_today = False
            pos.took_profit_today = False
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(
            pos.current_value(current_prices.get(symbol, pos.avg_buy_price))
            for symbol, pos in self.positions.items()
        )
        return self.cash + positions_value
    
    def calculate_position_size(self, price: float) -> int:
        """Calculate shares to buy based on position sizing"""
        portfolio_value = self.cash + sum(p.total_cost for p in self.positions.values())
        max_position_value = portfolio_value * (self.max_position_pct / 100)
        max_from_cash = self.cash * 0.95  # Keep 5% reserve
        
        position_value = min(max_position_value, max_from_cash)
        shares = int(position_value / price)
        
        return max(0, shares)
    
    def execute_buy(
        self,
        symbol: str,
        price: float,
        shares: int,
        timestamp: datetime,
        reason: str,
        market_condition: str = ""
    ) -> Optional[Trade]:
        """Execute a buy order"""
        if shares <= 0:
            return None
        
        total_cost = price * shares
        if total_cost > self.cash:
            return None
        
        self.cash -= total_cost
        
        self.positions[symbol] = Position(
            symbol=symbol,
            shares=shares,
            avg_buy_price=price,
            buy_timestamp=timestamp,
            traded_today=True
        )
        
        # Track day trade
        self.symbols_traded_today.add(symbol)
        self.day_trades_today += 1
        
        trade = Trade(
            symbol=symbol,
            trade_type=TradeType.BUY,
            price=price,
            shares=shares,
            timestamp=timestamp,
            reason=reason,
            market_condition=market_condition
        )
        self.trades.append(trade)
        return trade
    
    def execute_sell(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        reason: str,
        market_condition: str = ""
    ) -> Optional[Trade]:
        """Execute a sell order"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        shares = position.shares
        total_value = price * shares
        
        self.cash += total_value
        
        # Track if this was a take profit
        if reason == SellReason.TAKE_PROFIT.value:
            self.took_profit_today = True
        
        del self.positions[symbol]
        
        trade = Trade(
            symbol=symbol,
            trade_type=TradeType.SELL,
            price=price,
            shares=shares,
            timestamp=timestamp,
            reason=reason,
            market_condition=market_condition
        )
        self.trades.append(trade)
        return trade
    
    def run(
        self,
        symbols: List[str],
        days: int = 30,
        start_date: datetime = None,
        seed: int = None,
        verbose: bool = True
    ) -> Dict:
        """
        Run intraday backtest with generated data.
        
        ACCURATELY simulates main.py behavior including:
        - Market trend filtering
        - Dynamic SMA adjustment
        - Profit before EOD selling
        - No buying after 1:30pm
        - Slope-based ordering
        
        Args:
            symbols: List of stock symbols
            days: Number of trading days to simulate
            start_date: Starting date (defaults to today - days)
            seed: Random seed for reproducibility
            verbose: Print trade details
        
        Returns:
            Dictionary with backtest results
        """
        self.reset()
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days + 50)  # Extra days for SMA warmup
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"INTRADAY BACKTESTING - ACCURATE MAIN.PY SIMULATION")
            print(f"{'='*70}")
            print(f"Symbols: {', '.join(symbols)}")
            print(f"Trading Days: {days}")
            print(f"Initial Capital: ${self.initial_capital:,.2f}")
            print(f"Strategy: Hourly SMA({self.strategy.short_sma}/{self.strategy.long_sma})")
            print(f"Take Profit: {self.strategy.take_profit_pct}%")
            print(f"Stop Loss: {self.strategy.stop_loss_pct}%")
            print(f"\n⚙️  MAIN.PY MATCHING FEATURES:")
            print(f"   Market Filter: {'ON' if self.strategy.use_market_filter else 'OFF'}")
            print(f"   EOD Filter (no buy after 1:30pm): {'ON' if self.strategy.use_eod_filter else 'OFF'}")
            print(f"   Profit Before EOD Sell: {'ON' if self.strategy.use_profit_before_eod else 'OFF'}")
            print(f"   Price > 5hr Ago Check: {'ON' if self.strategy.use_price_5hr_check else 'OFF'}")
            print(f"   Dynamic SMA: {'ON' if self.strategy.use_dynamic_sma else 'OFF'}")
            print(f"   Slope Ordering: {'ON' if self.strategy.use_slope_ordering else 'OFF'}")
            print(f"   Price Cap (${self.strategy.price_cap_value}): {'ON' if self.strategy.use_price_cap else 'OFF'}")
            print(f"{'='*70}\n")
        
        # Generate market data for trend filtering
        market_data = generate_market_data(start_date, days + 60, seed)
        
        # Generate intraday data for each symbol
        symbol_data: Dict[str, pd.DataFrame] = {}
        for i, symbol in enumerate(symbols):
            if verbose:
                print(f"Generating intraday data for {symbol}...")
            
            sym_seed = seed + i if seed else None
            df = generate_golden_cross_intraday(
                symbol,
                start_date,
                days + 50,  # Extra for SMA warmup
                initial_price=100 + i * 50,  # Different starting prices
                seed=sym_seed
            )
            df = self.strategy.calculate_indicators(df)
            symbol_data[symbol] = df
        
        # Get unique timestamps across all data
        all_timestamps = set()
        for df in symbol_data.values():
            all_timestamps.update(df['datetime'].tolist())
        all_timestamps = sorted(all_timestamps)
        
        # Skip warmup period
        warmup_hours = self.strategy.long_sma + 10
        if len(all_timestamps) > warmup_hours:
            all_timestamps = all_timestamps[warmup_hours:]
        
        if verbose:
            trading_days = len(set(ts.date() for ts in all_timestamps))
            print(f"\nSimulating {len(all_timestamps)} hourly bars ({trading_days} trading days)...\n")
        
        current_date = None
        daily_trades = 0
        daily_pnl = 0.0
        
        # Stats tracking
        rejected_buys = {
            'market_downtrend': 0,
            'major_downtrend': 0,
            'eod_filter': 0,
            'price_5hr': 0,
            'slope_filter': 0,
            'price_cap': 0,
            'day_trade_limit': 0
        }
        
        # Main simulation loop
        for timestamp in all_timestamps:
            # Track daily stats and reset daily state
            if current_date != timestamp.date():
                if current_date is not None and verbose:
                    portfolio_value = self.get_portfolio_value({
                        sym: df[df['datetime'] == timestamp].iloc[0]['close']
                        if not df[df['datetime'] == timestamp].empty else 100
                        for sym, df in symbol_data.items()
                    })
                    
                    # Get market condition for logging
                    market_row = market_data[market_data['date'] == current_date]
                    market_status = market_row.iloc[0]['market_condition'] if not market_row.empty else 'unknown'
                    
                    if daily_trades > 0:
                        print(f"  📊 Day {current_date} [{market_status}]: {daily_trades} trades, P/L: ${daily_pnl:+.2f}, Portfolio: ${portfolio_value:,.2f}")
                
                current_date = timestamp.date()
                daily_trades = 0
                daily_pnl = 0.0
                self.reset_daily_state()
            
            # Get market condition for this day
            market_row = market_data[market_data['date'] == current_date]
            if market_row.empty:
                market_uptrend = True
                market_major_downtrend = False
                market_condition = MarketCondition.NEUTRAL.value
            else:
                market_uptrend = market_row.iloc[0]['is_uptrend']
                market_major_downtrend = market_row.iloc[0]['is_major_downtrend']
                market_condition = market_row.iloc[0]['market_condition']
            
            # Get dynamic SMA periods based on market condition
            n1, n2 = self.strategy.get_dynamic_sma_periods(
                market_in_downtrend=not market_uptrend,
                took_profit_today=self.took_profit_today,
                traded_today=len(self.symbols_traded_today) > 0
            )
            
            current_prices = {}
            for symbol, df in symbol_data.items():
                ts_rows = df[df['datetime'] == timestamp]
                if not ts_rows.empty:
                    current_prices[symbol] = ts_rows.iloc[0]['close']
            
            # Record equity
            portfolio_value = self.get_portfolio_value(current_prices)
            self.equity_curve.append((timestamp, portfolio_value))
            
            # Check positions for sell signals
            positions_to_check = list(self.positions.keys())
            for symbol in positions_to_check:
                if symbol not in current_prices:
                    continue
                
                price = current_prices[symbol]
                position = self.positions[symbol]
                df = symbol_data.get(symbol)
                
                if df is None:
                    continue
                
                ts_idx = df[df['datetime'] == timestamp].index
                if len(ts_idx) == 0:
                    continue
                current_idx = ts_idx[0]
                
                # Check sell conditions (priority order matching main.py)
                sell_reason = None
                
                # 1. Stop loss (highest priority)
                if self.strategy.check_stop_loss(position, price):
                    sell_reason = SellReason.STOP_LOSS.value
                
                # 2. Take profit
                elif self.strategy.check_take_profit(position, price):
                    sell_reason = SellReason.TAKE_PROFIT.value
                
                # 3. Sudden drop (10% in 2hr OR 15% in 1hr)
                elif self.strategy.check_sudden_drop(df, current_idx):
                    sell_reason = SellReason.SUDDEN_DROP.value
                
                # 4. Profit before EOD (NEW - matches main.py)
                elif self.strategy.check_profit_before_eod(position, price, timestamp):
                    sell_reason = SellReason.PROFIT_BEFORE_EOD.value
                
                # 5. Death cross (with dynamic SMA)
                elif self.strategy.check_death_cross(df, current_idx, n1=n1, n2=n2)[0]:
                    sell_reason = SellReason.DEATH_CROSS.value
                
                # 6. End of day close (optional)
                elif self.strategy.close_at_eod and timestamp.hour == 15:
                    sell_reason = SellReason.END_OF_DAY.value
                
                if sell_reason:
                    trade = self.execute_sell(symbol, price, timestamp, sell_reason, market_condition)
                    if trade:
                        pnl = (price - position.avg_buy_price) * position.shares
                        pnl_pct = position.profit_loss_pct(price)
                        daily_trades += 1
                        daily_pnl += pnl
                        
                        if verbose:
                            emoji = "✅" if pnl > 0 else "❌"
                            print(f"  {emoji} SELL {symbol}: {position.shares:.0f}sh @ ${price:.2f} "
                                  f"[{sell_reason}] P/L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
            
            # Check for buy signals (only if we have capacity)
            if len(self.positions) < self.max_positions:
                # Calculate slopes for all symbols (matches order_symbols_by_slope)
                symbol_slopes = []
                for symbol in symbols:
                    if symbol in self.positions:
                        continue
                    
                    df = symbol_data.get(symbol)
                    if df is None:
                        continue
                    
                    ts_idx = df[df['datetime'] == timestamp].index
                    if len(ts_idx) == 0:
                        continue
                    current_idx = ts_idx[0]
                    
                    slope = self.strategy.calculate_slope(df, current_idx)
                    symbol_slopes.append((symbol, slope, current_idx))
                
                # Sort by slope descending (matches main.py)
                if self.strategy.use_slope_ordering:
                    symbol_slopes.sort(key=lambda x: x[1], reverse=True)
                
                for symbol, slope, current_idx in symbol_slopes:
                    if symbol in self.positions:
                        continue
                    
                    if symbol not in current_prices:
                        continue
                    
                    df = symbol_data.get(symbol)
                    if df is None:
                        continue
                    
                    price = current_prices[symbol]
                    
                    # NEW: Check all buy conditions matching main.py
                    
                    # 1. Price cap filter
                    if self.strategy.use_price_cap and price > self.strategy.price_cap_value:
                        rejected_buys['price_cap'] += 1
                        continue
                    
                    # 2. Slope filter (matches order_symbols_by_slope threshold)
                    if self.strategy.use_slope_ordering and slope <= self.strategy.slope_threshold:
                        rejected_buys['slope_filter'] += 1
                        continue
                    
                    # 3. Market uptrend required
                    if self.strategy.use_market_filter and not market_uptrend:
                        rejected_buys['market_downtrend'] += 1
                        continue
                    
                    # 4. No buying in major downtrend
                    if self.strategy.use_market_filter and market_major_downtrend:
                        rejected_buys['major_downtrend'] += 1
                        continue
                    
                    # 5. No buying after 1:30pm (EOD filter)
                    if self.strategy.use_eod_filter and self.strategy.is_eod(timestamp):
                        rejected_buys['eod_filter'] += 1
                        continue
                    
                    # 6. Day trade limit check
                    if self.day_trades_today >= self.day_trade_limit:
                        rejected_buys['day_trade_limit'] += 1
                        continue
                    
                    # 7. Check golden cross with dynamic SMA
                    is_golden_cross, cross_price, price_5hr_ago = self.strategy.check_golden_cross(
                        df, current_idx, n1=n1, n2=n2
                    )
                    
                    if not is_golden_cross:
                        continue
                    
                    # 8. Price must be > 5 hours ago (matches main.py cross[2] > cross[3])
                    if self.strategy.use_price_5hr_check:
                        if price_5hr_ago is not None and price <= price_5hr_ago:
                            rejected_buys['price_5hr'] += 1
                            continue
                    
                    # All conditions passed - execute buy
                    shares = self.calculate_position_size(price)
                    if shares > 0:
                        trade = self.execute_buy(symbol, price, shares, timestamp, "golden_cross", market_condition)
                        if trade:
                            daily_trades += 1
                            if verbose:
                                print(f"  🟢 BUY  {symbol}: {shares}sh @ ${price:.2f} "
                                      f"[golden_cross, slope={slope:.4f}] Cost: ${price * shares:.2f}")
        
        # Close remaining positions at end of backtest
        if self.positions and verbose:
            print(f"\nClosing {len(self.positions)} remaining positions at end of backtest...")
        
        final_prices = {}
        for symbol, df in symbol_data.items():
            if not df.empty:
                final_prices[symbol] = df.iloc[-1]['close']
        
        for symbol in list(self.positions.keys()):
            if symbol in final_prices:
                price = final_prices[symbol]
                position = self.positions[symbol]
                trade = self.execute_sell(symbol, price, all_timestamps[-1], SellReason.END_OF_BACKTEST.value)
                if trade and verbose:
                    pnl = (price - position.avg_buy_price) * position.shares
                    pnl_pct = position.profit_loss_pct(price)
                    print(f"  📤 CLOSE {symbol}: {position.shares:.0f}sh @ ${price:.2f} P/L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
        
        # Calculate final results
        results = self._calculate_results(symbols, all_timestamps[0], all_timestamps[-1])
        
        # Add rejected buy stats
        results['rejected_buys'] = rejected_buys
        results['features'] = {
            'market_filter': self.strategy.use_market_filter,
            'eod_filter': self.strategy.use_eod_filter,
            'profit_before_eod': self.strategy.use_profit_before_eod,
            'price_5hr_check': self.strategy.use_price_5hr_check,
            'dynamic_sma': self.strategy.use_dynamic_sma,
            'slope_ordering': self.strategy.use_slope_ordering,
            'price_cap': self.strategy.use_price_cap
        }
        
        if verbose:
            self._print_results(results)
        
        return results
    
    def _calculate_results(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Calculate backtest metrics"""
        total_trades = len(self.trades)
        buy_trades = [t for t in self.trades if t.trade_type == TradeType.BUY]
        sell_trades = [t for t in self.trades if t.trade_type == TradeType.SELL]
        
        # Match buys with sells to calculate P&L
        winning_trades = 0
        losing_trades = 0
        total_wins = 0.0
        total_losses = 0.0
        
        buy_prices = {}
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
        
        final_capital = self.cash
        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Calculate max drawdown from equity curve
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        peak = self.initial_capital
        
        for _, equity in self.equity_curve:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = (drawdown / peak) * 100
        
        # Calculate Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                prev_eq = self.equity_curve[i-1][1]
                curr_eq = self.equity_curve[i][1]
                returns.append((curr_eq - prev_eq) / prev_eq)
            
            if returns and np.std(returns) > 0:
                # Annualize: assume 7 hours/day, 252 days/year = 1764 hours
                sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(1764)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        win_rate = (winning_trades / (winning_trades + losing_trades) * 100) if (winning_trades + losing_trades) > 0 else 0
        avg_win = total_wins / winning_trades if winning_trades > 0 else 0
        avg_loss = total_losses / losing_trades if losing_trades > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Calculate trades per day
        trading_days = len(set(ts.date() for ts, _ in self.equity_curve))
        trades_per_day = len(sell_trades) / trading_days if trading_days > 0 else 0
        
        return {
            'start_date': str(start_date),
            'end_date': str(end_date),
            'initial_capital': self.initial_capital,
            'final_capital': round(final_capital, 2),
            'total_return': round(total_return, 2),
            'total_return_pct': round(total_return_pct, 2),
            'total_trades': total_trades,
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'inf',
            'max_drawdown': round(max_drawdown, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'sharpe_ratio': round(sharpe_ratio, 4),
            'trading_days': trading_days,
            'trades_per_day': round(trades_per_day, 2),
            'symbols_tested': symbols,
            'strategy': {
                'short_sma': self.strategy.short_sma,
                'long_sma': self.strategy.long_sma,
                'golden_cross_hours': self.strategy.golden_cross_hours,
                'stop_loss_pct': self.strategy.stop_loss_pct,
                'take_profit_pct': self.strategy.take_profit_pct
            }
        }
    
    def _print_results(self, results: Dict):
        """Print formatted results"""
        print(f"\n{'='*70}")
        print(f"BACKTEST RESULTS (ACCURATE MAIN.PY SIMULATION)")
        print(f"{'='*70}")
        
        print(f"\n📅 Period: {results['start_date'][:10]} to {results['end_date'][:10]}")
        print(f"📈 Symbols: {', '.join(results['symbols_tested'])}")
        
        print(f"\n💰 FINANCIAL SUMMARY")
        print(f"   Initial Capital:  $ {results['initial_capital']:>12,.2f}")
        print(f"   Final Capital:    $ {results['final_capital']:>12,.2f}")
        print(f"   Total Return:     $ {results['total_return']:>+12,.2f} ({results['total_return_pct']:+.2f}%)")
        
        print(f"\n📊 PERFORMANCE METRICS")
        print(f"   Max Drawdown:     $ {results['max_drawdown']:>12,.2f} ({results['max_drawdown_pct']:.2f}%)")
        print(f"   Sharpe Ratio:       {results['sharpe_ratio']:>12.4f}")
        
        print(f"\n🎯 TRADING STATISTICS")
        print(f"   Trading Days:       {results['trading_days']:>12}")
        print(f"   Total Trades:       {results['total_trades']:>12}")
        print(f"   Trades/Day:         {results['trades_per_day']:>12.2f}")
        print(f"   Winning Trades:     {results['winning_trades']:>12}")
        print(f"   Losing Trades:      {results['losing_trades']:>12}")
        print(f"   Win Rate:           {results['win_rate']:>11.2f}%")
        print(f"   Avg Win:          $ {results['avg_win']:>12,.2f}")
        print(f"   Avg Loss:         $ {results['avg_loss']:>12,.2f}")
        pf = results['profit_factor']
        pf_str = f"{pf:>12.2f}" if pf != 'inf' else "         inf"
        print(f"   Profit Factor:      {pf_str}")
        
        # NEW: Show rejected buy reasons
        if 'rejected_buys' in results:
            print(f"\n🚫 REJECTED BUY SIGNALS (main.py filters)")
            rejected = results['rejected_buys']
            total_rejected = sum(rejected.values())
            print(f"   Total Rejected:     {total_rejected:>12}")
            if rejected.get('market_downtrend', 0) > 0:
                print(f"   Market Downtrend:   {rejected['market_downtrend']:>12}")
            if rejected.get('major_downtrend', 0) > 0:
                print(f"   Major Downtrend:    {rejected['major_downtrend']:>12}")
            if rejected.get('eod_filter', 0) > 0:
                print(f"   After 1:30pm EOD:   {rejected['eod_filter']:>12}")
            if rejected.get('price_5hr', 0) > 0:
                print(f"   Price < 5hr Ago:    {rejected['price_5hr']:>12}")
            if rejected.get('slope_filter', 0) > 0:
                print(f"   Low Slope:          {rejected['slope_filter']:>12}")
            if rejected.get('price_cap', 0) > 0:
                print(f"   Over Price Cap:     {rejected['price_cap']:>12}")
            if rejected.get('day_trade_limit', 0) > 0:
                print(f"   Day Trade Limit:    {rejected['day_trade_limit']:>12}")
        
        print(f"\n⚙️  STRATEGY CONFIGURATION")
        strat = results['strategy']
        print(f"   Hourly SMA: {strat['short_sma']}/{strat['long_sma']}")
        print(f"   Golden Cross Lookback: {strat['golden_cross_hours']} hours")
        print(f"   Stop Loss: {strat['stop_loss_pct']}%")
        print(f"   Take Profit: {strat['take_profit_pct']}%")
        
        # NEW: Show enabled features
        if 'features' in results:
            print(f"\n🔧 MAIN.PY MATCHING FEATURES")
            features = results['features']
            for feature, enabled in features.items():
                status = "✅ ON" if enabled else "❌ OFF"
                print(f"   {feature}: {status}")
        
        print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Intraday Backtesting for RobinhoodBot Day Trading (Accurate main.py Simulation)'
    )
    parser.add_argument(
        '--symbols', type=str, default='AAPL,MSFT,GOOGL',
        help='Comma-separated list of symbols (default: AAPL,MSFT,GOOGL)'
    )
    parser.add_argument(
        '--days', type=int, default=30,
        help='Number of trading days to simulate (default: 30)'
    )
    parser.add_argument(
        '--capital', type=float, default=10000,
        help='Initial capital (default: 10000)'
    )
    parser.add_argument(
        '--short-sma', type=int, default=20,
        help='Short SMA period in hours (default: 20)'
    )
    parser.add_argument(
        '--long-sma', type=int, default=50,
        help='Long SMA period in hours (default: 50)'
    )
    parser.add_argument(
        '--take-profit', type=float, default=take_profit_percent,
        help=f'Take profit percentage (default: {take_profit_percent})'
    )
    parser.add_argument(
        '--stop-loss', type=float, default=stop_loss_percent,
        help=f'Stop loss percentage (default: {stop_loss_percent})'
    )
    parser.add_argument(
        '--golden-cross-hours', type=int, default=24,
        help='Hours to look back for golden cross (default: 24)'
    )
    parser.add_argument(
        '--max-positions', type=int, default=5,
        help='Maximum number of concurrent positions (default: 5)'
    )
    parser.add_argument(
        '--close-eod', action='store_true',
        help='Close all positions at end of each day'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress trade-by-trade output'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output file for results (JSON)'
    )
    
    # NEW: Main.py matching feature toggles
    parser.add_argument(
        '--no-market-filter', action='store_true',
        help='Disable market trend filtering (not recommended - less accurate)'
    )
    parser.add_argument(
        '--no-eod-filter', action='store_true',
        help='Disable no-buy-after-1:30pm filter (not recommended - less accurate)'
    )
    parser.add_argument(
        '--no-profit-before-eod', action='store_true',
        help='Disable profit_before_eod sell trigger (not recommended - less accurate)'
    )
    parser.add_argument(
        '--no-price-5hr-check', action='store_true',
        help='Disable price > 5hr ago check (not recommended - less accurate)'
    )
    parser.add_argument(
        '--no-dynamic-sma', action='store_true',
        help='Disable dynamic SMA adjustment (not recommended - less accurate)'
    )
    parser.add_argument(
        '--no-slope-ordering', action='store_true',
        help='Disable slope-based symbol ordering (not recommended - less accurate)'
    )
    parser.add_argument(
        '--no-price-cap', action='store_true',
        help='Disable price cap filter'
    )
    parser.add_argument(
        '--price-cap', type=float, default=price_cap,
        help=f'Maximum stock price to consider (default: {price_cap})'
    )
    parser.add_argument(
        '--simple-mode', action='store_true',
        help='Disable ALL main.py filters (runs like old simplified backtest)'
    )
    
    args = parser.parse_args()
    
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    # Determine feature toggles
    if args.simple_mode:
        use_market_filter = False
        use_eod_filter = False
        use_profit_before_eod = False
        use_price_5hr_check = False
        use_dynamic_sma = False
        use_slope_ordering = False
        use_price_cap_filter = False
    else:
        use_market_filter = not args.no_market_filter
        use_eod_filter = not args.no_eod_filter
        use_profit_before_eod = not args.no_profit_before_eod
        use_price_5hr_check = not args.no_price_5hr_check
        use_dynamic_sma = not args.no_dynamic_sma
        use_slope_ordering = not args.no_slope_ordering
        use_price_cap_filter = not args.no_price_cap
    
    strategy = IntradayTradingStrategy(
        short_sma=args.short_sma,
        long_sma=args.long_sma,
        golden_cross_hours=args.golden_cross_hours,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        use_stop_loss=True,
        close_at_eod=args.close_eod,
        # Main.py matching features
        use_market_filter=use_market_filter,
        use_eod_filter=use_eod_filter,
        use_profit_before_eod=use_profit_before_eod,
        use_price_5hr_check=use_price_5hr_check,
        use_dynamic_sma=use_dynamic_sma,
        use_slope_ordering=use_slope_ordering,
        use_price_cap=use_price_cap_filter,
        price_cap_value=args.price_cap
    )
    
    backtester = IntradayBacktester(
        initial_capital=args.capital,
        max_positions=args.max_positions,
        strategy=strategy
    )
    
    results = backtester.run(
        symbols=symbols,
        days=args.days,
        seed=args.seed,
        verbose=not args.quiet
    )
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return results


if __name__ == '__main__':
    main()
