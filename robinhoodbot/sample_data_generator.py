#!/usr/bin/env python3
"""
Sample Data Generator for Backtesting

Generates synthetic stock data for testing the backtest module
without requiring a Robinhood login.

Usage:
    python sample_data_generator.py --symbol AAPL --days 365
    python sample_data_generator.py --symbols AAPL,MSFT,GOOGL --days 180 --output sample_data/
"""

import argparse
import json
import os
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
import pandas as pd


def generate_brownian_motion(
    initial_price: float,
    days: int,
    volatility: float = 0.02,
    drift: float = 0.0001
) -> List[float]:
    """
    Generate price series using geometric Brownian motion.
    
    Args:
        initial_price: Starting price
        days: Number of days to generate
        volatility: Daily volatility (std dev of returns)
        drift: Daily drift (expected return)
    
    Returns:
        List of prices
    """
    prices = [initial_price]
    
    for _ in range(days - 1):
        # Random shock
        shock = np.random.normal(drift, volatility)
        # New price = old price * (1 + shock)
        new_price = prices[-1] * (1 + shock)
        prices.append(max(0.01, new_price))  # Ensure positive price
    
    return prices


def generate_ohlcv_data(
    symbol: str,
    start_date: datetime,
    days: int,
    initial_price: float = 100.0,
    volatility: float = 0.02,
    drift: float = 0.0001,
    avg_volume: int = 1000000
) -> pd.DataFrame:
    """
    Generate realistic OHLCV (Open, High, Low, Close, Volume) data.
    
    Args:
        symbol: Stock symbol
        start_date: Starting date
        days: Number of trading days
        initial_price: Starting price
        volatility: Daily price volatility
        drift: Expected daily return
        avg_volume: Average daily volume
    
    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    # Generate close prices using Brownian motion
    closes = generate_brownian_motion(initial_price, days, volatility, drift)
    
    data = []
    current_date = start_date
    
    for i, close in enumerate(closes):
        # Skip weekends
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        
        # Generate OHLC with some randomness
        daily_range = close * volatility * 1.5
        
        # Open is close of previous day with small gap
        if i > 0:
            gap = random.uniform(-daily_range * 0.3, daily_range * 0.3)
            open_price = closes[i-1] + gap
        else:
            open_price = close * random.uniform(0.99, 1.01)
        
        # High and low based on daily range
        high = max(open_price, close) + random.uniform(0, daily_range * 0.5)
        low = min(open_price, close) - random.uniform(0, daily_range * 0.5)
        
        # Ensure OHLC consistency
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        low = max(0.01, low)
        
        # Volume with some randomness
        volume = int(avg_volume * random.uniform(0.5, 2.0))
        
        data.append({
            'date': current_date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(data)


def generate_trending_data(
    symbol: str,
    start_date: datetime,
    days: int,
    initial_price: float = 100.0,
    trend: str = 'bullish',  # 'bullish', 'bearish', 'sideways', 'volatile'
    avg_volume: int = 1000000
) -> pd.DataFrame:
    """
    Generate data with specific trend characteristics.
    
    Args:
        symbol: Stock symbol
        start_date: Starting date
        days: Number of trading days
        initial_price: Starting price
        trend: Type of trend
        avg_volume: Average daily volume
    
    Returns:
        DataFrame with OHLCV data
    """
    if trend == 'bullish':
        drift = 0.001  # 0.1% daily expected return
        volatility = 0.015
    elif trend == 'bearish':
        drift = -0.001
        volatility = 0.02
    elif trend == 'volatile':
        drift = 0.0
        volatility = 0.04
    else:  # sideways
        drift = 0.0
        volatility = 0.01
    
    return generate_ohlcv_data(
        symbol, start_date, days, initial_price, volatility, drift, avg_volume
    )


def generate_golden_cross_scenario(
    symbol: str,
    start_date: datetime,
    initial_price: float = 100.0,
    avg_volume: int = 1000000
) -> pd.DataFrame:
    """
    Generate data that includes a golden cross pattern.
    
    The data will have:
    - Initial downtrend (50 days)
    - Consolidation (30 days)
    - Uptrend leading to golden cross (70 days)
    - Continued uptrend (50 days)
    
    Returns:
        DataFrame with OHLCV data
    """
    data_frames = []
    current_date = start_date
    price = initial_price
    
    # Phase 1: Downtrend
    df1 = generate_trending_data(
        symbol, current_date, 50, price,
        trend='bearish', avg_volume=avg_volume
    )
    data_frames.append(df1)
    price = df1.iloc[-1]['close']
    current_date = df1.iloc[-1]['date'] + timedelta(days=1)
    
    # Phase 2: Consolidation
    df2 = generate_trending_data(
        symbol, current_date, 30, price,
        trend='sideways', avg_volume=avg_volume
    )
    data_frames.append(df2)
    price = df2.iloc[-1]['close']
    current_date = df2.iloc[-1]['date'] + timedelta(days=1)
    
    # Phase 3: Strong uptrend (leads to golden cross)
    df3 = generate_trending_data(
        symbol, current_date, 70, price,
        trend='bullish', avg_volume=avg_volume
    )
    data_frames.append(df3)
    price = df3.iloc[-1]['close']
    current_date = df3.iloc[-1]['date'] + timedelta(days=1)
    
    # Phase 4: Continued uptrend
    df4 = generate_trending_data(
        symbol, current_date, 50, price,
        trend='bullish', avg_volume=avg_volume
    )
    data_frames.append(df4)
    
    result = pd.concat(data_frames, ignore_index=True)
    return result


def generate_death_cross_scenario(
    symbol: str,
    start_date: datetime,
    initial_price: float = 100.0,
    avg_volume: int = 1000000
) -> pd.DataFrame:
    """
    Generate data that includes a death cross pattern.
    """
    data_frames = []
    current_date = start_date
    price = initial_price
    
    # Phase 1: Uptrend
    df1 = generate_trending_data(
        symbol, current_date, 60, price,
        trend='bullish', avg_volume=avg_volume
    )
    data_frames.append(df1)
    price = df1.iloc[-1]['close']
    current_date = df1.iloc[-1]['date'] + timedelta(days=1)
    
    # Phase 2: Topping
    df2 = generate_trending_data(
        symbol, current_date, 30, price,
        trend='sideways', avg_volume=avg_volume
    )
    data_frames.append(df2)
    price = df2.iloc[-1]['close']
    current_date = df2.iloc[-1]['date'] + timedelta(days=1)
    
    # Phase 3: Strong downtrend (leads to death cross)
    df3 = generate_trending_data(
        symbol, current_date, 70, price,
        trend='bearish', avg_volume=avg_volume
    )
    data_frames.append(df3)
    price = df3.iloc[-1]['close']
    current_date = df3.iloc[-1]['date'] + timedelta(days=1)
    
    # Phase 4: Continued downtrend
    df4 = generate_trending_data(
        symbol, current_date, 40, price,
        trend='bearish', avg_volume=avg_volume
    )
    data_frames.append(df4)
    
    result = pd.concat(data_frames, ignore_index=True)
    return result


def save_data(df: pd.DataFrame, symbol: str, output_dir: str):
    """Save generated data to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON (for backtest module)
    df_copy = df.copy()
    df_copy['date'] = df_copy['date'].astype(str)
    json_path = os.path.join(output_dir, f"{symbol}_historical.json")
    df_copy.to_json(json_path, orient='records', indent=2)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, f"{symbol}_historical.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"Saved {symbol} data to {json_path} and {csv_path}")


def generate_multiple_symbols(
    symbols: List[str],
    start_date: datetime,
    days: int,
    output_dir: str,
    scenario: str = 'random'
) -> Dict[str, pd.DataFrame]:
    """
    Generate data for multiple symbols.
    
    Args:
        symbols: List of symbols
        start_date: Starting date
        days: Number of days
        output_dir: Directory to save data
        scenario: 'random', 'golden_cross', 'death_cross', 'mixed'
    
    Returns:
        Dictionary of symbol -> DataFrame
    """
    all_data = {}
    
    for i, symbol in enumerate(symbols):
        print(f"Generating data for {symbol}...")
        
        initial_price = random.uniform(20, 500)
        
        if scenario == 'golden_cross':
            df = generate_golden_cross_scenario(symbol, start_date, initial_price)
        elif scenario == 'death_cross':
            df = generate_death_cross_scenario(symbol, start_date, initial_price)
        elif scenario == 'mixed':
            # Alternate between scenarios
            if i % 3 == 0:
                df = generate_golden_cross_scenario(symbol, start_date, initial_price)
            elif i % 3 == 1:
                df = generate_death_cross_scenario(symbol, start_date, initial_price)
            else:
                trend = random.choice(['bullish', 'bearish', 'sideways', 'volatile'])
                df = generate_trending_data(symbol, start_date, days, initial_price, trend)
        else:  # random
            trend = random.choice(['bullish', 'bearish', 'sideways', 'volatile'])
            df = generate_trending_data(symbol, start_date, days, initial_price, trend)
        
        all_data[symbol] = df
        save_data(df, symbol, output_dir)
    
    return all_data


def main():
    parser = argparse.ArgumentParser(
        description='Generate sample stock data for backtesting'
    )
    
    parser.add_argument(
        '--symbols', '-s',
        type=str,
        default='AAPL,MSFT,GOOGL,AMZN,NVDA',
        help='Comma-separated list of symbols'
    )
    
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=365,
        help='Number of trading days (default: 365)'
    )
    
    parser.add_argument(
        '--start', '-st',
        type=str,
        help='Start date (YYYY-MM-DD). Default: days ago from today'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='sample_data',
        help='Output directory (default: sample_data)'
    )
    
    parser.add_argument(
        '--scenario', '-sc',
        type=str,
        choices=['random', 'golden_cross', 'death_cross', 'mixed'],
        default='mixed',
        help='Data scenario type (default: mixed)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    if args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
    else:
        start_date = datetime.now() - timedelta(days=args.days)
    
    print(f"\n{'='*50}")
    print("SAMPLE DATA GENERATOR")
    print(f"{'='*50}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Start Date: {start_date.date()}")
    print(f"Days: {args.days}")
    print(f"Scenario: {args.scenario}")
    print(f"Output: {args.output}")
    print(f"{'='*50}\n")
    
    all_data = generate_multiple_symbols(
        symbols, start_date, args.days, args.output, args.scenario
    )
    
    print(f"\n✅ Generated data for {len(all_data)} symbols")
    print(f"📁 Data saved to: {args.output}/")


if __name__ == '__main__':
    main()
