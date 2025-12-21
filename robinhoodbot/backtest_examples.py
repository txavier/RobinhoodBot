"""
Example: How to use the RobinhoodBot Backtesting System

This script demonstrates various ways to run backtests
"""

from run_backtest import run_simple_backtest, run_custom_backtest, StrategyBacktest
from backtest_viz import generate_all_plots
from datetime import datetime, timedelta


def example_1_simple_backtest():
    """Example 1: Run a simple backtest with default settings"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Simple Backtest")
    print("=" * 80)
    print("Testing 5 tech stocks over the past year with $10,000")
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    metrics = run_simple_backtest(symbols=symbols, initial_cash=10000)
    
    return metrics


def example_2_custom_date_range():
    """Example 2: Test a specific time period"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Custom Date Range")
    print("=" * 80)
    print("Testing 2024 performance")
    
    symbols = ['AAPL', 'MSFT', 'TSLA', 'AMD', 'NFLX']
    metrics = run_custom_backtest(
        symbols=symbols,
        start_date='2024-01-01',
        end_date='2024-12-31',
        initial_cash=15000
    )
    
    return metrics


def example_3_parameter_tuning():
    """Example 3: Test with custom strategy parameters"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Custom Strategy Parameters")
    print("=" * 80)
    print("Testing with aggressive parameters (shorter SMAs, higher profit target)")
    
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Calculate dates
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    # Create custom backtest
    backtest = StrategyBacktest(symbols, start_date, end_date, initial_cash=20000)
    
    # Customize parameters
    backtest.n1_buy = 10  # Faster short-term SMA
    backtest.n2_buy = 30  # Faster long-term SMA
    backtest.n1_sell = 10
    backtest.n2_sell = 30
    backtest.take_profit_threshold = 3.5  # Higher profit target (3.5%)
    
    metrics = backtest.run()
    
    return metrics


def example_4_sector_comparison():
    """Example 4: Compare different sectors"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Sector Comparison")
    print("=" * 80)
    
    # Define sector symbols
    sectors = {
        'Tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV'],
        'Finance': ['JPM', 'BAC', 'WFC', 'GS'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB']
    }
    
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    initial_cash = 10000
    
    results = {}
    
    for sector, symbols in sectors.items():
        print(f"\nTesting {sector} sector...")
        metrics = run_custom_backtest(symbols, start_date, end_date, initial_cash)
        
        if metrics:
            results[sector] = {
                'return': metrics['total_return_pct'],
                'sharpe': metrics['sharpe_ratio'],
                'win_rate': metrics['win_rate'],
                'max_drawdown': metrics['max_drawdown_pct']
            }
    
    # Print comparison
    print("\n" + "=" * 80)
    print("SECTOR COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Sector':<12} {'Return %':<12} {'Sharpe':<10} {'Win Rate %':<12} {'Max DD %':<10}")
    print("-" * 80)
    
    for sector, data in results.items():
        print(f"{sector:<12} {data['return']:>10.2f}% {data['sharpe']:>8.2f} "
              f"{data['win_rate']:>10.2f}% {data['max_drawdown']:>8.2f}%")
    
    return results


def example_5_walk_forward():
    """Example 5: Walk-forward testing (simplified)"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Walk-Forward Testing")
    print("=" * 80)
    print("Testing strategy across multiple 3-month periods")
    
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Define periods
    periods = [
        ('2024-01-01', '2024-03-31', 'Q1 2024'),
        ('2024-04-01', '2024-06-30', 'Q2 2024'),
        ('2024-07-01', '2024-09-30', 'Q3 2024'),
        ('2024-10-01', '2024-12-31', 'Q4 2024'),
    ]
    
    results = []
    
    for start, end, label in periods:
        print(f"\nTesting {label}...")
        metrics = run_custom_backtest(symbols, start, end, initial_cash=10000)
        
        if metrics:
            results.append({
                'period': label,
                'return': metrics['total_return_pct'],
                'trades': metrics['sell_trades'],
                'win_rate': metrics['win_rate']
            })
    
    # Print results
    print("\n" + "=" * 80)
    print("QUARTERLY PERFORMANCE")
    print("=" * 80)
    print(f"{'Period':<12} {'Return %':<12} {'Trades':<10} {'Win Rate %':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['period']:<12} {result['return']:>10.2f}% {result['trades']:>8} "
              f"{result['win_rate']:>10.2f}%")
    
    return results


def example_6_visualization():
    """Example 6: Generate visualization plots"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Visualization")
    print("=" * 80)
    print("Generating charts from the last backtest...")
    
    # First run a backtest
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    print("\nRunning backtest...")
    run_simple_backtest(symbols=symbols, initial_cash=10000)
    
    # Then generate plots
    print("\nGenerating plots...")
    generate_all_plots(show_plots=False)  # Set to True to display plots
    
    print("\nPlots saved to:")
    print("  - robinhoodbot/equity_curve.png")
    print("  - robinhoodbot/trade_analysis.png")
    print("  - robinhoodbot/returns_distribution.png")


if __name__ == "__main__":
    """Run all examples"""
    
    print("\n" + "=" * 80)
    print("ROBINHOODBOT BACKTESTING EXAMPLES")
    print("=" * 80)
    print("\nThis script demonstrates various backtesting capabilities.")
    print("Choose an example to run:")
    print("\n1. Simple Backtest (5 tech stocks, 1 year)")
    print("2. Custom Date Range (2024 only)")
    print("3. Parameter Tuning (aggressive settings)")
    print("4. Sector Comparison (Tech vs Healthcare vs Finance vs Energy)")
    print("5. Walk-Forward Testing (quarterly periods)")
    print("6. Visualization (generate charts)")
    print("7. Run All Examples")
    print("\n" + "=" * 80)
    
    choice = input("\nEnter your choice (1-7): ").strip()
    
    if choice == '1':
        example_1_simple_backtest()
    elif choice == '2':
        example_2_custom_date_range()
    elif choice == '3':
        example_3_parameter_tuning()
    elif choice == '4':
        example_4_sector_comparison()
    elif choice == '5':
        example_5_walk_forward()
    elif choice == '6':
        example_6_visualization()
    elif choice == '7':
        print("\nRunning all examples...")
        example_1_simple_backtest()
        example_2_custom_date_range()
        example_3_parameter_tuning()
        example_4_sector_comparison()
        example_5_walk_forward()
        example_6_visualization()
    else:
        print("Invalid choice. Running Example 1...")
        example_1_simple_backtest()
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
