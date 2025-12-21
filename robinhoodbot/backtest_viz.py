"""
Visualization module for backtest results

Generates charts and plots for analyzing backtest performance
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


def plot_equity_curve(equity_file='robinhoodbot/backtest_equity.csv', 
                      output_file='robinhoodbot/equity_curve.png',
                      show_plot=False):
    """
    Plot the equity curve from backtest results
    
    Args:
        equity_file: Path to equity curve CSV
        output_file: Path to save plot
        show_plot: Whether to display the plot
    """
    try:
        df = pd.read_csv(equity_file)
        df['date'] = pd.to_datetime(df['date'])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Portfolio Value
        ax1.plot(df['date'], df['value'], label='Portfolio Value', linewidth=2, color='#2E86AB')
        ax1.axhline(y=df['value'].iloc[0], color='gray', linestyle='--', alpha=0.5, label='Initial Value')
        ax1.fill_between(df['date'], df['value'], df['value'].iloc[0], 
                         where=(df['value'] >= df['value'].iloc[0]), 
                         interpolate=True, alpha=0.3, color='green', label='Profit')
        ax1.fill_between(df['date'], df['value'], df['value'].iloc[0], 
                         where=(df['value'] < df['value'].iloc[0]), 
                         interpolate=True, alpha=0.3, color='red', label='Loss')
        
        ax1.set_title('Portfolio Value Over Time', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Cash vs Invested
        ax2.plot(df['date'], df['cash'], label='Cash', linewidth=2, color='#A23B72')
        ax2.plot(df['date'], df['positions_value'], label='Positions Value', linewidth=2, color='#F18F01')
        ax2.fill_between(df['date'], 0, df['cash'], alpha=0.3, color='#A23B72')
        ax2.fill_between(df['date'], 0, df['positions_value'], alpha=0.3, color='#F18F01')
        
        ax2.set_title('Cash vs Invested Capital', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Value ($)', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Equity curve saved to {output_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        print(f"Error plotting equity curve: {e}")


def plot_trade_analysis(trades_file='robinhoodbot/backtest_trades.csv',
                        output_file='robinhoodbot/trade_analysis.png',
                        show_plot=False):
    """
    Plot trade analysis charts
    
    Args:
        trades_file: Path to trades CSV
        output_file: Path to save plot
        show_plot: Whether to display the plot
    """
    try:
        df = pd.read_csv(trades_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter sell trades only (they have profit data)
        sells = df[df['type'] == 'SELL'].copy()
        
        if len(sells) == 0:
            print("No sell trades to analyze")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Profit/Loss per trade
        colors = ['green' if p > 0 else 'red' for p in sells['profit']]
        ax1.bar(range(len(sells)), sells['profit'], color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_title('Profit/Loss Per Trade', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Trade Number', fontsize=12)
        ax1.set_ylabel('Profit/Loss ($)', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Cumulative profit
        sells['cumulative_profit'] = sells['profit'].cumsum()
        ax2.plot(sells['date'], sells['cumulative_profit'], linewidth=2, color='#2E86AB')
        ax2.fill_between(sells['date'], 0, sells['cumulative_profit'], alpha=0.3, color='#2E86AB')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax2.set_title('Cumulative Profit', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Cumulative Profit ($)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 3: Win/Loss distribution
        wins = len(sells[sells['profit'] > 0])
        losses = len(sells[sells['profit'] <= 0])
        ax3.pie([wins, losses], labels=['Wins', 'Losses'], autopct='%1.1f%%',
                colors=['#06D6A0', '#EF476F'], startangle=90)
        ax3.set_title(f'Win/Loss Distribution ({wins} wins, {losses} losses)', 
                     fontsize=14, fontweight='bold')
        
        # Plot 4: Sell reasons
        if 'sell_reason' in sells.columns:
            reason_counts = sells['sell_reason'].value_counts()
            ax4.bar(range(len(reason_counts)), reason_counts.values, 
                   color='#F18F01', alpha=0.7)
            ax4.set_xticks(range(len(reason_counts)))
            ax4.set_xticklabels(reason_counts.index, rotation=45, ha='right')
            ax4.set_title('Sell Reasons Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Reason', fontsize=12)
            ax4.set_ylabel('Number of Trades', fontsize=12)
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Trade analysis saved to {output_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        print(f"Error plotting trade analysis: {e}")


def plot_returns_distribution(trades_file='robinhoodbot/backtest_trades.csv',
                              output_file='robinhoodbot/returns_distribution.png',
                              show_plot=False):
    """
    Plot returns distribution histogram
    
    Args:
        trades_file: Path to trades CSV
        output_file: Path to save plot
        show_plot: Whether to display the plot
    """
    try:
        df = pd.read_csv(trades_file)
        sells = df[df['type'] == 'SELL'].copy()
        
        if len(sells) == 0:
            print("No sell trades to analyze")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Dollar returns histogram
        ax1.hist(sells['profit'], bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break Even')
        ax1.axvline(x=sells['profit'].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: ${sells["profit"].mean():.2f}')
        ax1.set_title('Distribution of Returns ($)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Profit/Loss ($)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Percentage returns histogram
        ax2.hist(sells['profit_pct'], bins=30, color='#F18F01', alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break Even')
        ax2.axvline(x=sells['profit_pct'].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {sells["profit_pct"].mean():.2f}%')
        ax2.set_title('Distribution of Returns (%)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Return (%)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Returns distribution saved to {output_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        print(f"Error plotting returns distribution: {e}")


def generate_all_plots(equity_file='robinhoodbot/backtest_equity.csv',
                      trades_file='robinhoodbot/backtest_trades.csv',
                      show_plots=False):
    """
    Generate all visualization plots
    
    Args:
        equity_file: Path to equity curve CSV
        trades_file: Path to trades CSV
        show_plots: Whether to display plots
    """
    print("\nGenerating visualization plots...")
    plot_equity_curve(equity_file, show_plot=show_plots)
    plot_trade_analysis(trades_file, show_plot=show_plots)
    plot_returns_distribution(trades_file, show_plot=show_plots)
    print("All plots generated successfully!")


if __name__ == "__main__":
    """Generate plots from existing backtest results"""
    generate_all_plots(show_plots=True)
