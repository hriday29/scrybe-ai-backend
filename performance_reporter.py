# performance_reporter.py
import pandas as pd
import numpy as np
import database_manager
from logger_config import log
import argparse
import matplotlib.pyplot as plt
import config # Import the whole config file

# --- ANSI color codes for immersive terminal output ---
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'

def generate_performance_report(batch_id: str):
    """
    Generates a full backtest report for a specific batch_id, including an equity curve chart,
    using a realistic fixed position sizing model.
    """
    log.info(f"--- Generating Performance Report for Batch: '{batch_id}' ---")
    
    database_manager.init_db(purpose='scheduler')
    closed_trades = list(database_manager.performance_collection.find({"batch_id": batch_id}))
    
    if not closed_trades:
        log.warning(f"No closed trades found for batch '{batch_id}'. Cannot generate a report.")
        return

    df = pd.DataFrame(closed_trades)
    df['open_date'] = pd.to_datetime(df['open_date'])
    df['close_date'] = pd.to_datetime(df['close_date'])

    # --- NEW: Portfolio Simulation with Fixed Position Sizing ---
    initial_capital = config.BACKTEST_PORTFOLIO_CONFIG['initial_capital']
    position_size_pct = config.BACKTEST_PORTFOLIO_CONFIG['position_size_pct_of_capital']
    position_size = initial_capital * (position_size_pct / 100.0)
    
    df_sorted = df.sort_values(by='close_date').reset_index(drop=True)
    
    equity = initial_capital
    equity_curve = [initial_capital]
    
    for index, trade in df_sorted.iterrows():
        pnl_amount = position_size * (trade['net_return_pct'] / 100.0)
        equity += pnl_amount
        equity_curve.append(equity)

    # Add the realistic equity curve to the dataframe
    # We drop the first value because it's the starting capital for plotting purposes
    df_sorted['equity_curve'] = equity_curve[1:]
    
    # --- END OF NEW SIMULATION LOGIC ---

    # --- Recalculate metrics based on the new, realistic simulation ---
    total_trades = len(df_sorted)
    win_rate = (df_sorted['net_return_pct'] > 0).mean() * 100
    
    # Profit Factor is based on the sum of winning trades vs losing trades, scaled by win/loss rate
    gross_profit_sum = df_sorted[df_sorted['net_return_pct'] > 0]['net_return_pct'].sum()
    gross_loss_sum = abs(df_sorted[df_sorted['net_return_pct'] < 0]['net_return_pct'].sum())
    
    profit_factor = gross_profit_sum / gross_loss_sum if gross_loss_sum > 0 else float('inf')

    avg_gain = df_sorted[df_sorted['net_return_pct'] > 0]['net_return_pct'].mean() if not df_sorted[df_sorted['net_return_pct'] > 0].empty else 0
    avg_loss = df_sorted[df_sorted['net_return_pct'] < 0]['net_return_pct'].mean() if not df_sorted[df_sorted['net_return_pct'] < 0].empty else 0

    # Max Drawdown calculation now uses the realistic equity curve
    peak = df_sorted['equity_curve'].expanding().max()
    drawdown = (peak - df_sorted['equity_curve']) / peak
    max_drawdown = drawdown.max() * 100

    # Generate and save the equity curve chart
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_sorted['close_date'], df_sorted['equity_curve'], label='Strategy Equity', color='#3b82f6', linewidth=2)
    ax.fill_between(df_sorted['close_date'], df_sorted['equity_curve'], initial_capital, where=df_sorted['equity_curve'] > initial_capital, interpolate=True, alpha=0.2, color='green')
    ax.fill_between(df_sorted['close_date'], df_sorted['equity_curve'], initial_capital, where=df_sorted['equity_curve'] < initial_capital, interpolate=True, alpha=0.2, color='red')
    ax.axhline(y=initial_capital, color='white', linestyle='--', linewidth=1, label='Initial Capital')
    ax.set_title(f"Equity Curve for Batch: {batch_id}", fontsize=16, color='white')
    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Portfolio Value (INR)", color='white')
    ax.grid(True, color='#333A4C', linestyle='--')
    ax.legend()
    chart_filename = f"report_{batch_id}_equity_curve.png"
    plt.savefig(chart_filename)
    log.info(f"Equity curve chart saved to: {Colors.YELLOW}{chart_filename}{Colors.END}")

    # --- Print the Immersive Report ---
    print("\n" + "="*60)
    print(f"{Colors.CYAN}### Overall Performance Summary: '{batch_id}' ###{Colors.END}")
    print("="*60)
    print(f"{'Total Trades:':<30} {Colors.YELLOW}{total_trades}{Colors.END}")
    print(f"{'Win Rate:':<30} {Colors.GREEN if win_rate > 50 else Colors.RED}{win_rate:.2f}%{Colors.END}")
    print(f"{'Profit Factor:':<30} {Colors.GREEN if profit_factor > 1 else Colors.RED}{profit_factor:.2f}{Colors.END}")
    print(f"{'Maximum Drawdown:':<30} {Colors.RED}{max_drawdown:.2f}%{Colors.END}")
    print(f"{'Average Gain / Winning Trade:':<30} {Colors.GREEN}+{avg_gain:.2f}%{Colors.END}")
    print(f"{'Average Loss / Losing Trade:':<30} {Colors.RED}{avg_loss:.2f}%{Colors.END}")
    print(f"{'Final Portfolio Value:':<30} {Colors.YELLOW}₹{df_sorted['equity_curve'].iloc[-1]:,.2f}{Colors.END}")
    
    # --- Per-Ticker Breakdown ---
    per_ticker_stats = df.groupby('ticker').agg(
        total_trades=('ticker', 'count'),
        net_return=('net_return_pct', 'sum'),
        win_rate=('net_return_pct', lambda x: (x > 0).mean() * 100)
    ).round(2).sort_values(by='net_return', ascending=False)
    
    print("\n" + "="*62)
    print(f"{Colors.CYAN}### Per-Ticker Performance Breakdown ###{Colors.END}")
    print("="*62)
    print(per_ticker_stats.to_string())
    print("="*62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a performance report for a specific backtest batch.")
    parser.add_argument('--batch_id', required=True, help='The unique ID of the backtest batch to analyze.')
    args = parser.parse_args()
    
    generate_performance_report(batch_id=args.batch_id)
    database_manager.close_db_connection()