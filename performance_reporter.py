# performance_reporter.py
import pandas as pd
import os
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

# PASTE THIS ENTIRE FUNCTION INTO performance_reporter.py

def generate_performance_report(batch_id: str):
    """
    Generates a full backtest report for a specific batch_id, including an equity curve chart,
    a losing trade breakdown, and a signal funnel analysis.
    """
    log.info(f"--- Generating Performance Report for Batch: '{batch_id}' ---")
    
    database_manager.init_db(purpose='scheduler')
    
    # --- START: EXPANDED DATA FETCH FOR FAILURE ANALYSIS ---
    # Fetch closed trades for performance metrics
    closed_trades = list(database_manager.performance_collection.find({"batch_id": batch_id}))
    
    # Fetch ALL predictions for the AI signal funnel analysis
    all_predictions = list(database_manager.predictions_collection.find({"batch_id": batch_id}))
    # --- END: EXPANDED DATA FETCH ---

    if not closed_trades:
        log.warning(f"No closed trades found for batch '{batch_id}'. Cannot generate a full performance report.")
        # We can still try to generate the signal analysis even if no trades were closed
        if not all_predictions:
            return
    
    df = pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame()
    all_predictions_df = pd.DataFrame(all_predictions)

    # --- Overall Performance Summary (Your existing logic is perfect) ---
    if not df.empty:
        df['open_date'] = pd.to_datetime(df['open_date'])
        df['close_date'] = pd.to_datetime(df['close_date'])

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

        df_sorted['equity_curve'] = equity_curve[1:]
        
        total_trades = len(df_sorted)
        win_rate = (df_sorted['net_return_pct'] > 0).mean() * 100 if total_trades > 0 else 0
        
        gross_profit_sum = df_sorted[df_sorted['net_return_pct'] > 0]['net_return_pct'].sum()
        gross_loss_sum = abs(df_sorted[df_sorted['net_return_pct'] < 0]['net_return_pct'].sum())
        profit_factor = gross_profit_sum / gross_loss_sum if gross_loss_sum > 0 else float('inf')
        avg_gain = df_sorted[df_sorted['net_return_pct'] > 0]['net_return_pct'].mean() if not df_sorted[df_sorted['net_return_pct'] > 0].empty else 0
        avg_loss = df_sorted[df_sorted['net_return_pct'] < 0]['net_return_pct'].mean() if not df_sorted[df_sorted['net_return_pct'] < 0].empty else 0
        peak = df_sorted['equity_curve'].expanding().max()
        drawdown = (peak - df_sorted['equity_curve']) / peak if peak.all() > 0 else pd.Series(0, index=peak.index)
        max_drawdown = drawdown.max() * 100
        final_equity = df_sorted['equity_curve'].iloc[-1] if not df_sorted.empty else initial_capital

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
        reports_dir = 'reports'
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        chart_filename = os.path.join(reports_dir, f"report_{batch_id}_equity_curve.png")
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
        print(f"{'Final Portfolio Value:':<30} {Colors.YELLOW}₹{final_equity:,.2f}{Colors.END}")
        
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

    # --- START: NEW SECTION - LOSING TRADES BREAKDOWN ---
    if not df.empty:
        losing_trades_df = df[df['net_return_pct'] < 0].sort_values(by='net_return_pct')
        if not losing_trades_df.empty:
            print("\n" + "="*80)
            print(f"{Colors.RED}### Deep Dive: Losing Trades Analysis ###{Colors.END}")
            print("="*80)
            print(losing_trades_df[['ticker', 'strategy', 'open_date', 'close_date', 'net_return_pct', 'closing_reason']].to_string(index=False))
            print("="*80)
    # --- END: LOSING TRADES BREAKDOWN ---

    # --- START: NEW SECTION - AI & RISK FILTER ANALYSIS ---
    if not all_predictions_df.empty:
        total_analyses = len(all_predictions_df)
        # Extract the original signal from the verdict if it was vetoed
        def get_original_signal(row):
            verdict = row.get('analystVerdict', '')
            if 'vetoed' in verdict:
                if "'BUY'" in verdict: return 'BUY'
                if "'SELL'" in verdict: return 'SELL'
            return row.get('signal')

        all_predictions_df['original_signal'] = all_predictions_df.apply(get_original_signal, axis=1)
        
        initial_signals = all_predictions_df[all_predictions_df['original_signal'].isin(['BUY', 'SELL'])]
        
        vetoed_regime = all_predictions_df[all_predictions_df['reason_code'] == "REGIME_VETO"]
        vetoed_conviction = all_predictions_df[all_predictions_df['reason_code'] == "LOW_CONVICTION"]
        vetoed_quality = all_predictions_df[all_predictions_df['reason_code'] == "QUALITY_VETO"]

        final_approved_signals = all_predictions_df[all_predictions_df['signal'].isin(['BUY', 'SELL'])]

        print("\n" + "="*60)
        print(f"{Colors.CYAN}### Signal Funnel & Risk Manager Analysis ###{Colors.END}")
        print("="*60)
        print(f"{'Total AI Analyses Conducted:':<40} {total_analyses}")
        print(f"{'Initial BUY/SELL Signals Generated:':<40} {len(initial_signals)}")
        print("-" * 60)
        print(f"{Colors.YELLOW}Risk Manager Vetoes:{Colors.END}")
        print(f"{'  - By Market Regime:':<39} {len(vetoed_regime)}")
        print(f"{'  - By Low Conviction (<60):':<39} {len(vetoed_conviction)}")
        print(f"{'  - By Poor Quality/Durability:':<39} {len(vetoed_quality)}")
        print("-" * 60)
        print(f"{Colors.GREEN}{'Final Approved Signals for Trading:':<40} {len(final_approved_signals)}{Colors.END}")
        print("="*60)
    # --- END: AI & RISK FILTER ANALYSIS ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a performance report for a specific backtest batch.")
    parser.add_argument('--batch_id', required=True, help='The unique ID of the backtest batch to analyze.')
    args = parser.parse_args()
    
    generate_performance_report(batch_id=args.batch_id)
    database_manager.close_db_connection()