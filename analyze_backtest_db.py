#analyze_backtest_db.py
import os
import pandas as pd
import numpy as np
import argparse
import pymongo
import certifi
import quantstats as qs

# Assuming these modules are in the same project directory
import config
import database_manager
from logger_config import log
from datetime import datetime

# --- Configuration ---
# REPORTS_DIR is now defined in config.py
REPORT_FILENAME_TEMPLATE = os.path.join(config.REPORTS_DIR, '{batch_id}_analysis_summary.txt')
INITIAL_CAPITAL = config.BACKTEST_PORTFOLIO_CONFIG['initial_capital']
RISK_PER_TRADE_PCT = config.BACKTEST_PORTFOLIO_CONFIG['risk_per_trade_pct']

# --- Helper for output ---
class DualWriter:
    """Writes output to both console and a file."""
    def __init__(self, *targets):
        self.targets = targets
    def write(self, text):
        for t in self.targets:
            t.write(str(text) + "\n") # Ensure newline
    def flush(self):
        for t in self.targets:
            t.flush()

def analyze_backtest_from_db(batch_id: str):
    """
    Analyzes backtest results stored in the database for a specific batch_id.
    Focuses on trade-based metrics and includes placeholders for portfolio metrics.
    """
    log.info(f"--- Starting Database Backtest Analysis for Batch: {batch_id} ---")
    report_file = REPORT_FILENAME_TEMPLATE.format(batch_id=batch_id)
    os.makedirs(os.path.dirname(report_file), exist_ok=True)

    # --- Fetch Data ---
    try:
        if getattr(database_manager, "performance_collection", None) is None:
            database_manager.init_db(purpose="scheduler")
            if getattr(database_manager, "performance_collection", None) is None:
                log.error("Failed to initialize database connection to 'scheduler'.")
                return

        log.info(f"Fetching closed trades for batch_id '{batch_id}' from performance collection...")
        query = {"batch_id": batch_id, "status": "Closed"}
        try:
            closed_trades_list = list(database_manager.performance_collection.find(query))
        except pymongo.errors.InvalidOperation as inv_e:
            # Handle stale/closed client case gracefully by re-initializing once
            if "after close" in str(inv_e):
                log.warning("Performance collection bound to a closed MongoClient. Re-initializing DB (scheduler) and retrying once...")
                database_manager.init_db(purpose="scheduler")
                closed_trades_list = list(database_manager.performance_collection.find(query))
            else:
                raise

        if not closed_trades_list:
            log.warning(f"No closed trades found in the database for batch_id '{batch_id}'.")
            with open(report_file, 'w', encoding='utf-8') as f:
                 f.write(f"No closed trades found for batch_id: {batch_id}\n")
            print(f"Analysis summary (empty) saved to: {report_file}")
            return

        df = pd.DataFrame(closed_trades_list)
        log.info(f"Successfully loaded {len(df)} closed trades from database.")

        # Add cleaning for net_pnl
        if 'net_pnl' not in df.columns:
            log.warning("[REPORT] 'net_pnl' column missing. Cannot calculate portfolio metrics.")
            df['net_pnl'] = 0  # Add a zero column to prevent errors later
        else:
            df['net_pnl'] = pd.to_numeric(df.get('net_pnl'), errors='coerce')
            df['net_pnl'].fillna(0, inplace=True)  # Replace failed conversions with 0 P&L

    except Exception as e:
        log.error(f"Failed to fetch or process trades from database: {e}", exc_info=True)
        return

    # --- Data Cleaning & Preparation ---
    required_cols = ['open_date', 'close_date', 'net_return_pct', 'closing_reason', 'signal', 'ticker']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        log.error(f"Database data is missing required columns: {', '.join(missing_cols)}")
        return

    # Convert dates and calculate holding period
    try:
        df['open_date'] = pd.to_datetime(df['open_date'], errors='coerce')
        df['close_date'] = pd.to_datetime(df['close_date'], errors='coerce')
        df.dropna(subset=['open_date', 'close_date'], inplace=True)
        if df.empty:
            log.error("No valid date entries found after cleaning.")
            return
        # Use business days for holding period calculation
        df['holding_period_days'] = np.busday_count(df['open_date'].values.astype('M8[D]'),
                                                  df['close_date'].values.astype('M8[D]'))
        df['holding_period_days'] = df['holding_period_days'].apply(lambda x: max(x, 1)) # Min 1 day

    except Exception as e:
        log.error(f"Could not process date columns: {e}")
        return

    # Convert PnL % to numeric
    df['net_return_pct'] = pd.to_numeric(df['net_return_pct'], errors='coerce')
    df.dropna(subset=['net_return_pct'], inplace=True)

    if df.empty:
        log.warning("No valid trades remain after cleaning PnL data.")
        return

    # Sort by close date for potential future equity curve calculation
    df.sort_values(by='close_date', inplace=True)

    # --- Reconstruct Equity Curve ---
    df['equity'] = INITIAL_CAPITAL + df['net_pnl'].cumsum()
    equity_curve = df.set_index('close_date')['equity']  # Series with dates as index
    # Add initial capital point
    start_date = df['open_date'].min() - pd.Timedelta(days=1)  # Day before first trade
    equity_curve = pd.concat([pd.Series({start_date: INITIAL_CAPITAL}), equity_curve])
    equity_curve = equity_curve.resample('D').last().ffill()  # Resample to daily, forward fill weekends/holidays
    daily_returns = equity_curve.pct_change().fillna(0)

    # --- Portfolio Metric Calculations ---
    final_equity = float(equity_curve.iloc[-1]) if not equity_curve.empty else float(INITIAL_CAPITAL)
    total_net_pnl = final_equity - INITIAL_CAPITAL
    total_return_pct = (total_net_pnl / INITIAL_CAPITAL) * 100 if INITIAL_CAPITAL > 0 else 0

    # Calculate Drawdown from equity curve
    peak = equity_curve.cummax() if not equity_curve.empty else equity_curve
    drawdown = (peak - equity_curve) / peak if not equity_curve.empty else pd.Series(dtype=float)
    max_drawdown_pct = float(drawdown.max() * 100) if not drawdown.empty else 0.0

    # Calculate Sharpe Ratio using quantstats for robustness
    try:
        sharpe_ratio_val = qs.stats.sharpe(daily_returns, periods=252)  # periods=252 for daily data annualized
        # Handle nan/None cases
        sharpe_ratio = float(sharpe_ratio_val) if pd.notnull(sharpe_ratio_val) else 0.0
    except Exception:
        sharpe_ratio = 0.0

    # Calculate Profit Factor from absolute P&Ls
    gross_profit = float(df[df['net_pnl'] > 0]['net_pnl'].sum())
    gross_loss_val = df[df['net_pnl'] < 0]['net_pnl'].sum()
    gross_loss = float(abs(gross_loss_val))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)

    # Prepare report metrics dict to return
    report_metrics = {
        "final_equity": round(final_equity, 2),
        "total_net_pnl": round(total_net_pnl, 2),
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "profit_factor": round(profit_factor, 2),
    }

    # --- Analysis & Reporting ---
    with open(report_file, 'w', encoding='utf-8') as f:
        writer = DualWriter(f, os.sys.stdout) # Use DualWriter

        writer.write("\n" + "=" * 80)
        writer.write(f"### Backtest Performance Analysis: Batch '{batch_id}' ###")
        writer.write(f"Period: {df['open_date'].min().date()} to {df['close_date'].max().date()}")
        writer.write("=" * 80)

        # --- Trade Statistics ---
        writer.write("\n--- Trade Statistics ---")
        total_trades = len(df)
        wins = df[df['net_return_pct'] > 0]
        losses = df[df['net_return_pct'] <= 0]
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0

        avg_win_pct = wins['net_return_pct'].mean() if not wins.empty else 0
        avg_loss_pct = losses['net_return_pct'].mean() if not losses.empty else 0
        median_win_pct = wins['net_return_pct'].median() if not wins.empty else 0
        median_loss_pct = losses['net_return_pct'].median() if not losses.empty else 0

        expectancy_pct = (win_rate/100 * avg_win_pct) + ((100-win_rate)/100 * avg_loss_pct) if total_trades > 0 else 0

        # Profit Factor cannot be accurately calculated without absolute PnL
        # avg_rr_ratio = abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else float('inf')

        writer.write(f"{'Total Trades Closed:':<30} {total_trades}")
        writer.write(f"{'Win Rate:':<30} {win_rate:.2f}%")
        writer.write(f"{'Avg Win %:':<30} {avg_win_pct:.2f}%")
        writer.write(f"{'Avg Loss %:':<30} {avg_loss_pct:.2f}%")
        writer.write(f"{'Median Win %:':<30} {median_win_pct:.2f}%")
        writer.write(f"{'Median Loss %:':<30} {median_loss_pct:.2f}%")
        # writer.write(f"{'Avg Reward/Risk Ratio (%):':<30} {avg_rr_ratio:.2f}")
        writer.write(f"{'Expectancy per Trade (%):':<30} {expectancy_pct:.2f}%")
        writer.write(f"{'Avg Holding Period (Bus Days):':<30} {df['holding_period_days'].mean():.1f}")
        writer.write(f"{'Median Holding Period (Bus Days):':<30} {df['holding_period_days'].median():.1f}")

        # --- Update Portfolio Metrics Section ---
        writer.write("\n" + "-" * 80)
        writer.write("--- Portfolio Metrics ---")
        writer.write(f"{'Initial Capital:':<30} ₹{INITIAL_CAPITAL:,.2f}")
        writer.write(f"{'Final Portfolio Value:':<30} ₹{final_equity:,.2f}")
        writer.write(f"{'Total Net P&L:':<30} ₹{total_net_pnl:,.2f}")
        writer.write(f"{'Total Return %:':<30} {total_return_pct:.2f}%")
        writer.write(f"{'Max Drawdown %:':<30} {max_drawdown_pct:.2f}%")
        writer.write(f"{'Sharpe Ratio (Annualized):':<30} {sharpe_ratio:.2f}")
        writer.write(f"{'Profit Factor:':<30} {profit_factor:.2f}")
        writer.write("-" * 80)

        # --- Generate QuantStats HTML Report (Optional) ---
        try:
            qs_report_file = os.path.join(config.REPORTS_DIR, f'{batch_id}_quantstats_report.html')
            qs.reports.html(daily_returns, output=qs_report_file, title=f'QuantStats Report: Batch {batch_id}')
            log.info(f"Generated QuantStats HTML report: {qs_report_file}")
            writer.write(f"\nQuantStats HTML report generated at: {qs_report_file}")
        except Exception as qe:
            log.error(f"Failed to generate QuantStats report: {qe}")
        # --- End QuantStats Report ---

        # --- Performance by Holding Period ---
        writer.write("\n" + "=" * 80)
        writer.write("### Performance by Holding Period (Business Days) ###")
        writer.write("=" * 80)

        bins = [0, 5, 10, 15, 20, 30, float('inf')]
        labels = ['1-5 Days', '6-10 Days', '11-15 Days', '16-20 Days', '21-30 Days', '31+ Days']
        df['holding_period_group'] = pd.cut(df['holding_period_days'], bins=bins, labels=labels, right=True, include_lowest=True)

        grouped_holding = df.groupby('holding_period_group', observed=False).agg(
            num_trades=('ticker', 'size'),
            win_rate=('net_return_pct', lambda x: (x > 0).mean() * 100),
            avg_return_pct=('net_return_pct', 'mean'),
            median_return_pct=('net_return_pct', 'median'),
            expectancy_pct=('net_return_pct', lambda x: ( (x>0).mean() * x[x>0].mean() + (x<=0).mean() * x[x<=0].mean() ) if len(x)>0 else 0),
        ).reset_index()

        writer.write(grouped_holding.to_string(float_format="%.2f", index=False))

        # --- Performance by Closing Reason ---
        writer.write("\n" + "=" * 80)
        writer.write("### Performance by Closing Reason ###")
        writer.write("=" * 80)

        grouped_reason = df.groupby('closing_reason').agg(
            num_trades=('ticker', 'size'),
            win_rate=('net_return_pct', lambda x: (x > 0).mean() * 100),
            avg_return_pct=('net_return_pct', 'mean'),
            median_return_pct=('net_return_pct', 'median'),
            expectancy_pct=('net_return_pct', lambda x: ( (x>0).mean() * x[x>0].mean() + (x<=0).mean() * x[x<=0].mean() ) if len(x)>0 else 0),
        ).sort_values('num_trades', ascending=False).reset_index()

        writer.write(grouped_reason.to_string(float_format="%.2f", index=False))

        # --- Performance by Signal Type ---
        writer.write("\n" + "=" * 80)
        writer.write("### Performance by Signal Type ###")
        writer.write("=" * 80)

        if 'signal' in df.columns:
            grouped_signal = df.groupby('signal').agg(
                num_trades=('ticker', 'size'),
                win_rate=('net_return_pct', lambda x: (x > 0).mean() * 100),
                avg_return_pct=('net_return_pct', 'mean'),
                median_return_pct=('net_return_pct', 'median'),
                expectancy_pct=('net_return_pct', lambda x: ( (x>0).mean() * x[x>0].mean() + (x<=0).mean() * x[x<=0].mean() ) if len(x)>0 else 0),
            ).reset_index()
            writer.write(grouped_signal.to_string(float_format="%.2f", index=False))
        else:
            writer.write("Column 'signal' not found in data.")

        # --- Biggest Winners and Losers ---
        writer.write("\n" + "=" * 80)
        writer.write("### Top 5 Winning Trades (by Return %) ###")
        writer.write("=" * 80)
        writer.write(df.nlargest(5, 'net_return_pct')[['ticker', 'signal', 'open_date', 'close_date',
                'holding_period_days', 'net_return_pct', 'closing_reason']].to_string(index=False))

        writer.write("\n" + "=" * 80)
        writer.write("### Top 5 Losing Trades (by Return %) ###")
        writer.write("=" * 80)
        writer.write(df.nsmallest(5, 'net_return_pct')[['ticker', 'signal', 'open_date', 'close_date',
                'holding_period_days', 'net_return_pct', 'closing_reason']].to_string(index=False))

        writer.write("\n" + "=" * 80)
        writer.write("### Analysis Complete ###")
        writer.write("=" * 80)

    print(f"\nDetailed analysis summary saved to: {report_file}")

    # Return computed metrics for programmatic use
    return report_metrics

# --- CLI Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze backtest results stored in MongoDB.")
    parser.add_argument("--batch_id", required=True, help="Unique ID of the backtest batch to analyze.")
    args = parser.parse_args()

    analyze_backtest_from_db(batch_id=args.batch_id)
