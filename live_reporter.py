# live_reporter.py
from datetime import datetime, timedelta, timezone
import pandas as pd
import database_manager
from logger_config import log

def _calculate_metrics(trades_df: pd.DataFrame) -> dict:
    """A helper function to calculate performance KPIs from a DataFrame of trades."""
    
    if trades_df.empty:
        return {
            "Total Trades": 0, "Win Rate": "N/A", "Profit Factor": "N/A",
            "Average Gain %": "N/A", "Average Loss %": "N/A",
            "Biggest Winner %": "N/A", "Biggest Loser %": "N/A"
        }

    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['return_pct'] > 0]
    losing_trades = trades_df[trades_df['return_pct'] < 0]
    
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    
    # --- Profit Factor Calculation ---
    total_gains = winning_trades['return_pct'].sum()
    total_losses = abs(losing_trades['return_pct'].sum())
    profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')

    return {
        "Total Trades": total_trades,
        "Win Rate": f"{win_rate:.2f}%",
        "Profit Factor": f"{profit_factor:.2f}" if profit_factor != float('inf') else "inf",
        "Average Gain %": f"{winning_trades['return_pct'].mean():.2f}%" if not winning_trades.empty else "N/A",
        "Average Loss %": f"{losing_trades['return_pct'].mean():.2f}%" if not losing_trades.empty else "N/A",
        "Biggest Winner %": f"{winning_trades['return_pct'].max():.2f}%" if not winning_trades.empty else "N/A",
        "Biggest Loser %": f"{losing_trades['return_pct'].min():.2f}%" if not losing_trades.empty else "N/A",
    }

def _print_report(title: str, metrics: dict):
    """A helper function to print a formatted report to the console."""
    log.info(f"--- {title} Performance Summary ---")
    for key, value in metrics.items():
        log.info(f"{key:<20}: {value}")
    log.info("-" * 40)

def generate_performance_report():
    """
    Fetches all closed trades and generates dynamic performance reports for
    today, the current week (WTD), and the current month (MTD).
    """
    log.info("--- 📊 Generating Daily Performance Report ---")
    database_manager.init_db(purpose='analysis')
    
    all_closed_trades = database_manager.get_live_track_record()
    
    if not all_closed_trades:
        log.warning("No closed trades found in 'live_performance' collection. Skipping report.")
        return
        
    df = pd.DataFrame(all_closed_trades)
    df['close_date'] = pd.to_datetime(df['close_date'])

    # --- Define Time Periods ---
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=now.weekday()) # Monday
    month_start = today_start.replace(day=1)

    # --- Filter DataFrames for each period ---
    today_df = df[df['close_date'] >= today_start]
    wtd_df = df[df['close_date'] >= week_start]
    mtd_df = df[df['close_date'] >= month_start]
    
    # --- Calculate and Print Metrics ---
    today_metrics = _calculate_metrics(today_df)
    wtd_metrics = _calculate_metrics(wtd_df)
    mtd_metrics = _calculate_metrics(mtd_df)
    
    _print_report("Today's", today_metrics)
    _print_report("Week-to-Date", wtd_metrics)
    _print_report("Month-to-Date", mtd_metrics)

if __name__ == '__main__':
    # This allows you to run the report manually for testing
    generate_performance_report()