# _dev_tools/generate_partial_report.py

import argparse
import pandas as pd
from logger_config import log
import sys
import os

# Ensure the script can find project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    import database_manager
    import performance_analyzer
    import config
except ImportError as e:
    log.fatal(f"Failed to import project modules. Make sure script is run from project root or _dev_tools folder. Error: {e}")
    sys.exit(1)

def run_partial_report(batch_id: str):
    """
    Fetches completed trades for a given batch_id from the scheduler DB
    and generates a performance report based on that partial data.
    """
    log.info(f"--- Generating Partial Performance Report for Batch: {batch_id} ---")

    try:
        # 1. Connect to the SCHEDULER database
        database_manager.init_db(purpose='scheduler')
        if database_manager.db is None or database_manager.performance_collection is None:
            log.error("Failed to initialize scheduler database connection.")
            return

        # 2. Fetch all closed trades for the batch
        log.info(f"Fetching closed trades for batch_id='{batch_id}'...")
        closed_trades = list(database_manager.performance_collection.find({"batch_id": batch_id}))
        log.info(f"Found {len(closed_trades)} closed trades.")

        if not closed_trades:
            log.warning("No closed trades found for this batch_id. Cannot generate report.")
            return

        # 3. Reconstruct a simplified portfolio object
        initial_capital = config.BACKTEST_PORTFOLIO_CONFIG['initial_capital']
        partial_portfolio = {
            'equity': initial_capital, # Will be updated below
            'open_positions': [], # Assume no open positions for partial report
            'closed_trades': closed_trades,
            'daily_equity_log': []
        }

        # 4. Reconstruct a simplified daily equity log based on closed trades
        log.info("Reconstructing simplified equity log...")
        equity_log = [{'date': pd.Timestamp.min.tz_localize('UTC'), 'equity': initial_capital}] # Start point
        current_equity = initial_capital

        # Sort trades by close date to apply P&L chronologically
        try:
            # Convert date fields to datetime objects for proper sorting
            for trade in closed_trades:
                 if isinstance(trade.get('close_date'), str):
                     trade['close_date'] = pd.to_datetime(trade['close_date']) # Simple conversion, might need timezone handling if inconsistent
                 elif not isinstance(trade.get('close_date'), pd.Timestamp):
                      # Handle cases where it might be a native datetime or None
                      trade['close_date'] = pd.to_datetime(trade.get('close_date'), errors='coerce', utc=True)

            trades_df = pd.DataFrame(closed_trades).sort_values('close_date')

            for _, trade in trades_df.iterrows():
                # Ensure net_pnl is present and numeric
                net_pnl = trade.get('net_pnl', 0)
                if pd.notna(net_pnl):
                    current_equity += net_pnl
                    # Ensure close_date is valid
                    close_date = trade.get('close_date')
                    if pd.notna(close_date):
                         # Ensure timezone consistency if needed, assuming UTC here
                         equity_log.append({'date': close_date, 'equity': round(current_equity, 2)})
                    else:
                         log.warning(f"Skipping equity log entry for trade due to missing close_date: {trade.get('ticker')}")
                else:
                    log.warning(f"Skipping P&L calculation for trade due to missing net_pnl: {trade.get('ticker')}")


            partial_portfolio['daily_equity_log'] = equity_log
            # Set final equity for the report function
            partial_portfolio['equity'] = current_equity
            log.info(f"Reconstructed equity log with {len(equity_log)} points. Final simulated equity: {current_equity:,.2f}")

        except Exception as e:
             log.error(f"Error reconstructing equity log: {e}", exc_info=True)
             log.warning("Proceeding with report generation using only closed trade stats (no equity curve/drawdown).")
             # Ensure equity log is at least present as an empty list
             partial_portfolio['daily_equity_log'] = []


        # 5. Generate and Print the Report
        log.info("Calling performance_analyzer.generate_backtest_report...")
        # Note: Metrics like Drawdown and Sharpe Ratio might be inaccurate
        # if the equity log reconstruction is incomplete or simplified.
        report_metrics = performance_analyzer.generate_backtest_report(partial_portfolio, batch_id)

        print("\n" + "="*50)
        print(" Partial Backtest Report Metrics (JSON)")
        print("="*50)
        print(report_metrics)
        print("="*50)
        log.info("--- Partial Report Generation Complete ---")

    except Exception as e:
        log.critical(f"An error occurred during partial report generation: {e}", exc_info=True)
    finally:
        # 6. Close DB connection
        database_manager.close_db_connection()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a partial performance report for an incomplete backtest.")
    parser.add_argument("--batch_id", required=True, help="The batch_id of the backtest run to analyze.")
    args = parser.parse_args()

    run_partial_report(batch_id=args.batch_id)