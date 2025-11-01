"""
performance_analyzer.py

Purpose
- Generates a comprehensive backtest performance report by delegating core
    computations to analyze_backtest_from_db, and persists the final report document
    to the analysis database for the UI.

How it fits
- Called after the risk-based backtest simulator writes closed trades. Wraps the
    analyzer output with batch metadata and saves to the reports collection.

Main role
- Orchestrate analysis/report saving and provide small helpers like historical
    performance snapshots and market correlations to enrich views when needed.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import config
from logger_config import log
import database_manager

# --- IMPORT THE NEW, CORRECT ANALYZER ---
from analyze_backtest_db import analyze_backtest_from_db


def calculate_historical_performance(historical_data: pd.DataFrame) -> dict:
    """
    Calculates the stock's price performance over standard trading day periods.
    """
    if historical_data.empty:
        return {}

    log.info("Calculating historical performance snapshot...")
    performance: dict[str, dict[str, str]] = {}

    # Use standard trading days instead of calendar days
    periods = {
        "1-Week": 5,
        "1-Month": 21,
        "3-Month": 63,
        "6-Month": 126,
        "1-Year": 252,
    }
    latest_price = historical_data["close"].iloc[-1]

    for name, days in periods.items():
        if len(historical_data) > days:
            old_price = historical_data["close"].iloc[-days]
            change_pct = ((latest_price - old_price) / old_price) * 100
            performance[name] = {"change_percent": f"{change_pct:.2f}%"}
        else:
            performance[name] = {"change_percent": "N/A"}

    log.info("Successfully calculated performance snapshot.")
    return performance


def calculate_correlations(
    stock_historical_data: pd.DataFrame, benchmarks_data: pd.DataFrame
) -> dict:
    """
    Calculates the 90-day correlation between a stock's returns and the returns
    of key benchmarks.
    """
    log.info("Calculating inter-market correlations...")
    if (
        stock_historical_data is None
        or benchmarks_data is None
        or stock_historical_data.empty
        or benchmarks_data.empty
    ):
        return {}

    try:
        # Using pct_change() for returns is the more robust method
        stock_returns = stock_historical_data["close"].pct_change().rename("Stock")
        benchmark_returns = benchmarks_data.pct_change(fill_method=None)

        combined_returns = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()

        if len(combined_returns) < 90:
            log.warning("Not enough data for 90-day correlation.")
            return {}

        correlation_matrix = combined_returns.tail(90).corr()
        stock_correlations = correlation_matrix["Stock"].drop("Stock")

        formatted_correlations = {
            f"{k} Correlation": f"{v:.2f}" for k, v in stock_correlations.items()
        }
        log.info(f"Successfully calculated {len(formatted_correlations)} correlations.")
        return formatted_correlations

    except Exception as e:
        log.error(f"An error occurred during correlation calculation: {e}")
        return {}


def generate_backtest_report(batch_id: str) -> dict | None:
    """
    Generates a comprehensive performance summary for a completed backtest
    by calling the consolidated 'analyze_backtest_from_db' function
    and then saving its full results to the analysis database.
    """
    log.info(f"--- Generating Final Performance Report for Batch: {batch_id} ---")
    log.info("Calling main analyzer 'analyze_backtest_from_db'...")

    try:
        # Call the new, consolidated analyzer.
        # This function now handles all calculations AND prints the detailed report.
        report_metrics = analyze_backtest_from_db(batch_id)

        if not report_metrics:
            log.warning(f"[REPORT] analyze_backtest_from_db returned no metrics for batch {batch_id}.")
            return {"status": "No metrics returned from analyzer", "batch_id": batch_id}
        
        # Add batch_id to the metrics if it's not already there
        report_metrics.setdefault("batch_id", batch_id)

    except Exception as e:
        log.error(f"[REPORT] Error running 'analyze_backtest_from_db': {e}", exc_info=True)
        return None

    # --- Metrics Dict (is now populated with ALL metrics from the analyzer) ---
    log.info(f"Analyzer returned {len(report_metrics)} metrics. Proceeding to save to DB.")

    # --- Save Report to DB ---
    try:
        if getattr(database_manager, "backtest_reports_collection", None) is None:
            database_manager.init_db(purpose="analysis")
            if getattr(database_manager, "backtest_reports_collection", None) is None:
                log.error(
                    "[REPORT] Cannot save report: "
                    "'analysis' database reports collection not available."
                )
                return report_metrics  # Return the metrics even if saving fails

        report_name = f"{config.APEX_SWING_STRATEGY['name']}_Batch_{batch_id}"

        # The report_metrics dict already has everything:
        # (final_equity, total_net_pnl, total_return_pct, max_drawdown_pct, 
        # sharpe_ratio, profit_factor, total_trades, win_rate_pct, etc.)
        
        # We just need to add the metadata
        report_data_to_save = report_metrics.copy()
        report_data_to_save["report_name"] = report_name
        report_data_to_save["last_updated"] = datetime.now(timezone.utc)
        
        # Fetch start/end dates from the performance DB
        try:
            if getattr(database_manager, "performance_collection", None) is None:
                database_manager.init_db(purpose="scheduler")
            
            pipeline = [
                {"$match": {"batch_id": batch_id, "status": "Closed"}},
                {"$group": {
                    "_id": "$batch_id",
                    "backtest_start_date": {"$min": "$open_date"},
                    "backtest_end_date": {"$max": "$close_date"}
                }}
            ]
            date_agg = list(database_manager.performance_collection.aggregate(pipeline))
            if date_agg:
                report_data_to_save["backtest_start_date"] = date_agg[0].get("backtest_start_date")
                report_data_to_save["backtest_end_date"] = date_agg[0].get("backtest_end_date")
            else:
                log.warning("Could not find trades to aggregate start/end dates for report.")
        except Exception as date_e:
            log.warning(f"Could not aggregate start/end dates for report: {date_e}")

        # Now, we save the COMPLETE report dictionary
        database_manager.backtest_reports_collection.update_one(
            {"batch_id": batch_id},
            {"$set": report_data_to_save},
            upsert=True,
        )
        log.info(
            f"✅ Successfully saved FULL backtest report to ANALYSIS database for batch {batch_id}."
        )

    except Exception as e:
        log.error(f"[REPORT] Failed to save backtest report: {e}", exc_info=True)

    return report_metrics