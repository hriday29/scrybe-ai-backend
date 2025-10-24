# performance_analyzer.py

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import config
from logger_config import log
import database_manager


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
    by fetching closed trades directly from the database using the batch_id.
    Focuses on trade-based metrics.
    """
    log.info(f"--- Generating Final Performance Report for Batch: {batch_id} ---")

    # --- Fetch Closed Trades from Database ---
    try:
        if getattr(database_manager, "performance_collection", None) is None:
            database_manager.init_db(purpose="scheduler")
            if getattr(database_manager, "performance_collection", None) is None:
                log.error(
                    "[REPORT] Cannot generate report: "
                    "'scheduler' database performance collection not available."
                )
                return None

        closed_trades_list = list(
            database_manager.performance_collection.find(
                {"batch_id": batch_id, "status": "Closed"}
            )
        )

        if not closed_trades_list:
            log.warning(
                f"[REPORT] Found 0 closed trades in the database for batch_id '{batch_id}'."
            )
            print("\n" + "=" * 80)
            print(f"### Backtest Performance Summary: Batch '{batch_id}' ###")
            print("=" * 80)
            print("No closed trades found for this batch.")
            print("=" * 80)
            return {"status": "No closed trades found", "batch_id": batch_id}

        closed_trades_df = pd.DataFrame(closed_trades_list)
        log.info(f"Fetched {len(closed_trades_df)} closed trades from database for analysis.")

        if "net_return_pct" not in closed_trades_df.columns:
            log.error("[REPORT] 'net_return_pct' column missing in performance data.")
            return None

        closed_trades_df["net_return_pct"] = pd.to_numeric(
            closed_trades_df["net_return_pct"], errors="coerce"
        )
        closed_trades_df.dropna(subset=["net_return_pct"], inplace=True)

        if closed_trades_df.empty:
            log.warning("[REPORT] No valid closed trades remain after cleaning PnL data.")
            return {"status": "No valid closed trades found after cleaning", "batch_id": batch_id}

    except Exception as e:
        log.error(f"[REPORT] Error fetching or processing trades: {e}", exc_info=True)
        return None

    # --- Trade Metric Calculations ---
    total_trades = len(closed_trades_df)
    _ = config.BACKTEST_PORTFOLIO_CONFIG.get("initial_capital", 0)

    wins_df = closed_trades_df[closed_trades_df["net_return_pct"] > 0]
    losses_df = closed_trades_df[closed_trades_df["net_return_pct"] <= 0]

    win_rate = (len(wins_df) / total_trades) * 100 if total_trades > 0 else 0
    avg_win_pct = wins_df["net_return_pct"].mean() if not wins_df.empty else 0
    avg_loss_pct = losses_df["net_return_pct"].mean() if not losses_df.empty else 0

    expectancy_pct = (
        (win_rate / 100 * avg_win_pct) + ((100 - win_rate) / 100 * avg_loss_pct)
        if total_trades > 0
        else 0
    )

    # --- Print the Report ---
    print("\n" + "=" * 80)
    print(f"### Backtest Performance Summary: Batch '{batch_id}' ###")
    print("=" * 80)
    print(f"{'Total Trades Closed:':<25} {total_trades}")
    print(f"{'Win Rate (%):':<25} {win_rate:.2f}%")
    print(f"{'Avg Win (%):':<25} {avg_win_pct:.2f}%")
    print(f"{'Avg Loss (%):':<25} {avg_loss_pct:.2f}%")
    print(f"{'Expectancy per Trade (%):':<25} {expectancy_pct:.2f}%")
    print("=" * 80)

    # --- Metrics Dict ---
    report_metrics = {
        "batch_id": batch_id,
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate, 2),
        "avg_win_pct": round(avg_win_pct, 2),
        "avg_loss_pct": round(avg_loss_pct, 2),
        "expectancy_pct": round(expectancy_pct, 2),
    }

    # --- Save Report to DB ---
    try:
        if getattr(database_manager, "backtest_reports_collection", None) is None:
            database_manager.init_db(purpose="analysis")
            if getattr(database_manager, "backtest_reports_collection", None) is None:
                log.error(
                    "[REPORT] Cannot save report: "
                    "'analysis' database reports collection not available."
                )
                return report_metrics

        report_name = f"{config.APEX_SWING_STRATEGY['name']}_Batch_{batch_id}"

        report_data_to_save = report_metrics.copy()
        report_data_to_save.update(
            {
                "report_name": report_name,
                "last_updated": datetime.now(timezone.utc),
                "backtest_start_date": closed_trades_df["open_date"].min()
                if not closed_trades_df.empty
                else None,
                "backtest_end_date": closed_trades_df["close_date"].max()
                if not closed_trades_df.empty
                else None,
            }
        )

        database_manager.backtest_reports_collection.update_one(
            {"batch_id": batch_id},
            {"$set": report_data_to_save},
            upsert=True,
        )
        log.info(
            f"✅ Successfully saved backtest report metrics to ANALYSIS database for batch {batch_id}."
        )

    except Exception as e:
        log.error(f"[REPORT] Failed to save backtest report: {e}", exc_info=True)

    return report_metrics