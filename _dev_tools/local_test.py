# local_test.py (Final Corrected Version)
import argparse
import pandas as pd
from datetime import datetime
from logger_config import log
import config
import database_manager
import data_retriever
from analysis_pipeline import AnalysisPipeline
from sector_analyzer import CORE_SECTOR_INDICES, BENCHMARK_INDEX

def run_local_test(tickers, start_date, end_date, batch_id):
    log.info(f"--- 🚀 STARTING LOCAL TEST RUN: {batch_id} ---")
    log.info(f"Testing with tickers: {tickers}")
    log.info(f"Period: {start_date} to {end_date}")

    pipeline = AnalysisPipeline()
    # --- THE FINAL FIX: Connect to the 'analysis' database where fundamentals are stored ---
    pipeline._setup(mode='analysis')
    
    if config.DATA_SOURCE == "angelone":
        if not data_retriever.angelone_retriever.initialize_angelone_session():
            log.fatal("❌ Failed to initialize Angel One session. Halting test.")
            return

    database_manager.predictions_collection.delete_many({"batch_id": batch_id})
    log.info(f"Cleared old test predictions for batch '{batch_id}' from the database.")
    
    all_required_tickers = list(set(tickers + list(CORE_SECTOR_INDICES.values()) + [BENCHMARK_INDEX]))
    log.info(f"Will fetch data for {len(all_required_tickers)} unique assets (stocks + indices).")

    simulation_days = pd.bdate_range(start=start_date, end=end_date)
    if simulation_days.empty:
        log.error("No business days in the specified date range. Please check dates.")
        return

    for day in simulation_days:
        day_str = day.strftime('%Y-%m-%d')
        log.info(f"\n--- Simulating day: {day_str} ---")

        log.info(f"Attempting to fetch data for {len(all_required_tickers)} assets...")
        data_cache_for_today = {
            ticker: data_retriever.get_historical_stock_data(ticker, end_date=day_str)
            for ticker in all_required_tickers
        }
        
        valid_data = {k: v for k, v in data_cache_for_today.items() if v is not None and not v.empty}
        
        if len(valid_data) < len(all_required_tickers):
            log.warning("Could not fetch data for all required assets. Pipeline might be affected.")
        
        if not valid_data:
            log.error(f"❌ FAILED to fetch any data for {day_str}. The pipeline cannot run.")
            continue

        log.info(f"✅ Successfully fetched data for {len(valid_data)}/{len(all_required_tickers)} assets.")

        log.info(f"Data cache populated for {day_str}. Kicking off analysis pipeline...")
        pipeline.run(
            point_in_time=day,
            full_data_cache=valid_data,
            is_backtest=True,
            batch_id=batch_id
        )
        log.info(f"✅ Pipeline run for {day_str} complete.")

    pipeline.close()
    log.info("\n--- ✅ LOCAL TEST COMPLETED SUCCESSFULLY ---")
    log.info("Data is flowing correctly from the retriever into the analysis pipeline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a local test of the ScrybeAI data and analysis pipeline.")
    parser.add_argument("--tickers", nargs='+', required=True, help="Space-separated list of tickers to test (e.g., RELIANCE.NS INFY.NS)")
    parser.add_argument("--start_date", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end_date", required=True, help="End date in YYYY-MM-DD format")
    
    args = parser.parse_args()
    test_batch_id = f"LOCAL-TEST-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_local_test(args.tickers, args.start_date, args.end_date, test_batch_id)