# run_daily_jobs.py

import pandas as pd
from logger_config import log
import config
import database_manager
import data_retriever
import market_regime_analyzer
import sector_analyzer
from analysis_pipeline import AnalysisPipeline, BENCHMARK_TICKERS
import index_manager

def run_unified_daily_analysis():
    """
    Orchestrates the live, daily analysis for the Nifty 50.
    This is the production engine that runs on a schedule.
    """
    log.info("--- 🚀 Kicking off the LIVE Daily Analysis ---")
    pipeline = None
    try:
        # --- 1. SETUP & DATA PREPARATION ---

        # CRITICAL FIX: For a live run, we must fetch the CURRENT list of Nifty 50 stocks,
        # not a historical point-in-time list.
        log.info("Fetching the current Nifty 50 stock universe...")
        stock_universe = index_manager.get_nifty50_tickers()
        if not stock_universe:
            log.critical("Could not get the Nifty 50 stock universe. Aborting daily run.")
            return

        # CORRECTED LOGIC: Build the full list of assets needed for the analysis.
        # This includes the Nifty 50 stocks, all core sector indices, and global benchmarks.
        tickers_to_load = list(set(
            stock_universe +
            list(sector_analyzer.CORE_SECTOR_INDICES.values()) +
            [sector_analyzer.BENCHMARK_INDEX] +
            list(BENCHMARK_TICKERS.values())
        ))
        
        # Pre-load all required historical data into a cache. The data_retriever will
        # efficiently use its own file cache to speed this up.
        log.info(f"Pre-loading historical data for {len(tickers_to_load)} total assets...")
        full_data_cache = {
            ticker: data_retriever.get_historical_stock_data(ticker) for ticker in tickers_to_load
        }
        # Filter out any assets for which data could not be fetched.
        full_data_cache = {k: v for k, v in full_data_cache.items() if v is not None and not v.empty}
        log.info(f"Successfully loaded data for {len(full_data_cache)} assets into the session cache.")
        
        # --- 2. INITIALIZE AND RUN THE PIPELINE ---
        pipeline = AnalysisPipeline()
        # Set up the pipeline in 'live' mode to connect to the correct database and collections.
        pipeline._setup(mode='live')
        
        # CRITICAL FIX: Use a timezone-aware timestamp for the point-in-time analysis
        # to ensure consistency with UTC data stored in the database.
        point_in_time = pd.Timestamp.now(tz='UTC').floor('D')
        
        log.info(f"Starting pipeline run for point-in-time: {point_in_time.date()}")
        pipeline.run(
            point_in_time=point_in_time,
            full_data_cache=full_data_cache,
            is_backtest=False  # Explicitly flag this as a live run
        )

    except Exception as e:
        log.critical(f"A critical failure occurred in the daily job orchestrator: {e}", exc_info=True)
    
    finally:
        # Ensure database connections are always closed, even if errors occur.
        if pipeline:
            pipeline.close()
        log.info("--- ✅ Daily Analysis Job Finished ---")

if __name__ == "__main__":
    run_unified_daily_analysis()
