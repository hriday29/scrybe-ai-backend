# run_daily_jobs.py (FINAL CORRECTED VERSION)

import pandas as pd
import index_manager
import quantitative_screener
import technical_analyzer
import data_retriever
import market_regime_analyzer
import database_manager
import sector_analyzer
from ai_analyzer import AIAnalyzer
from logger_config import log
import config
from datetime import datetime, timezone
from utils import setup_api_key_iterator, sanitize_context
from sector_analyzer import CORE_SECTOR_INDICES, BENCHMARK_INDEX
import yfinance as yf # Import yfinance for info caching
from analysis_pipeline import AnalysisPipeline

def run_unified_daily_analysis():
    """
    Orchestrates the daily analysis by preparing data and running the pipeline.
    """
    log.info("--- 🚀 Kicking off the Daily Analysis Pipeline ---")
    pipeline = None
    try:
        # --- 1. SETUP & DATA PREPARATION ---
        stock_universe = index_manager.get_nifty50_tickers()
        if not stock_universe:
            log.error("Could not get the stock universe. Aborting.")
            return

        # For the live run, we analyze the full universe.
        from analysis_pipeline import BENCHMARK_TICKERS
        tickers_to_load = stock_universe + list(sector_analyzer.CORE_SECTOR_INDICES.values()) + [sector_analyzer.BENCHMARK_INDEX] + list(BENCHMARK_TICKERS.values())
        
        log.info(f"Pre-caching historical data for {len(tickers_to_load)} assets...")
        full_data_cache = {
            ticker: data_retriever.get_historical_stock_data(ticker) for ticker in tickers_to_load
        }
        full_data_cache = {k: v for k, v in full_data_cache.items() if v is not None and not v.empty}
        log.info(f"Successfully cached data for {len(full_data_cache)} assets.")
        
        # --- 2. RUN THE PIPELINE ---
        pipeline = AnalysisPipeline()
        pipeline._setup(mode='live')
        
        point_in_time = pd.Timestamp.now().floor('D')
        pipeline.run(
            point_in_time=point_in_time,
            full_data_cache=full_data_cache,
            is_backtest=False
        )

    except Exception as e:
        log.critical(f"A critical failure occurred in the daily job orchestrator: {e}", exc_info=True)
    
    finally:
        if pipeline:
            pipeline.close()
        log.info("--- ✅ Daily Analysis Job Finished ---")

if __name__ == "__main__":
    run_unified_daily_analysis()