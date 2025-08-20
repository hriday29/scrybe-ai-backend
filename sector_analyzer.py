# sector_analyzer.py (ROBUST VERSION 2.0)
import pandas as pd
import config
import data_retriever
from logger_config import log

# We are using a more curated and reliable list of sector indices.
CORE_SECTOR_INDICES = {
    "NIFTY Bank": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "NIFTY Auto": "^CNXAUTO",
    "NIFTY Pharma": "^CNXPHARMA",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY Metal": "^CNXMETAL",
    "NIFTY PSU Bank": "^CNXPSUBANK",
    "NIFTY Oil & Gas": "^CNXENERGY",
    "NIFTY India Consumption": "^CNXCONSUM",
}

# --- THE FIX: Using Nifty 50 as the benchmark is far more reliable ---
BENCHMARK_INDEX = "^NSEI"
LOOKBACK_PERIOD_DAYS = 21
TOP_N_SECTORS = 5

def get_top_performing_sectors() -> list[str]:
    """
    Calculates the performance of core sectors relative to the Nifty 50 benchmark
    over a defined lookback period.
    """
    log.info("--- [Funnel Step 2] Analyzing Sector Relative Strength (Robust Mode) ---")
    
    all_tickers = list(CORE_SECTOR_INDICES.values()) + [BENCHMARK_INDEX]
    all_historical_data = {}

    for ticker in all_tickers:
        data = data_retriever.get_historical_stock_data(ticker)
        if data is None or len(data) < LOOKBACK_PERIOD_DAYS + 5:
            log.warning(f"Could not fetch sufficient historical data for {ticker}. Skipping it.")
            continue
        all_historical_data[ticker] = data['close']

    if BENCHMARK_INDEX not in all_historical_data:
        log.error(f"Could not fetch data for benchmark index {BENCHMARK_INDEX}. Cannot perform sector analysis.")
        return []

    performance = {}
    for ticker, close_prices in all_historical_data.items():
        price_now = close_prices.iloc[-1]
        price_then = close_prices.iloc[-1 - LOOKBACK_PERIOD_DAYS]
        performance[ticker] = ((price_now - price_then) / price_then) * 100 if price_then != 0 else 0

    benchmark_performance = performance.pop(BENCHMARK_INDEX)
    log.info(f"Benchmark ({BENCHMARK_INDEX}) {LOOKBACK_PERIOD_DAYS}-day performance: {benchmark_performance:.2f}%")

    outperforming_sectors = {
        ticker: perf for ticker, perf in performance.items() if perf > benchmark_performance
    }

    sorted_sectors = sorted(outperforming_sectors.items(), key=lambda item: item[1], reverse=True)
    
    top_sector_names = []
    for ticker, perf in sorted_sectors[:TOP_N_SECTORS]:
        sector_name = next((name for name, t in CORE_SECTOR_INDICES.items() if t == ticker), ticker)
        top_sector_names.append(sector_name)
        log.info(f"  -> Strong Sector Found: {sector_name} (+{perf:.2f}%)")

    if not top_sector_names:
        log.warning("No sectors were found to be outperforming the market benchmark.")

    log.info(f"✅ Sector analysis complete. Identified {len(top_sector_names)} strong sector(s).")
    return top_sector_names