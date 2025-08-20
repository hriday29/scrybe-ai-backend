# sector_analyzer.py
import pandas as pd
import config
import data_retriever
from logger_config import log

# Define which indices from the config file represent the core sectors we want to track.
# We exclude broad market, thematic, and non-NSE indices for this analysis.
# (in sector_analyzer.py)
CORE_SECTOR_INDICES = {
    "NIFTY Bank": "^NSEBANK",
    "NIFTY Financial Services": "NIFTY_FIN_SERVICE.NS", # This one is often unstable, but we'll try it.
    "NIFTY IT": "^CNXIT",
    "NIFTY Auto": "^CNXAUTO",
    "NIFTY Pharma": "^CNXPHARMA",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY Metal": "^CNXMETAL",
    "NIFTY PSU Bank": "^CNXPSUBANK",
    "NIFTY Private Bank": "^CNXPRIVBANK", # Corrected Ticker
    "NIFTY Oil & Gas": "^CNXENERGY",      # Corrected Ticker
    "NIFTY India Consumption": "^CNXCONSUM",
}

BENCHMARK_INDEX = "^CNX500" # Using Nifty 500 as a broad market benchmark
LOOKBACK_PERIOD_DAYS = 21 # Approximately one trading month
TOP_N_SECTORS = 5 # The number of top sectors we want to focus on

def get_top_performing_sectors() -> list[str]:
    """
    Calculates the performance of core sectors relative to the Nifty 500 benchmark
    over a defined lookback period.

    Returns:
        list[str]: A list of the names of the top N outperforming sectors.
                   Returns an empty list if the analysis cannot be completed.
    """
    log.info("--- [Funnel Step 2] Analyzing Sector Relative Strength ---")
    
    all_tickers = list(CORE_SECTOR_INDICES.values()) + [BENCHMARK_INDEX]
    all_historical_data = {}

    # 1. Fetch data for all indices
    for ticker in all_tickers:
        # Fetching a bit more data to ensure we have enough for the lookback
        data = data_retriever.get_historical_stock_data(ticker)
        if data is None or len(data) < LOOKBACK_PERIOD_DAYS + 5:
            log.warning(f"Could not fetch sufficient historical data for {ticker}. Skipping it.")
            continue
        all_historical_data[ticker] = data['close']

    if BENCHMARK_INDEX not in all_historical_data:
        log.error(f"Could not fetch data for benchmark index {BENCHMARK_INDEX}. Cannot perform sector analysis.")
        return []

    # 2. Calculate performance
    performance = {}
    for ticker, close_prices in all_historical_data.items():
        price_now = close_prices.iloc[-1]
        price_then = close_prices.iloc[-1 - LOOKBACK_PERIOD_DAYS]
        performance[ticker] = ((price_now - price_then) / price_then) * 100

    benchmark_performance = performance.pop(BENCHMARK_INDEX)
    log.info(f"Benchmark ({BENCHMARK_INDEX}) {LOOKBACK_PERIOD_DAYS}-day performance: {benchmark_performance:.2f}%")

    # 3. Filter for outperforming sectors
    outperforming_sectors = {
        ticker: perf for ticker, perf in performance.items() if perf > benchmark_performance
    }

    # 4. Sort and select the top N
    sorted_sectors = sorted(outperforming_sectors.items(), key=lambda item: item[1], reverse=True)
    
    # Get the original names of the top tickers
    top_sector_names = []
    for ticker, perf in sorted_sectors[:TOP_N_SECTORS]:
        # Find the human-readable name from our CORE_SECTOR_INDICES dictionary
        sector_name = next((name for name, t in CORE_SECTOR_INDICES.items() if t == ticker), ticker)
        top_sector_names.append(sector_name)
        log.info(f"  -> Strong Sector: {sector_name} (+{perf:.2f}%)")

    if not top_sector_names:
        log.warning("No sectors were found to be outperforming the market benchmark.")

    log.info(f"✅ Sector analysis complete. Identified {len(top_sector_names)} strong sector(s).")
    return top_sector_names