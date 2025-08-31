# sector_analyzer.py
import pandas as pd
from logger_config import log

# Curated & reliable list of sector indices
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
    "NIFTY Realty": "^CNXREALTY",
    "NIFTY Media": "^CNXMEDIA",
    "NIFTY Private Bank": "NIFTY_PRIVATEBANK.NS", # Corrected
    "NIFTY Financial Services": "NIFTY_FIN_SERVICE.NS", # Corrected
    "NIFTY Healthcare": "NIFTY_HEALTHCARE.NS", # Corrected
    "NIFTY Services Sector": "^CNXSERVICE",
    "NIFTY Commodities": "NIFTY_COMMODITIES.NS", # Corrected
    "NIFTY Infrastructure": "^CNXINFRA",
    "NIFTY MNC": "^CNXMNC",
    # "NIFTY Energy": "^CNXENERGY", # Removed as it's a duplicate of Oil & Gas
    "NIFTY CPSE": "NIFTY_CPSE.NS", # Corrected
    "NIFTY PSE": "^CNXPSE",
    "NIFTY100 ESG": "NIFTY100_ESG.NS", # Corrected
}

BENCHMARK_INDEX = "^NSEI"
LOOKBACK_PERIOD_DAYS = 21
TOP_N_SECTORS = 5

def get_top_performing_sectors(full_data_cache: dict, point_in_time: pd.Timestamp) -> list[str]:
    """
    Calculates the performance of core sectors relative to the Nifty 50 benchmark
    over a defined lookback period using ONLY point-in-time data.

    Args:
        full_data_cache (dict): Dict of ticker -> historical DataFrame
        point_in_time (pd.Timestamp): The simulation day to slice up to
    """
    log.info("--- [Funnel Step 2] Analyzing Sector Relative Strength (Bias-Free Mode) ---")
    
    all_tickers = list(CORE_SECTOR_INDICES.values()) + [BENCHMARK_INDEX]
    performance = {}

    for ticker in all_tickers:
        if ticker not in full_data_cache:
            log.warning(f"Data for index {ticker} not found in cache. Skipping it for sector analysis.")
            continue
        
        # Slice to ensure only point-in-time data is considered
        data_slice = full_data_cache[ticker].loc[:point_in_time]

        if len(data_slice) < LOOKBACK_PERIOD_DAYS + 5:
            log.warning(f"Not enough point-in-time data for {ticker} on {point_in_time.strftime('%Y-%m-%d')}. Skipping.")
            continue

        price_now = data_slice['close'].iloc[-1]
        price_then = data_slice['close'].iloc[-1 - LOOKBACK_PERIOD_DAYS]
        performance[ticker] = ((price_now - price_then) / price_then) * 100 if price_then != 0 else 0

    if BENCHMARK_INDEX not in performance:
        log.error(f"Could not calculate performance for benchmark index {BENCHMARK_INDEX}. Cannot perform sector analysis.")
        return []

    benchmark_performance = performance.pop(BENCHMARK_INDEX, 0)
    log.info(f"Benchmark ({BENCHMARK_INDEX}) {LOOKBACK_PERIOD_DAYS}-day performance: {benchmark_performance:.2f}%")

    log.info(f"Ranking all sectors by performance (Top {TOP_N_SECTORS} will be selected)...")
    sorted_sectors = sorted(performance.items(), key=lambda item: item[1], reverse=True)
    
    top_sector_names = []
    for ticker, perf in sorted_sectors[:TOP_N_SECTORS]:
        sector_name = next((name for name, t in CORE_SECTOR_INDICES.items() if t == ticker), ticker)
        top_sector_names.append(sector_name)
        log.info(f"  -> Strong Sector Found: {sector_name} ({perf:+.2f}%)")

    if not top_sector_names:
        log.warning("No sectors were found to be outperforming the market benchmark.")

    log.info(f"✅ Sector analysis complete. Identified {len(top_sector_names)} strong sector(s).")
    return top_sector_names
