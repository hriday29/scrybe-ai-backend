# quantitative_screener.py (FINAL, CORRECTED VERSION 3.0)
import pandas as pd
import config
import data_retriever
from logger_config import log
import pandas_ta as ta
import yfinance as yf

SECTOR_NAME_MAPPING = {
    "NIFTY Bank": "Financial Services",
    "NIFTY IT": "Technology",
    "NIFTY Auto": "Consumer Cyclical",
    "NIFTY Pharma": "Healthcare",
    "NIFTY FMCG": "Consumer Defensive",
    "NIFTY Metal": "Basic Materials",
    "NIFTY PSU Bank": "Financial Services",
    "NIFTY Oil & Gas": "Energy",
    "NIFTY India Consumption": "Consumer Defensive",
}

MIN_AVG_VOLUME = 500000
TREND_CHECK_EMA = 50

def _passes_fundamental_health_check(ticker: str) -> bool:
    """
    Performs a basic check on key fundamental metrics using yfinance.
    Returns True if the stock is considered healthy, False otherwise.
    """
    try:
        info = yf.Ticker(ticker).info
        
        pe_ratio = info.get('trailingPE')
        debt_to_equity = info.get('debtToEquity')
        return_on_equity = info.get('returnOnEquity')

        # Rule 1: Must be profitable (positive P/E) and not absurdly valued
        if pe_ratio is None or pe_ratio < 0 or pe_ratio > 100:
            log.warning(f"    - Skipping {ticker}: Fails P/E check (P/E: {pe_ratio}).")
            return False
        
        # Rule 2: Debt must be manageable (for non-financials, typically < 1.5 or 150)
        # We use a high threshold to only filter out extreme cases.
        if debt_to_equity is not None and debt_to_equity > 200:
             log.warning(f"    - Skipping {ticker}: Fails Debt/Equity check (D/E: {debt_to_equity}).")
             return False

        # Rule 3: Must be generating good returns for shareholders
        if return_on_equity is not None and return_on_equity < 0.10: # (ROE < 10%)
            log.warning(f"    - Skipping {ticker}: Fails ROE check (ROE: {return_on_equity:.2f}).")
            return False
            
        return True
    except Exception as e:
        log.warning(f"    - Could not perform fundamental check for {ticker}. Error: {e}. Skipping.")
        return False
    
def _get_stock_sector_map(tickers: list[str]) -> dict:
    """
    Builds a dictionary mapping each stock ticker to its sector.
    Leverages the sector cache in data_retriever for efficiency.
    """
    log.info("Building stock-to-sector map...")
    sector_cache = data_retriever.load_sector_cache()
    stock_sector_map = {}
    tickers_to_fetch = [ticker for ticker in tickers if ticker not in sector_cache or sector_cache[ticker] == 'Other']

    if tickers_to_fetch:
        log.info(f"Fetching sector info for {len(tickers_to_fetch)} new or uncategorized tickers...")
        for i, ticker in enumerate(tickers_to_fetch):
            try:
                info = yf.Ticker(ticker).info
                sector = info.get('sector', 'Other')
                stock_sector_map[ticker] = sector
                sector_cache[ticker] = sector
                log.info(f"({i+1}/{len(tickers_to_fetch)}) Fetched sector for {ticker}: {sector}")
            except Exception:
                stock_sector_map[ticker] = 'Other'
        data_retriever.save_sector_cache(sector_cache)
    
    # Populate map from cache for tickers that were not re-fetched
    for ticker in tickers:
        if ticker not in stock_sector_map:
            stock_sector_map[ticker] = sector_cache.get(ticker, 'Other')

    log.info("✅ Stock-to-sector map complete.")
    return stock_sector_map

def generate_dynamic_watchlist(strong_sectors: list[str], full_data_cache: dict, point_in_time: str) -> list[str]:
    """
    V2 Screener: Upgraded with stricter trend and new momentum filters.
    """
    log.info("--- [Funnel Step 3] Running V2 Quantitative Pre-Screener ---")
    
    target_sectors = {SECTOR_NAME_MAPPING[name] for name in strong_sectors if name in SECTOR_NAME_MAPPING}
    if not target_sectors:
        log.warning("No target sectors found after mapping. Aborting screen.")
        return []
    log.info(f"Screening for stocks in these strong sectors: {target_sectors}")

    universe = list(full_data_cache.keys())
    stock_sector_map = _get_stock_sector_map(universe)

    sector_filtered_stocks = [
        ticker for ticker, sector in stock_sector_map.items() if sector in target_sectors
    ]
    log.info(f"Initial Universe: {len(universe)} stocks -> Sector Filter: {len(sector_filtered_stocks)} stocks")
    if not sector_filtered_stocks: return []

    final_watchlist = []
    for i, ticker in enumerate(sector_filtered_stocks):
        log.info(f"  -> Applying V2 technical screen for {ticker} ({i+1}/{len(sector_filtered_stocks)})...")
        
        # --- DEBUGGING: Use the full data cache first, then slice ---
        full_historical_data = full_data_cache.get(ticker)
        if full_historical_data is None or len(full_historical_data) < TREND_CHECK_EMA + 20:
            log.warning(f"    - Skipping {ticker}: Insufficient full historical data.")
            continue
        
        # Correctly slice the data to the point_in_time for the backtest
        data = full_historical_data.loc[:point_in_time].copy()
        if data.empty or len(data) < TREND_CHECK_EMA + 20:
             log.warning(f"    - Skipping {ticker}: Insufficient point-in-time data after slicing.")
             continue

        avg_volume = data['volume'].tail(20).mean()
        if avg_volume < MIN_AVG_VOLUME:
            log.warning(f"    - Skipping {ticker}: Fails volume check (Avg Vol: {int(avg_volume)})")
            continue

        # Calculate indicators on the correct point-in-time slice
        data.ta.ema(length=50, append=True)
        data.ta.rsi(length=14, append=True)
        data.dropna(inplace=True)

        latest_close = data['close'].iloc[-1]
        ema_50 = data['EMA_50'].iloc[-1]
        rsi_14 = data['RSI_14'].iloc[-1]

        # --- NEW DEBUG LOGGING ---
        log.info(f"    - DEBUG DATA for {ticker}: Close={latest_close:.2f}, 50-EMA={ema_50:.2f}, RSI={rsi_14:.2f}")

        # --- Our V3.0 Filters (More Flexible) ---
        proximity_threshold = 0.02  # Allow price to be within 2% of the EMA
        is_in_uptrend_or_testing_support = latest_close >= (ema_50 * (1 - proximity_threshold))

        if not is_in_uptrend_or_testing_support:
            log.warning(f"    - Skipping {ticker}: Fails trend check (Price {latest_close:.2f} is too far below 50-EMA {ema_50:.2f}).")
            continue
        
        if rsi_14 < 45:
             log.warning(f"    - Skipping {ticker}: Fails momentum check.")
             continue

        log.info(f"    - ✅ PASS: {ticker} passed all technical checks.")
        final_watchlist.append(ticker)

    log.info(f"✅ V2 Pre-screening complete. Final dynamic watchlist contains {len(final_watchlist)} stocks.")
    if final_watchlist: log.info(f"Final Watchlist: {final_watchlist}")
    return final_watchlist
