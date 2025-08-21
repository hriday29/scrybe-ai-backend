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
    The main screening function. Filters the universe down to a small
    watchlist of high-potential stocks using PRE-LOADED, POINT-IN-TIME data.
    """
    log.info("--- [Funnel Step 3] Running Quantitative Pre-Screener ---")
    
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
    if not sector_filtered_stocks:
        return []

    final_watchlist = []
    for i, ticker in enumerate(sector_filtered_stocks):
        log.info(f"  -> Applying technical screen for {ticker} ({i+1}/{len(sector_filtered_stocks)})...")
        
        data = full_data_cache.get(ticker).loc[:point_in_time].copy()
        if data is None or len(data) < TREND_CHECK_EMA + 20:
            log.warning(f"    - Skipping {ticker}: Insufficient point-in-time data in cache.")
            continue

        avg_volume = data['volume'].tail(20).mean()
        if avg_volume < MIN_AVG_VOLUME:
            log.warning(f"    - Skipping {ticker}: Fails volume check (Avg Vol: {int(avg_volume)})")
            continue

        data.ta.ema(length=TREND_CHECK_EMA, append=True)
        latest_close = data['close'].iloc[-1]
        ema_value = data[f'EMA_{TREND_CHECK_EMA}'].iloc[-1]

        if latest_close < ema_value:
            log.warning(f"    - Skipping {ticker}: Fails trend check (Price {latest_close:.2f} < {TREND_CHECK_EMA}-EMA {ema_value:.2f})")
            continue
        
        log.info(f"    - ✅ PASS: {ticker} passed all technical checks.")
        final_watchlist.append(ticker)

    log.info(f"✅ Pre-screening complete. Final dynamic watchlist contains {len(final_watchlist)} stocks.")
    log.info(f"Final Watchlist: {final_watchlist}")
    return final_watchlist