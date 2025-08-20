# quantitative_screener.py
import pandas as pd
import config
import data_retriever
from logger_config import log
import pandas_ta as ta
import yfinance as yf # We need yfinance for the sector info

# This dictionary maps the Sector Index names from our sector_analyzer
# to the corresponding sector names provided by Yahoo Finance. This is crucial for matching.
SECTOR_NAME_MAPPING = {
    "NIFTY Bank": "Financial Services",
    "NIFTY IT": "Technology",
    "NIFTY Auto": "Consumer Cyclical",  # Corrected from "Automobile and Auto Components"
    "NIFTY Pharma": "Healthcare",
    "NIFTY FMCG": "Consumer Defensive", # Corrected from "Fast Moving Consumer Goods"
    "NIFTY Metal": "Basic Materials",   # Corrected from "Metals & Mining"
    "NIFTY PSU Bank": "Financial Services",
    "NIFTY Oil & Gas": "Energy",
    "NIFTY India Consumption": "Consumer Defensive", # Corrected from "Fast Moving Consumer Goods"
}

# --- Screener Technical Parameters ---
MIN_AVG_VOLUME = 500000  # Minimum 20-day average volume
TREND_CHECK_EMA = 50       # We'll check if the price is above the 50-day EMA

def _get_stock_sector_map(tickers: list[str]) -> dict:
    """
    Builds a dictionary mapping each stock ticker to its sector.
    Leverages the sector cache in data_retriever for efficiency.
    """
    log.info("Building stock-to-sector map...")
    sector_cache = data_retriever.load_sector_cache()
    stock_sector_map = {}
    tickers_to_fetch = []

    for ticker in tickers:
        if ticker in sector_cache and sector_cache[ticker] != 'Other':
            stock_sector_map[ticker] = sector_cache[ticker]
        else:
            tickers_to_fetch.append(ticker)

    if tickers_to_fetch:
        log.info(f"Fetching sector info for {len(tickers_to_fetch)} new tickers...")
        for i, ticker in enumerate(tickers_to_fetch):
            try:
                info = yf.Ticker(ticker).info
                sector = info.get('sector', 'Other')
                stock_sector_map[ticker] = sector
                sector_cache[ticker] = sector
                log.info(f"({i+1}/{len(tickers_to_fetch)}) Fetched sector for {ticker}: {sector}")
            except Exception as e:
                log.warning(f"Could not fetch info for {ticker}: {e}")
                stock_sector_map[ticker] = 'Other'
        data_retriever.save_sector_cache(sector_cache)
    
    log.info("✅ Stock-to-sector map complete.")
    return stock_sector_map

def generate_dynamic_watchlist(strong_sectors: list[str]) -> list[str]:
    """
    The main screening function. Filters the Nifty 50 universe down to a small
    watchlist of high-potential stocks.
    """
    log.info("--- [Funnel Step 3] Running Quantitative Pre-Screener ---")
    
    # 1. Get the list of strong sector names as defined by yfinance
    target_sectors = {SECTOR_NAME_MAPPING[name] for name in strong_sectors if name in SECTOR_NAME_MAPPING}
    if not target_sectors:
        log.warning("No target sectors found after mapping. Aborting screen.")
        return []
    log.info(f"Screening for stocks in these strong sectors: {target_sectors}")

    # 2. Get the full stock universe and their sectors
    universe = config.NIFTY_50_TICKERS
    stock_sector_map = _get_stock_sector_map(universe)

    # 3. First Filter: Sector Strength
    sector_filtered_stocks = [
        ticker for ticker, sector in stock_sector_map.items() if sector in target_sectors
    ]
    log.info(f"Initial Universe: {len(universe)} stocks -> Sector Filter: {len(sector_filtered_stocks)} stocks")
    if not sector_filtered_stocks:
        return []

    # 4. Second Filter: Technical Health (Volume and Trend)
    final_watchlist = []
    for i, ticker in enumerate(sector_filtered_stocks):
        log.info(f"  -> Applying technical screen for {ticker} ({i+1}/{len(sector_filtered_stocks)})...")
        data = data_retriever.get_historical_stock_data(ticker)
        if data is None or len(data) < TREND_CHECK_EMA + 20:
            log.warning(f"    - Skipping {ticker}: Insufficient data.")
            continue

        # Volume Check
        avg_volume = data['volume'].tail(20).mean()
        if avg_volume < MIN_AVG_VOLUME:
            log.warning(f"    - Skipping {ticker}: Fails volume check (Avg Vol: {int(avg_volume)})")
            continue

        # Trend Check
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