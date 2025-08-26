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
        log.warning("Screener Funnel: No target sectors found after mapping. Watchlist will be empty.")
        return []
    log.info(f"Screener Funnel: Screening for stocks in these strong sectors: {target_sectors}")

    universe = list(full_data_cache.keys())
    stock_sector_map = _get_stock_sector_map(universe)

    sector_filtered_stocks = [
        ticker for ticker, sector in stock_sector_map.items() if sector in target_sectors
    ]
    # --- NEW SUMMARY LOG (Gemini style) ---
    log.info(f"Screener Funnel | Sector Filter Result: {len(universe)} stocks -> {len(sector_filtered_stocks)} stocks")
    if not sector_filtered_stocks: 
        return []

    final_watchlist = []
    for i, ticker in enumerate(sector_filtered_stocks):
        log.info(f"  -> Applying V2 technical screen for {ticker} ({i+1}/{len(sector_filtered_stocks)})...")
        
        # Use the full data cache first, then slice
        full_historical_data = full_data_cache.get(ticker)
        if full_historical_data is None or len(full_historical_data) < TREND_CHECK_EMA + 20:
            log.warning(f"    - VETO (Data): Skipping {ticker} due to insufficient full historical data.")
            continue
        
        # Slice correctly to the point_in_time
        data = full_historical_data.loc[:point_in_time].copy()
        if data.empty or len(data) < TREND_CHECK_EMA + 20:
            log.warning(f"    - VETO (Data): Skipping {ticker} due to insufficient point-in-time data.")
            continue

        avg_volume = data['volume'].tail(20).mean()
        if avg_volume < MIN_AVG_VOLUME:
            log.warning(f"    - VETO (Volume): Skipping {ticker} (Avg Vol: {int(avg_volume)} < {MIN_AVG_VOLUME})")
            continue

        # Indicators
        data.ta.ema(length=50, append=True)
        data.ta.rsi(length=14, append=True)
        data.dropna(inplace=True)

        latest_close = data['close'].iloc[-1]
        ema_50 = data['EMA_50'].iloc[-1]
        rsi_14 = data['RSI_14'].iloc[-1]

        log.info(f"    - DEBUG DATA for {ticker}: Close={latest_close:.2f}, 50-EMA={ema_50:.2f}, RSI={rsi_14:.2f}")

        # --- Filters ---
        proximity_threshold = 0.04  # Stock must be within 4% of 50-EMA

        # Condition A: "Stable Trend-Follower"
        is_stable_trend = (latest_close >= (ema_50 * (1 - proximity_threshold))) and (rsi_14 >= 40)
        
        # Condition B: "Strong Momentum Breakout/Pullback"
        is_strong_momentum = rsi_14 > 65

        if is_stable_trend or is_strong_momentum:
            log.info(f"    - ✅ PASS: {ticker} passed all technical checks (Stable Trend: {is_stable_trend}, Strong Momentum: {is_strong_momentum}).")
            final_watchlist.append(ticker)
        else:
            log.warning(f"    - VETO (Trend/Momentum): Skipping {ticker} (Stable Trend={is_stable_trend}, Strong Momentum={is_strong_momentum})")
            continue

    # --- NEW FINAL SUMMARY LOG (Gemini style) ---
    log.info(f"✅ Screener Funnel | Final Result: {len(final_watchlist)} stocks passed all technical checks.")
    if final_watchlist: 
        log.info(f"Final Watchlist: {final_watchlist}")
    return final_watchlist