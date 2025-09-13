# quantitative_screener.py (MODIFIED to support single-stock checks)
import pandas as pd
import config
import data_retriever
from logger_config import log
import pandas_ta as ta
import yfinance as yf
import os

# --- Sector Mapping ---
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


# --- Fundamental Health Check ---
def _passes_fundamental_health_check(ticker: str, sector: str, company_info: dict = None) -> bool:
    """
    Performs a robust, multi-factor check on key fundamental metrics.
    It calculates a weighted Quality Score based on profitability, efficiency, and institutional trust.
    """
    try:
        info = company_info if company_info is not None else yf.Ticker(ticker).info
        
        # --- Hard Filters ---
        pe_ratio = info.get('trailingPE')
        if pe_ratio is None or pe_ratio < 0 or pe_ratio > 150:
            return False
        
        debt_to_equity = info.get('debtToEquity')
        if sector != "Financial Services" and debt_to_equity is not None and debt_to_equity > 200:
            return False

        # --- Multi-Factor Quality Score Calculation ---
        quality_score = 0
        
        return_on_equity = info.get('returnOnEquity')
        if return_on_equity is not None and return_on_equity > 0.15:
            quality_score += 40

        profit_margins = info.get('profitMargins')
        if profit_margins is not None and profit_margins > 0.10:
            quality_score += 30

        held_percent_institutions = info.get('heldPercentInstitutions')
        if held_percent_institutions is not None and held_percent_institutions > 0.30:
            quality_score += 30
            
        if quality_score < 40:
            return False

        return True

    except Exception as e:
        log.warning(f"Could not perform fundamental check for {ticker}. Error: {e}. Skipping.")
        return False

# --- Sector Map ---
def _get_stock_sector_map(tickers: list[str]) -> dict:
    log.info("Building stock-to-sector map...")
    sector_cache = data_retriever.load_sector_cache()
    stock_sector_map = {}
    tickers_to_fetch = [t for t in tickers if t not in sector_cache or sector_cache[t] == 'Other']

    if tickers_to_fetch:
        log.info(f"Fetching sector info for {len(tickers_to_fetch)} new/uncategorized tickers...")
        for i, ticker in enumerate(tickers_to_fetch):
            try:
                info = yf.Ticker(ticker).info
                sector = info.get('sector', 'Other')
                stock_sector_map[ticker] = sector
                sector_cache[ticker] = sector
                log.info(f"({i+1}/{len(tickers_to_fetch)}) {ticker}: {sector}")
            except Exception:
                stock_sector_map[ticker] = 'Other'
        data_retriever.save_sector_cache(sector_cache)

    for ticker in tickers:
        if ticker not in stock_sector_map:
            stock_sector_map[ticker] = sector_cache.get(ticker, 'Other')

    log.info("✅ Stock-to-sector map complete.")
    return stock_sector_map


# =================================================================================
# --- NEW SINGLE-STOCK CHECKER FUNCTIONS for the Unified Analysis Plan ---
# =================================================================================

def _passes_preflight_checks_single(ticker: str, data: pd.DataFrame, stock_sector: str, strong_sectors: list[str], point_in_time: pd.Timestamp, company_info: dict = None) -> bool:
    """Performs all basic checks (sector, volume, data length, fundamentals) for a single stock."""
    # 1. Sector Check
    target_sectors = {SECTOR_NAME_MAPPING[name] for name in strong_sectors if name in SECTOR_NAME_MAPPING}
    if stock_sector not in target_sectors:
        return False
        
    # 2. Data Integrity and Volume Check
    if data is None or len(data) < 252: return False
    df = data.loc[:point_in_time]
    if df.empty or len(df) < 252: return False
    if df['volume'].tail(20).mean() < MIN_AVG_VOLUME: return False
    
    # 3. Fundamental Health Check (this is the slowest part, so it's last)
    if not _passes_fundamental_health_check(ticker, stock_sector, company_info=company_info):
        return False
        
    return True

def _check_pullback_rules(df: pd.DataFrame) -> bool:
    """Contains the specific technical rules for a Pullback."""
    latest = df.iloc[-1]
    return latest['close'] > latest['EMA_50'] > latest['EMA_200'] and latest['RSI_14'] < 65

def _check_momentum_rules(df: pd.DataFrame) -> bool:
    """Contains the specific technical rules for Momentum."""
    latest = df.iloc[-1]
    return latest['close'] > latest['EMA_50'] and latest['RSI_14'] > 60 and latest['ADX_14'] > config.ADX_THRESHOLD

def _check_mean_reversion_rules(df: pd.DataFrame) -> bool:
    """Contains the specific technical rules for Mean Reversion."""
    latest = df.iloc[-1]
    return latest['close'] > latest['EMA_200'] and latest['EMA_50'] > latest['EMA_200'] and latest['RSI_14'] < 40 and latest['ADX_14'] < 25

def check_strategy_candidate(
    ticker: str, data: pd.DataFrame, stock_sector: str, strong_sectors: list[str],
    market_regime: str, volatility_regime: str, point_in_time: pd.Timestamp,
    company_info: dict = None
) -> str | None:
    """
    Main dispatcher function for the unified daily job. Checks a single stock against the
    appropriate strategy based on the market regime and returns the reason if it's a candidate.
    """
    # Step 1: Run all pre-flight checks. If it fails any, it's not a candidate.
    if not _passes_preflight_checks_single(ticker, data, stock_sector, strong_sectors, point_in_time, company_info=company_info):
        return None

    # Step 2: Calculate necessary indicators
    df = data.loc[:point_in_time].copy()
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.dropna(inplace=True)
    if df.empty: return None

    # Step 3: Apply the correct technical rules based on the market regime
    if market_regime == "Bearish":
        if _check_mean_reversion_rules(df): return "Mean Reversion"
    elif market_regime == "Bullish":
        if volatility_regime == "High-Risk":
            if _check_pullback_rules(df): return "Pullback"
        else: # Low or Normal Volatility
            if _check_momentum_rules(df): return "Momentum"
    elif market_regime == "Neutral":
        if _check_mean_reversion_rules(df): return "Mean Reversion"
            
    # If none of the conditions were met, it's not a candidate
    return None


# =================================================================================
# --- ORIGINAL BATCH SCREENER FUNCTIONS (PRESERVED FOR BACKTESTER) ---
# =================================================================================

def _prepare_filtered_universe(strong_sectors: list[str], full_data_cache: dict, point_in_time: pd.Timestamp) -> list[str]:
    target_sectors = {SECTOR_NAME_MAPPING[name] for name in strong_sectors if name in SECTOR_NAME_MAPPING}
    if not target_sectors:
        log.warning("Screener Funnel: No target sectors found after mapping. Returning empty universe.")
        return []

    universe = [t for t in full_data_cache.keys() if ".NS" in t] # Filter out indices
    stock_sector_map = _get_stock_sector_map(universe)

    sector_filtered = [t for t, s in stock_sector_map.items() if s in target_sectors]
    log.info(f"Screener Funnel | Sector Filter Result: {len(universe)} -> {len(sector_filtered)} stocks")

    qualified = []
    for ticker in sector_filtered:
        data = full_data_cache.get(ticker)
        if data is None or len(data) < 252: continue
        df = data.loc[:point_in_time]
        if df.empty or len(df) < 252: continue
        if df['volume'].tail(20).mean() < MIN_AVG_VOLUME: continue
        if not _passes_fundamental_health_check(ticker, stock_sector_map.get(ticker)): continue
        qualified.append(ticker)

    log.info(f"Screener Funnel | Pre-flight checks passed: {len(qualified)} stocks")
    return qualified

def screen_for_pullbacks(strong_sectors: list[str], full_data_cache: dict, point_in_time: pd.Timestamp) -> list[tuple[str, str]]:
    """Healthy short-term pullback in strong trend."""
    log.info("--- Screening for Pullback Setups ---")
    tickers = _prepare_filtered_universe(strong_sectors, full_data_cache, point_in_time)
    watchlist = []
    for ticker in tickers:
        df = full_data_cache[ticker].loc[:point_in_time].copy()
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)
        df.ta.rsi(length=14, append=True)
        df.dropna(inplace=True)
        if df.empty: continue
        if _check_pullback_rules(df):
            watchlist.append((ticker, "Pullback"))
            log.info(f" 	-> ✅ Pullback Candidate: {ticker}")
    log.info(f"✅ Pullback Screener Result: {len(watchlist)} candidates")
    return watchlist

def screen_for_momentum(strong_sectors: list[str], full_data_cache: dict, point_in_time: pd.Timestamp) -> list[tuple[str, str]]:
    """Strong, trending stocks not yet overextended."""
    log.info("--- Screening for Momentum Setups ---")
    tickers = _prepare_filtered_universe(strong_sectors, full_data_cache, point_in_time)
    watchlist = []
    for ticker in tickers:
        df = full_data_cache[ticker].loc[:point_in_time].copy()
        df.ta.ema(length=50, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.adx(length=14, append=True)
        df.dropna(inplace=True)
        if df.empty: continue
        if _check_momentum_rules(df):
            watchlist.append((ticker, "Momentum"))
            log.info(f" 	-> ✅ Momentum Candidate: {ticker}")
    log.info(f"✅ Momentum Screener Result: {len(watchlist)} candidates")
    return watchlist

def screen_for_mean_reversion(strong_sectors: list[str], full_data_cache: dict, point_in_time: pd.Timestamp) -> list[tuple[str, str]]:
    """Deep oversold setups in long-term uptrend."""
    log.info("--- Screening for Mean Reversion Setups ---")
    tickers = _prepare_filtered_universe(strong_sectors, full_data_cache, point_in_time)
    watchlist = []
    for ticker in tickers:
        df = full_data_cache[ticker].loc[:point_in_time].copy()
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.adx(length=14, append=True)
        df.dropna(inplace=True)
        if df.empty: continue
        if _check_mean_reversion_rules(df):
            watchlist.append((ticker, "Mean Reversion"))
            log.info(f" 	-> ✅ Mean Reversion Candidate: {ticker}")
    log.info(f"✅ Mean Reversion Screener Result: {len(watchlist)} candidates")
    return watchlist