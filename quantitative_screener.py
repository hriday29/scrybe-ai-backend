# quantitative_screener.py (v4.2 - Regime-Adaptive Funnel with 3 Modules)
import pandas as pd
import config
import data_retriever
from logger_config import log
import pandas_ta as ta
import yfinance as yf

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
        if debt_to_equity is not None and debt_to_equity > 200:
            log.warning(f"    - Skipping {ticker}: Fails Debt/Equity check (D/E: {debt_to_equity}).")
            return False

        # Rule 3: Must be generating good returns for shareholders
        if return_on_equity is not None and return_on_equity < 0.10:  # (ROE < 10%)
            log.warning(f"    - Skipping {ticker}: Fails ROE check (ROE: {return_on_equity:.2f}).")
            return False

        # Rule 4: Fundamental quality proxy
        hist = yf.Ticker(ticker).history(period="1y")
        if hist is not None and not hist.empty:
            proxies = data_retriever.get_fundamental_proxies(hist)
            if proxies and proxies.get("quality_score", 50) < 60:
                log.warning(f"    - Skipping {ticker}: Poor quality score ({proxies['quality_score']})")
                return False

        return True

    except Exception as e:
        log.warning(f"    - Could not perform fundamental check for {ticker}. Error: {e}. Skipping.")
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


# --- Core Screener Utility ---
def _prepare_filtered_universe(strong_sectors: list[str], full_data_cache: dict, point_in_time: pd.Timestamp) -> list[str]:
    target_sectors = {SECTOR_NAME_MAPPING[name] for name in strong_sectors if name in SECTOR_NAME_MAPPING}
    if not target_sectors:
        log.warning("Screener Funnel: No target sectors found after mapping. Returning empty universe.")
        return []

    universe = list(full_data_cache.keys())
    stock_sector_map = _get_stock_sector_map(universe)

    sector_filtered = [t for t, s in stock_sector_map.items() if s in target_sectors]
    log.info(f"Screener Funnel | Sector Filter Result: {len(universe)} -> {len(sector_filtered)} stocks")

    qualified = []
    for ticker in sector_filtered:
        data = full_data_cache.get(ticker)
        if data is None or len(data) < TREND_CHECK_EMA + 20:
            continue
        df = data.loc[:point_in_time].copy()
        if df.empty or len(df) < TREND_CHECK_EMA + 20:
            continue
        if df['volume'].tail(20).mean() < MIN_AVG_VOLUME:
            continue
        if not _passes_fundamental_health_check(ticker):
            continue
        qualified.append(ticker)

    log.info(f"Screener Funnel | Pre-flight checks passed: {len(qualified)} stocks")
    return qualified


# --- Strategy Modules ---
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

        latest_close = df['close'].iloc[-1]
        ema50 = df['EMA_50'].iloc[-1]
        ema200 = df['EMA_200'].iloc[-1]
        rsi = df['RSI_14'].iloc[-1]

        if latest_close > ema50 > ema200 and rsi < 65:
            watchlist.append((ticker, "Pullback"))
            log.info(f"  -> ✅ Pullback Candidate: {ticker}")

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

        latest_close = df['close'].iloc[-1]
        ema50 = df['EMA_50'].iloc[-1]
        rsi = df['RSI_14'].iloc[-1]
        adx = df['ADX_14'].iloc[-1]

        if latest_close > ema50 and 55 < rsi < 70 and adx > config.ADX_THRESHOLD:
            watchlist.append((ticker, "Momentum"))
            log.info(f"  -> ✅ Momentum Candidate: {ticker}")

    log.info(f"✅ Momentum Screener Result: {len(watchlist)} candidates")
    return watchlist


def screen_for_mean_reversion(strong_sectors: list[str], full_data_cache: dict, point_in_time: pd.Timestamp) -> list[tuple[str, str]]:
    """Deep oversold setups in long-term uptrend."""
    log.info("--- Screening for Mean Reversion Setups ---")
    tickers = _prepare_filtered_universe(strong_sectors, full_data_cache, point_in_time)
    watchlist = []

    for ticker in tickers:
        df = full_data_cache[ticker].loc[:point_in_time].copy()
        df.ta.ema(length=200, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.adx(length=14, append=True)
        df.dropna(inplace=True)

        latest_close = df['close'].iloc[-1]
        ema200 = df['EMA_200'].iloc[-1]
        rsi = df['RSI_14'].iloc[-1]
        adx = df['ADX_14'].iloc[-1]

        if latest_close > ema200 and rsi < 35 and adx < 22:
            watchlist.append((ticker, "Mean Reversion"))
            log.info(f"  -> ✅ Mean Reversion Candidate: {ticker}")

    log.info(f"✅ Mean Reversion Screener Result: {len(watchlist)} candidates")
    return watchlist
