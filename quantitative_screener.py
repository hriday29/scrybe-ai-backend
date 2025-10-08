# quantitative_screener.py (FINAL CORRECTED VERSION)
import pandas as pd
import config
import data_retriever
from logger_config import log
import pandas_ta as ta
import yfinance as yf
import os
import database_manager

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
def _passes_fundamental_health_check(ticker: str, point_in_time: pd.Timestamp) -> bool:
    """
    Performs a robust, score-based fundamental health check.

    A stock is considered healthy if it passes a minimum number of fundamental
    criteria, as defined in FUNDAMENTAL_THRESHOLDS. This approach gracefully
    handles missing data (e.g., null ROE) by simply not awarding a point for that
    metric, rather than disqualifying the stock entirely.
    """
    try:
        point_in_time_utc = pd.to_datetime(point_in_time, utc=True).to_pydatetime()

        cursor = database_manager.db.fundamentals.find({
            "ticker": ticker,
            "asOfDate": {"$lte": point_in_time_utc}
        }).sort("asOfDate", -1).limit(1)

        results = list(cursor)
        if not results:
            log.warning(f"No point-in-time fundamentals found for {ticker} on or before {point_in_time.date()}.")
            return False
        
        fundamentals = results[0]
        score = 0
        reasons = []

        # 1. Return on Equity (ROE) Check
        roe = fundamentals.get("returnOnEquity")
        if roe is not None:
            if roe > FUNDAMENTAL_THRESHOLDS["MIN_ROE"]:
                score += 1
                reasons.append(f"ROE OK ({roe:.2%})")
            else:
                reasons.append(f"ROE Low ({roe:.2%})")
        else:
            reasons.append("ROE Missing")

        # 2. Profit Margins Check
        margins = fundamentals.get("profitMargins")
        if margins is not None:
            if margins > FUNDAMENTAL_THRESHOLDS["MIN_PROFIT_MARGIN"]:
                score += 1
                reasons.append(f"Margin OK ({margins:.2%})")
            else:
                reasons.append(f"Margin Low ({margins:.2%})")
        else:
            reasons.append("Margin Missing")
            
        # 3. Debt-to-Equity Check
        d2e = fundamentals.get("debtToEquity")
        if d2e is not None:
            # Note: yfinance provides D/E as a percentage, so 1.5 is 150.
            if d2e < (FUNDAMENTAL_THRESHOLDS["MAX_DEBT_TO_EQUITY"] * 100):
                score += 1
                reasons.append(f"D/E OK ({d2e:.1f})")
            else:
                reasons.append(f"D/E High ({d2e:.1f})")
        else:
            reasons.append("D/E Missing")
            
        # 4. Revenue Growth Check
        rev_growth = fundamentals.get("revenueGrowth")
        if rev_growth is not None:
            if rev_growth > FUNDAMENTAL_THRESHOLDS["MIN_REVENUE_GROWTH"]:
                score += 1
                reasons.append(f"RevGrowth OK ({rev_growth:.2%})")
            else:
                reasons.append(f"RevGrowth Low ({rev_growth:.2%})")
        else:
            reasons.append("RevGrowth Missing")

        # --- Final Decision ---
        is_healthy = score >= FUNDAMENTAL_THRESHOLDS["MIN_HEALTH_SCORE"]
        
        report_date = fundamentals['asOfDate'].date()
        log_func = log.info if is_healthy else log.warning
        status_icon = "✅" if is_healthy else "->"
        
        log_func(
            f"{status_icon} {ticker} passed fundamental check for {point_in_time.date()} "
            f"(Score: {score}/{FUNDAMENTAL_THRESHOLDS['MIN_HEALTH_SCORE']}) "
            f"| Reasons: [{', '.join(reasons)}] | Report Date: {report_date}"
        )
        
        return is_healthy
            
    except Exception as e:
        log.error(f"Error during point-in-time fundamental check for {ticker}: {e}. The stock will fail the check.")
        return False

# --- Sector Map ---
def _get_stock_sector_map(tickers: list[str]) -> dict:
    log.info("Building stock-to-sector map...")
    sector_cache = data_retriever.load_sector_cache()
    stock_sector_map = {}
    tickers_to_fetch = [t for t in tickers if t not in sector_cache or sector_cache.get(t) == 'Other']

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
# --- SINGLE-STOCK CHECKER FUNCTIONS (for the live run_daily_jobs.py) ---
# =================================================================================

def _passes_preflight_checks_single(ticker: str, data: pd.DataFrame, stock_sector: str, strong_sectors: list[str], point_in_time: pd.Timestamp) -> bool:
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
    
    # 3. Fundamental Health Check - **FIXED: Passing the data slice**
    if not _passes_fundamental_health_check(ticker, point_in_time):
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
    market_regime: str, volatility_regime: str, point_in_time: pd.Timestamp
) -> str | None:
    """
    Main dispatcher function for the unified daily job. Checks a single stock against the
    appropriate strategy based on the market regime and returns the reason if it's a candidate.
    """
    # Step 1: Run all pre-flight checks. **FIXED: Removed company_info**
    if not _passes_preflight_checks_single(ticker, data, stock_sector, strong_sectors, point_in_time):
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
            
    return None

# =================================================================================
# --- BATCH SCREENER FUNCTIONS (for the backtester main_orchestrator.py) ---
# =================================================================================

def _prepare_filtered_universe(strong_sectors: list[str], full_data_cache: dict, point_in_time: pd.Timestamp) -> list[str]:
    target_sectors = {SECTOR_NAME_MAPPING[name] for name in strong_sectors if name in SECTOR_NAME_MAPPING}
    if not target_sectors:
        log.warning("Screener Funnel: No target sectors found after mapping. Returning empty universe.")
        return []

    universe = [t for t in full_data_cache.keys() if ".NS" in t]
    stock_sector_map = _get_stock_sector_map(universe)

    sector_filtered = [t for t, s in stock_sector_map.items() if s in target_sectors]
    log.info(f"Screener Funnel | Sector Filter Result: {len(universe)} -> {len(sector_filtered)} stocks")

    qualified = []
    for ticker in sector_filtered:
        data = full_data_cache.get(ticker)
        if data is None or len(data) < 252: continue
        
        df_slice = data.loc[:point_in_time]
        if df_slice.empty or len(df_slice) < 252: continue
        if df_slice['volume'].tail(20).mean() < MIN_AVG_VOLUME: continue
        
        # **FIXED: Passing the correct data slice**
        if not _passes_fundamental_health_check(ticker, point_in_time): continue
        
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
            log.info(f"     -> ✅ Pullback Candidate: {ticker}")
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
            log.info(f"     -> ✅ Momentum Candidate: {ticker}")
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
            log.info(f"     -> ✅ Mean Reversion Candidate: {ticker}")
    log.info(f"✅ Mean Reversion Screener Result: {len(watchlist)} candidates")
    return watchlist

def get_analyzable_universe(full_data_cache: dict, point_in_time: pd.Timestamp) -> list[str]:
    """
    Performs basic pre-flight checks to filter for a universe of stocks that are
    healthy and liquid enough for AI analysis.
    """
    log.info("--- Filtering for a broad, analyzable universe ---")
    
    universe = [t for t in full_data_cache.keys() if ".NS" in t]
    qualified_tickers = []
    
    for ticker in universe:
        data = full_data_cache.get(ticker)
        if data is None: continue
        
        df_slice = data.loc[:point_in_time]
        if df_slice.empty or len(df_slice) < 252: continue
        if df_slice['volume'].tail(20).mean() < MIN_AVG_VOLUME: continue
        if not _passes_fundamental_health_check(ticker, point_in_time): continue
        
        qualified_tickers.append(ticker)

    log.info(f"✅ Found {len(qualified_tickers)} liquid and healthy stocks for AI analysis.")
    return qualified_tickers
