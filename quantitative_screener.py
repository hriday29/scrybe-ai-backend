# quantitative_screener.py (Correct, Restored Version)
import pandas as pd
import config
from logger_config import log
import pandas_ta as ta
import database_manager

# --- Constants ---
SECTOR_NAME_MAPPING = {
    "NIFTY Bank": "Financial Services", "NIFTY IT": "Technology",
    "NIFTY Auto": "Consumer Cyclical", "NIFTY Pharma": "Healthcare",
    "NIFTY FMCG": "Consumer Defensive", "NIFTY Metal": "Basic Materials",
    "NIFTY PSU Bank": "Financial Services", "NIFTY Oil & Gas": "Energy",
    "NIFTY India Consumption": "Consumer Defensive",
}
FUNDAMENTAL_THRESHOLDS = {
    "MIN_ROE": 0.15, "MIN_PROFIT_MARGIN": 0.10, "MAX_DEBT_TO_EQUITY": 2.0,
}
MIN_AVG_VOLUME = 500000

# ==============================================================================
# --- CORE HELPER FUNCTIONS ---
# ==============================================================================

def _passes_fundamental_health_check(ticker: str, point_in_time: pd.Timestamp) -> bool:
    """Performs a robust, adaptive, score-based fundamental health check."""
    try:
        point_in_time_utc = pd.to_datetime(point_in_time, utc=True).to_pydatetime()
        cursor = database_manager.db.fundamentals.find({
            "ticker": ticker, "asOfDate": {"$lte": point_in_time_utc}
        }).sort("asOfDate", -1).limit(1)
        results = list(cursor)
        if not results: return False
        
        fundamentals = results[0]
        score, available_metrics = 0, 0
        
        if fundamentals.get("returnOnEquity") is not None:
            available_metrics += 1
            if fundamentals["returnOnEquity"] > FUNDAMENTAL_THRESHOLDS["MIN_ROE"]: score += 1
        if fundamentals.get("profitMargins") is not None:
            available_metrics += 1
            if fundamentals["profitMargins"] > FUNDAMENTAL_THRESHOLDS["MIN_PROFIT_MARGIN"]: score += 1
        if fundamentals.get("debtToEquity") is not None:
            available_metrics += 1
            # yfinance provides D/E as a percentage, so 2.0 is 200.
            if fundamentals["debtToEquity"] < (FUNDAMENTAL_THRESHOLDS["MAX_DEBT_TO_EQUITY"] * 100): score += 1
        
        if available_metrics == 0: return False
        # Stock must pass at least half of the available checks
        required_score = available_metrics / 2.0
        return score >= required_score
    except Exception: return False

def _is_fundamentally_vulnerable(ticker: str, point_in_time: pd.Timestamp) -> bool:
    """
    Performs a score-based check to identify fundamentally WEAK companies for shorting.

    A stock is considered vulnerable if it fails a minimum number of fundamental
    criteria, indicating financial weakness.
    """
    try:
        point_in_time_utc = pd.to_datetime(point_in_time, utc=True).to_pydatetime()
        cursor = database_manager.db.fundamentals.find({
            "ticker": ticker, "asOfDate": {"$lte": point_in_time_utc}
        }).sort("asOfDate", -1).limit(1)
        results = list(cursor)
        if not results: return False
        
        fundamentals = results[0]
        failure_score, available_metrics = 0, 0
        
        # Check for LOW Return on Equity (a sign of inefficiency)
        if fundamentals.get("returnOnEquity") is not None:
            available_metrics += 1
            if fundamentals["returnOnEquity"] < 0.10: failure_score += 1
            
        # Check for LOW Profit Margins (a sign of weak pricing power or high costs)
        if fundamentals.get("profitMargins") is not None:
            available_metrics += 1
            if fundamentals["profitMargins"] < 0.05: failure_score += 1

        # Check for HIGH Debt-to-Equity (a sign of financial risk)
        if fundamentals.get("debtToEquity") is not None:
            available_metrics += 1
            if fundamentals["debtToEquity"] > (FUNDAMENTAL_THRESHOLDS["MAX_DEBT_TO_EQUITY"] * 100): failure_score += 1
        
        if available_metrics == 0: return False
        # If the company fails at least half of the available checks, it's vulnerable.
        required_failures = available_metrics / 2.0
        return failure_score >= required_failures
    except Exception: return False

def _prepare_filtered_universe(actionable_sectors: list[str], full_data_cache: dict, point_in_time: pd.Timestamp, market_state: dict) -> list[str]:
    """Pre-filters the universe by sector, liquidity, and fundamental health."""
    import data_retriever # Local import to avoid circular dependency issues
    target_sectors = {SECTOR_NAME_MAPPING[name] for name in actionable_sectors if name in SECTOR_NAME_MAPPING}
    if not target_sectors: return []

    universe = [t for t in full_data_cache.keys() if ".NS" in t]
    stock_sector_map = data_retriever.get_stock_sector_map(universe)
    
    sector_filtered = [t for t in universe if stock_sector_map.get(t) in target_sectors]
    
    qualified = []
    for ticker in sector_filtered:
        data = full_data_cache.get(ticker)
        if data is None: continue
        
        df_slice = data.loc[:point_in_time]
        if len(df_slice) < 252: continue
        if df_slice['volume'].tail(20).mean() < MIN_AVG_VOLUME: continue
        # Apply the correct fundamental filter based on the market regime.
        is_shorting_regime = market_state.get('market_regime', {}).get('regime_status') in ["Downtrend", "Bearish Rally"]

        if is_shorting_regime:
            # For shorting, we look for FUNDAMENTALLY VULNERABLE companies.
            if not _is_fundamentally_vulnerable(ticker, point_in_time): continue
        else:
            # For longs, we look for FUNDAMENTALLY HEALTHY companies.
            if not _passes_fundamental_health_check(ticker, point_in_time): continue
        qualified.append(ticker)
    return qualified

# ==============================================================================
# --- TECHNICAL RULESETS FOR THE STRATEGIC PLAYBOOK ---
# ==============================================================================

def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Helper to calculate all necessary indicators at once."""
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.dropna(inplace=True)
    return df

def _check_long_momentum_rules(df: pd.DataFrame) -> bool:
    """Rules for long momentum in an uptrend."""
    latest = df.iloc[-1]
    is_trending = latest['ADX_14'] > 22
    has_momentum = latest['RSI_14'] > 60
    return latest['close'] > latest['EMA_50'] and (is_trending or has_momentum)

def _check_short_breakdown_rules(df: pd.DataFrame) -> bool:
    """Rules for short breakdown in a downtrend."""
    latest = df.iloc[-1]
    is_trending = latest['ADX_14'] > 22
    has_momentum = latest['RSI_14'] < 40
    return latest['close'] < latest['EMA_50'] and (is_trending or has_momentum)

def _check_long_mean_reversion_rules(df: pd.DataFrame) -> bool:
    """Rule for 'buy the dip' setups in a bullish pullback."""
    latest = df.iloc[-1]
    # Primary trend is up, but stock is oversold and not in a strong short-term trend
    return latest['EMA_50'] > latest['EMA_200'] and latest['RSI_14'] < 40 and latest['ADX_14'] < 25

def _check_short_mean_reversion_rules(df: pd.DataFrame) -> bool:
    """Rule for 'short the rip' setups in a bearish rally."""
    latest = df.iloc[-1]
    # Primary trend is down, but stock is overbought and not in a strong short-term trend
    return latest['EMA_50'] < latest['EMA_200'] and latest['RSI_14'] > 60 and latest['ADX_14'] < 25

# ==============================================================================
# --- THE UNIFIED, PLAYBOOK-DRIVEN CANDIDATE GENERATOR ---
# ==============================================================================

def get_strategy_candidates(market_state: dict, full_data_cache: dict, point_in_time: pd.Timestamp) -> list[tuple[str, str]]:
    """
    The final master dispatcher. It runs a playbook of strategies based on the
    market regime to find high-probability trade candidates.
    """
    regime = market_state.get('market_regime', {}).get('regime_status', 'Sideways')
    actionable_sectors = market_state.get('actionable_sectors', [])
    candidates = []

    log.info(f"--- Running Strategic Playbook for Regime: {regime} ---")
    
    base_universe = _prepare_filtered_universe(actionable_sectors, full_data_cache, point_in_time, market_state)
    log.info(f"Found {len(base_universe)} healthy, liquid stocks in actionable sectors to screen.")

    # --- Execute Plays Based on Market Regime ---
    if regime == "Uptrend":
        log.info("Executing Play: Long Momentum")
        for ticker in base_universe:
            df = _add_indicators(full_data_cache[ticker].loc[:point_in_time].copy())
            if not df.empty and _check_long_momentum_rules(df):
                candidates.append((ticker, "Long Momentum"))

    elif regime == "Bullish Pullback":
        log.info("Executing Play: Long Mean Reversion (Buy the Dip)")
        for ticker in base_universe:
            df = _add_indicators(full_data_cache[ticker].loc[:point_in_time].copy())
            if not df.empty and _check_long_mean_reversion_rules(df):
                candidates.append((ticker, "Long Mean Reversion"))

    elif regime == "Downtrend":
        log.info("Executing Play: Short Breakdown")
        for ticker in base_universe:
            df = _add_indicators(full_data_cache[ticker].loc[:point_in_time].copy())
            if not df.empty and _check_short_breakdown_rules(df):
                candidates.append((ticker, "Short Breakdown"))

    elif regime == "Bearish Rally":
        log.info("Executing Play: Short Mean Reversion (Short the Rip)")
        for ticker in base_universe:
            df = _add_indicators(full_data_cache[ticker].loc[:point_in_time].copy())
            if not df.empty and _check_short_mean_reversion_rules(df):
                candidates.append((ticker, "Short Mean Reversion"))

    # --- AI Wildcard Play (Fallback) ---
    if not candidates and base_universe:
        log.warning("No rule-based candidates found. Executing 'AI Wildcard' play.")
        # As a simple fallback, let's select the stock with the highest 5-day momentum in the regime's direction.
        best_performer, performance = None, -float('inf')
        is_bullish = regime in ["Uptrend", "Bullish Pullback"]
        
        for ticker in base_universe:
            df = full_data_cache[ticker].loc[:point_in_time]
            if len(df) > 5:
                perf = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100
                
                # In bullish regime, find max positive perf. In bearish, find max negative perf (i.e., min perf).
                current_metric = perf if is_bullish else -perf
                if current_metric > performance:
                    performance = current_metric
                    best_performer = ticker

        if best_performer:
            log.info(f"AI Wildcard selected: {best_performer}")
            candidates.append((best_performer, "AI Wildcard Review"))

    unique_candidates = list(dict.fromkeys(candidates)) # Preserve order and get unique
    log.info(f"✅ Playbook complete. Found {len(unique_candidates)} final candidate(s).")
    return unique_candidates