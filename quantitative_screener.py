# quantitative_screener.py (FINAL & COMPLETE VERSION)
import pandas as pd
import config
from logger_config import log
import pandas_ta as ta
import database_manager

# --- Constants (No changes needed) ---
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
    """Performs a robust, ADAPTIVE, score-based fundamental health check."""
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
            if fundamentals["debtToEquity"] < (FUNDAMENTAL_THRESHOLDS["MAX_DEBT_TO_EQUITY"] * 100): score += 1
        
        if available_metrics == 0: return False
        required_score = available_metrics / 2.0
        is_healthy = score >= required_score
        
        log_func = log.info if is_healthy else log.warning
        status_icon = "✅" if is_healthy else "->"
        log_func(
            f"{status_icon} {ticker} fundamental check for {point_in_time.date()}: "
            f"Score {score}/{available_metrics} (Required: {required_score:.1f})"
        )
        return is_healthy
    except Exception: return False

def _prepare_filtered_universe(actionable_sectors: list[str], full_data_cache: dict, point_in_time: pd.Timestamp) -> list[str]:
    """Pre-filters the universe by sector, liquidity, and fundamental health."""
    import data_retriever # Local import to avoid circular dependency
    target_sectors = {SECTOR_NAME_MAPPING[name] for name in actionable_sectors if name in SECTOR_NAME_MAPPING}
    if not target_sectors: return []

    universe = [t for t in full_data_cache.keys() if ".NS" in t]
    stock_sector_map = data_retriever.get_stock_sector_map(universe)
    
    sector_filtered = [t for t, s in stock_sector_map.items() if s in target_sectors]
    
    qualified = []
    for ticker in sector_filtered:
        data = full_data_cache.get(ticker)
        if data is None: continue
        
        df_slice = data.loc[:point_in_time]
        if len(df_slice) < 252: continue
        if df_slice['volume'].tail(20).mean() < MIN_AVG_VOLUME: continue
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

# --- PLAY #1 (CORE STRATEGY) RULESET ---
def _check_long_momentum_rules(df: pd.DataFrame) -> bool:
    """MODIFIED: Softer rules. Requires trend OR momentum, not a strict AND."""
    latest = df.iloc[-1]
    is_trending = latest['ADX_14'] > 22
    has_momentum = latest['RSI_14'] > 60
    return latest['close'] > latest['EMA_50'] and (is_trending or has_momentum)

def _check_short_breakdown_rules(df: pd.DataFrame) -> bool:
    """MODIFIED: Softer rules for breakdown."""
    latest = df.iloc[-1]
    is_trending = latest['ADX_14'] > 22
    has_momentum = latest['RSI_14'] < 40
    return latest['close'] < latest['EMA_50'] and (is_trending or has_momentum)

# --- PLAY #2 (CONTRARIAN STRATEGY) RULESET ---
def _check_long_mean_reversion_rules(df: pd.DataFrame) -> bool:
    """Rule for 'buy the dip' setups."""
    latest = df.iloc[-1]
    # Primary trend is up, but stock is oversold and not in a strong short-term trend
    return latest['EMA_50'] > latest['EMA_200'] and latest['RSI_14'] < 40 and latest['ADX_14'] < 25

def _check_short_mean_reversion_rules(df: pd.DataFrame) -> bool:
    """Rule for 'short the rip' setups."""
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
    
    # Prepare a single, pre-filtered universe of healthy stocks in the right sectors
    base_universe = _prepare_filtered_universe(actionable_sectors, full_data_cache, point_in_time)
    log.info(f"Found {len(base_universe)} healthy, liquid stocks in actionable sectors to screen.")

    # --- Execute Plays Based on Market Regime ---
    if regime == "Uptrend":
        log.info("Executing Play #1: Long Momentum/Trend Confluence")
        for ticker in base_universe:
            df = _add_indicators(full_data_cache[ticker].loc[:point_in_time].copy())
            if not df.empty and _check_long_momentum_rules(df):
                candidates.append((ticker, "Long Momentum"))

    elif regime == "Bullish Pullback":
        log.info("Executing Play #2: Long Mean Reversion (Buy the Dip)")
        for ticker in base_universe:
            df = _add_indicators(full_data_cache[ticker].loc[:point_in_time].copy())
            if not df.empty and _check_long_mean_reversion_rules(df):
                candidates.append((ticker, "Long Mean Reversion"))

    elif regime == "Downtrend":
        log.info("Executing Play #1: Short Breakdown/Trend Confluence")
        for ticker in base_universe:
            df = _add_indicators(full_data_cache[ticker].loc[:point_in_time].copy())
            if not df.empty and _check_short_breakdown_rules(df):
                candidates.append((ticker, "Short Breakdown"))

    elif regime == "Bearish Rally":
        log.info("Executing Play #2: Short Mean Reversion (Short the Rip)")
        for ticker in base_universe:
            df = _add_indicators(full_data_cache[ticker].loc[:point_in_time].copy())
            if not df.empty and _check_short_mean_reversion_rules(df):
                candidates.append((ticker, "Short Mean Reversion"))

    # --- Play #3: The AI Wildcard (if no candidates were found by rules) ---
    if not candidates and base_universe:
        log.warning("No candidates found from standard plays. Executing Play #3: AI Wildcard.")
        best_performer, max_perf = None, -100
        
        # Determine direction based on regime
        is_bullish = regime in ["Uptrend", "Bullish Pullback"]
        
        for ticker in base_universe:
            df = full_data_cache[ticker].loc[:point_in_time]
            if len(df) > 5:
                # Calculate 5-day performance
                perf = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100
                
                # In bullish regime, find best performer. In bearish, find worst performer.
                if is_bullish and perf > max_perf:
                    max_perf = perf
                    best_performer = ticker
                elif not is_bullish and perf < -max_perf: # Using -max_perf to find the most negative
                    max_perf = -perf 
                    best_performer = ticker

        if best_performer:
            log.info(f"AI Wildcard selected: {best_performer} (5-day perf: {max_perf if is_bullish else -max_perf:.2f}%)")
            candidates.append((best_performer, "AI Wildcard Review"))

    unique_candidates = list({t[0]: t for t in candidates}.values())
    log.info(f"✅ Playbook complete. Found {len(unique_candidates)} final candidate(s).")
    return unique_candidates