# quantitative_screener.py
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
MIN_AVG_VOLUME = 500000

# ==============================================================================
# --- NEW UNIFIED FUNDAMENTAL SCORING FUNCTION ---
# ==============================================================================

def get_fundamental_score(ticker: str, point_in_time: pd.Timestamp) -> float:
    """
    Calculates a normalized fundamental score from -1 (very weak) to +1 (very strong).
    Returns 0.0 if data is unavailable.
    """
    try:
        point_in_time_utc = pd.to_datetime(point_in_time, utc=True).to_pydatetime()
        cursor = database_manager.db.fundamentals.find({
            "ticker": ticker, "asOfDate": {"$lte": point_in_time_utc}
        }).sort("asOfDate", -1).limit(1)
        results = list(cursor)
        
        if not results: return 0.0
        
        fundamentals = results[0]
        score, available_metrics = 0, 0
        
        # Metric 1: Return on Equity (ROE)
        if fundamentals.get("returnOnEquity") is not None:
            available_metrics += 1
            roe = fundamentals["returnOnEquity"]
            if roe > 0.15: score += 1      # Strong
            elif roe > 0.05: score += 0.5  # Okay
            elif roe < 0: score -= 1      # Weak (losing money)
            
        # Metric 2: Profit Margins
        if fundamentals.get("profitMargins") is not None:
            available_metrics += 1
            margin = fundamentals["profitMargins"]
            if margin > 0.15: score += 1      # Strong
            elif margin > 0.05: score += 0.5  # Okay
            elif margin < 0: score -= 1      # Weak
            
        # Metric 3: Debt to Equity
        if fundamentals.get("debtToEquity") is not None:
            available_metrics += 1
            de = fundamentals["debtToEquity"]
            if de < 100: score += 1      # Strong (D/E < 1.0)
            elif de < 200: score += 0.5  # Okay   (D/E < 2.0)
            else: score -= 1             # Weak   (D/E > 2.0)

        if available_metrics == 0: return 0.0
        
        # Normalize the score to be between -1 and +1
        max_possible_score = available_metrics
        return score / max_possible_score

    except Exception as e:
        log.error(f"Error calculating fundamental score for {ticker}: {e}")
        return 0.0

# ==============================================================================
# --- MODIFIED UNIVERSE PREPARATION ---
# ==============================================================================

def _prepare_filtered_universe(actionable_sectors: list[str], full_data_cache: dict, point_in_time: pd.Timestamp) -> list[str]:
    """
    Pre-filters the universe ONLY by sector and liquidity.
    The rigid fundamental checks have been REMOVED.
    """
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
        
        # *** THE CRITICAL CHANGE IS HERE: ***
        # The calls to _is_fundamentally_vulnerable and _passes_fundamental_health_check
        # have been completely removed.
        
        qualified.append(ticker)
    return qualified

# ==============================================================================
# --- TECHNICAL RULESETS (UNCHANGED) ---
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
# --- NEW UNIFIED, PLAYBOOK-DRIVEN CANDIDATE GENERATOR ---
# ==============================================================================

def get_strategy_candidates(market_state: dict, full_data_cache: dict, point_in_time: pd.Timestamp) -> list[tuple[str, str]]:
    """
    Generates a ranked list of candidates based on a combined fundamental
    and technical score, aligned with the market regime.
    """
    regime = market_state.get('market_regime', {}).get('regime_status', 'Sideways')
    actionable_sectors = market_state.get('actionable_sectors', [])
    
    # 1. Get base universe (now only filters by liquidity and sector)
    base_universe = _prepare_filtered_universe(actionable_sectors, full_data_cache, point_in_time)
    log.info(f"Found {len(base_universe)} liquid stocks in actionable sectors to screen.")

    candidate_scores = []
    for ticker in base_universe:
        # 2. Calculate Fundamental Score for every stock
        fundamental_score = get_fundamental_score(ticker, point_in_time)

        # 3. Calculate Technical Score based on regime-specific rules
        df = _add_indicators(full_data_cache[ticker].loc[:point_in_time].copy())
        if df.empty: continue
        
        technical_score = 0.0
        play_reason = "N/A"
        
        if regime == "Uptrend" and _check_long_momentum_rules(df):
            technical_score = 1.0
            play_reason = "Long Momentum"
        elif regime == "Bullish Pullback" and _check_long_mean_reversion_rules(df):
            technical_score = 1.0
            play_reason = "Long Mean Reversion"
        elif regime == "Downtrend" and _check_short_breakdown_rules(df):
            technical_score = -1.0 # Negative score for short signals
            play_reason = "Short Breakdown"
        elif regime == "Bearish Rally" and _check_short_mean_reversion_rules(df):
            technical_score = -1.0 # Negative score for short signals
            play_reason = "Short Mean Reversion"

        # 4. Only consider stocks that match a technical play
        if technical_score != 0.0:
            # 5. Calculate Final Weighted Score
            # Weights can be tuned. Start with 60% Technical, 40% Fundamental.
            final_score = (technical_score * 0.6) + (fundamental_score * 0.4)
            candidate_scores.append((ticker, play_reason, final_score))

    # 6. Rank candidates by their final score
    is_bullish_regime = regime in ["Uptrend", "Bullish Pullback"]
    # For bullish, we want the highest positive scores. For bearish, the most negative scores.
    sorted_candidates = sorted(candidate_scores, key=lambda x: x[2], reverse=is_bullish_regime)

    log.info(f"Screened {len(candidate_scores)} potential plays. Top 5 scores: {[(c[0], c[2]) for c in sorted_candidates[:5]]}")
    
    # 7. Return the top N candidates for the AI to analyze
    top_n = 5 
    final_candidates = [(ticker, reason) for ticker, reason, score in sorted_candidates[:top_n]]
    
    log.info(f"✅ Playbook complete. Found {len(final_candidates)} final candidate(s).")
    return final_candidates
