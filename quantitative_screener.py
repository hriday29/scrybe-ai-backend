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
# --- FUNDAMENTAL SCORING (No Changes Here) ---
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
        
        max_possible_score = available_metrics
        return score / max_possible_score

    except Exception as e:
        log.error(f"Error calculating fundamental score for {ticker}: {e}")
        return 0.0

# ==============================================================================
# --- UNIVERSE PREPARATION (No Changes Here) ---
# ==============================================================================

def _prepare_filtered_universe(actionable_sectors: list[str], full_data_cache: dict, point_in_time: pd.Timestamp) -> list[str]:
    """
    Pre-filters the universe ONLY by sector and liquidity.
    """
    import data_retriever
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
        
        qualified.append(ticker)
    return qualified

# ==============================================================================
# --- TECHNICAL RULESETS (No Changes Here) ---
# ==============================================================================

def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.dropna(inplace=True)
    return df

def _check_long_momentum_rules(df: pd.DataFrame) -> bool:
    latest = df.iloc[-1]
    is_trending = latest['ADX_14'] > 22
    has_momentum = latest['RSI_14'] > 60
    return latest['close'] > latest['EMA_50'] and (is_trending or has_momentum)

def _check_short_breakdown_rules(df: pd.DataFrame) -> bool:
    latest = df.iloc[-1]
    is_trending = latest['ADX_14'] > 22
    has_momentum = latest['RSI_14'] < 40
    return latest['close'] < latest['EMA_50'] and (is_trending or has_momentum)

def _check_long_mean_reversion_rules(df: pd.DataFrame) -> bool:
    latest = df.iloc[-1]
    return latest['EMA_50'] > latest['EMA_200'] and latest['RSI_14'] < 40 and latest['ADX_14'] < 25

def _check_short_mean_reversion_rules(df: pd.DataFrame) -> bool:
    latest = df.iloc[-1]
    return latest['EMA_50'] < latest['EMA_200'] and latest['RSI_14'] > 60 and latest['ADX_14'] < 25

# ==============================================================================
# --- MODIFIED: THE UNIFIED CANDIDATE GENERATOR & RANKING ENGINE ---
# ==============================================================================

def get_strategy_candidates(market_state: dict, full_data_cache: dict, point_in_time: pd.Timestamp) -> list[tuple[str, str]]:
    """
    MODIFIED: This function now runs ALL playbooks to generate a diverse pool of
    both long and short candidates, then ranks them to find the best absolute
    opportunities for the AI to analyze.
    """
    actionable_sectors = market_state.get('actionable_sectors', [])
    
    # 1. Get base universe (filters by liquidity and active sectors)
    base_universe = _prepare_filtered_universe(actionable_sectors, full_data_cache, point_in_time)
    log.info(f"Found {len(base_universe)} liquid stocks in actionable sectors to screen.")

    # 2. *** NEW LOGIC ***
    # Execute ALL relevant playbooks to generate a diverse, unfiltered candidate pool.
    unranked_candidates = []
    log.info("Executing both Bullish and Bearish playbooks to find all potential setups...")
    
    for ticker in base_universe:
        df = _add_indicators(full_data_cache[ticker].loc[:point_in_time].copy())
        if not df.empty:
            # Bullish Playbooks
            if _check_long_momentum_rules(df):
                unranked_candidates.append((ticker, "Long Momentum"))
            if _check_long_mean_reversion_rules(df):
                unranked_candidates.append((ticker, "Long Mean Reversion"))
            # Bearish Playbooks
            if _check_short_breakdown_rules(df):
                unranked_candidates.append((ticker, "Short Breakdown"))
            if _check_short_mean_reversion_rules(df):
                unranked_candidates.append((ticker, "Short Mean Reversion"))

    unique_unranked_candidates = list(dict.fromkeys(unranked_candidates))
    log.info(f"Found {len(unique_unranked_candidates)} raw technical setups across all playbooks.")
    
    if not unique_unranked_candidates:
        log.warning("No technical setups found across any playbook today.")
        return []

    # 3. *** NEW LOGIC ***
    # Score and rank the combined pool of candidates to find the absolute best setups.
    candidate_scores = []
    for ticker, play_reason in unique_unranked_candidates:
        fundamental_score = get_fundamental_score(ticker, point_in_time)
        
        technical_score = 0.0
        if "Long" in play_reason:
            technical_score = 1.0
        elif "Short" in play_reason:
            technical_score = -1.0

        # This final score now represents a blend of technical setup and fundamental health.
        final_score = (technical_score * 0.6) + (fundamental_score * 0.4)
        candidate_scores.append((ticker, play_reason, final_score))

    # 4. *** NEW LOGIC ***
    # Sort by the ABSOLUTE value of the score. This finds the strongest signals,
    # whether they are long (e.g., +1.5) or short (e.g., -1.5).
    sorted_candidates = sorted(candidate_scores, key=lambda x: abs(x[2]), reverse=True)
    log.info(f"Ranked {len(candidate_scores)} candidates. Top 5 scores: {[(c[0], c[2]) for c in sorted_candidates[:5]]}")
    
    # 5. Return the top N candidates for the AI to make the final decision.
    top_n = 8 
    final_candidates = [(ticker, reason) for ticker, reason, score in sorted_candidates[:top_n]]
    
    log.info(f"✅ Playbook complete. Forwarding {len(final_candidates)} final candidate(s) to AI Committee.")
    return final_candidates
