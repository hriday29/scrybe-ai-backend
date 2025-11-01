"""
Quantitative Screener
---------------------
Point-in-time candidate generator that scans the universe for significant daily candlestick patterns
using TA-Lib and forwards all matches to the AI committee without ranking.

Role in the system:
- Acts as a master dispatcher for candidate selection; the strategy/AI layers decide conviction.
- Keeps logic simple and bias-free by focusing only on patterns at the decision date.

Inputs/Outputs:
- Inputs: market_state (currently unused but reserved), full_data_cache dict of OHLCV DataFrames,
  point_in_time for slicing.
- Output: list of (ticker, reason) tuples like ("RELIANCE.NS", "Bullish Pattern: Hammer").

Notes:
- Requires at least ~20 bars per ticker and standard OHLC column names.
- Handles TA-Lib errors defensively and de-duplicates results.
"""
import pandas as pd
from logger_config import log
import pandas_ta as ta # Keep pandas_ta if needed for other potential future indicators
import talib # <-- ADDED
import numpy as np # <-- ADDED

# --- Constants ---
# MIN_AVG_VOLUME = 500000 # Can be added back later if liquidity filtering is needed

# ==============================================================================
# --- REMOVED: FUNDAMENTAL SCORING ---
# ==============================================================================
# (get_fundamental_score function removed)

# ==============================================================================
# --- REMOVED: UNIVERSE PREPARATION ---
# ==============================================================================
# (_prepare_filtered_universe function removed, filtering happens implicitly)

# ==============================================================================
# --- REMOVED: OLD TECHNICAL RULESETS ---
# ==============================================================================
# (_add_indicators, _check_long_momentum_rules, _check_short_breakdown_rules,
#  _check_long_mean_reversion_rules, _check_short_mean_reversion_rules removed)

# ==============================================================================
# --- NEW: CANDLESTICK PATTERN RECOGNITION ---
# ==============================================================================

def _check_candlestick_patterns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """
    Identifies significant bullish or bearish candlestick patterns on the last day.
    Uses TA-Lib for pattern recognition.

    Returns:
        tuple[str | None, str | None]: (bullish_pattern_name, bearish_pattern_name)
                                       Returns pattern name if found, else None.
    """
    if df is None or len(df) < 20: # Need some history for patterns
         # Log less verbosely, maybe debug level if too noisy
         # log.debug(f"Skipping pattern check: DataFrame None or length {len(df) if df is not None else 'None'} < 20")
         return None, None

    # TA-Lib requires numpy arrays and specific column names
    # Ensure columns exist before accessing .values
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        log.warning(f"Skipping pattern check: Missing one of {required_cols} columns.")
        return None, None

    try:
        op = df['open'].values
        hi = df['high'].values
        lo = df['low'].values
        cl = df['close'].values

        # --- Bullish Patterns ---
        # Look for a positive signal (-100 for bearish, 0, +100 for bullish) on the *last* day
        # Using [-1] to get the result for the most recent candle
        if talib.CDLHAMMER(op, hi, lo, cl)[-1] > 0: return "Hammer", None
        if talib.CDLINVERTEDHAMMER(op, hi, lo, cl)[-1] > 0: return "Inverted Hammer", None
        if talib.CDLMORNINGSTAR(op, hi, lo, cl, penetration=0)[-1] > 0: return "Morning Star", None # penetration=0 relaxes the rule slightly
        if talib.CDL3WHITESOLDIERS(op, hi, lo, cl)[-1] > 0: return "Three White Soldiers", None
        if talib.CDLENGULFING(op, hi, lo, cl)[-1] > 0: return "Bullish Engulfing", None
        if talib.CDLPIERCING(op, hi, lo, cl)[-1] > 0: return "Piercing Pattern", None
        # Add more bullish patterns here if desired (e.g., CDL3INSIDEUP)

        # --- Bearish Patterns ---
        if talib.CDLHANGINGMAN(op, hi, lo, cl)[-1] < 0: return None, "Hanging Man"
        if talib.CDLSHOOTINGSTAR(op, hi, lo, cl)[-1] < 0: return None, "Shooting Star"
        if talib.CDLEVENINGSTAR(op, hi, lo, cl, penetration=0)[-1] < 0: return None, "Evening Star" # penetration=0 relaxes the rule slightly
        if talib.CDL3BLACKCROWS(op, hi, lo, cl)[-1] < 0: return None, "Three Black Crows"
        if talib.CDLENGULFING(op, hi, lo, cl)[-1] < 0: return None, "Bearish Engulfing"
        if talib.CDLDARKCLOUDCOVER(op, hi, lo, cl)[-1] < 0: return None, "Dark Cloud Cover"
        # Add more bearish patterns here if desired (e.g., CDL3INSIDEDOWN)

        # --- Neutral/Indecision Patterns (Optional) ---
        # if talib.CDLDOJI(op, hi, lo, cl)[-1] != 0: return "Doji", "Doji" # Could signal reversal potential

    except Exception as e:
        # Catch potential errors from TA-Lib (e.g., insufficient data length after NaNs internally)
        log.error(f"TA-Lib Error during pattern check: {e}")
        return None, None

    return None, None # No significant pattern found

# ==============================================================================
# --- MODIFIED: THE UNIFIED CANDIDATE GENERATOR (NO RANKING) ---
# ==============================================================================

def get_strategy_candidates(market_state: dict, full_data_cache: dict, point_in_time: pd.Timestamp) -> list[tuple[str, str]]:
    """
    MODIFIED: Screens the entire universe based on significant candlestick patterns
    identified on the decision date. Returns ALL matches without ranking.
    """
    log.info("--- Starting Candlestick Pattern Screener ---")

    # --- Step 1: Define Universe ---
    # Use ALL tickers from the cache that look like stocks (.NS suffix)
    stock_universe = [ticker for ticker in full_data_cache.keys() if '.NS' in ticker]
    if not stock_universe:
        log.warning("No stock tickers found in the data cache.")
        return []
    log.info(f"Screening {len(stock_universe)} potential stock tickers for candlestick patterns...")


    # --- Step 2: Iterate and Check Patterns ---
    all_pattern_matches = []
    patterns_found_count = 0
    for i, ticker in enumerate(stock_universe):
        df_full = full_data_cache.get(ticker)
        if df_full is None or df_full.empty:
            continue # Skip if no data for this ticker

        # Slice data up to the decision point INCLUSIVE
        df_slice = df_full.loc[:point_in_time]

        # Basic check: Ensure the slice isn't empty and has enough data for TA-Lib
        if df_slice.empty or len(df_slice) < 20: # TA-Lib often needs a minimum lookback
            continue

        # Check for patterns on the latest day's data within the slice
        try:
            # Pass the DataFrame slice directly
            bullish_pattern, bearish_pattern = _check_candlestick_patterns(df_slice)

            if bullish_pattern:
                # Add tuple: (ticker_symbol, reason_string)
                all_pattern_matches.append((ticker, f"Bullish Pattern: {bullish_pattern}"))
                patterns_found_count += 1
                # Log progress intermittently to avoid flooding logs
                if (i + 1) % 200 == 0: log.info(f"({i+1}/{len(stock_universe)}) Screen progress...")

            elif bearish_pattern:
                # Add tuple: (ticker_symbol, reason_string)
                all_pattern_matches.append((ticker, f"Short Pattern: {bearish_pattern}"))
                patterns_found_count += 1
                if (i + 1) % 200 == 0: log.info(f"({i+1}/{len(stock_universe)}) Screen progress...")

        except Exception as e:
            # Log specific error but continue screening other stocks
            log.error(f"Error checking patterns for {ticker} on {point_in_time.date()}: {e}")
            continue # Move to the next ticker


    log.info(f"--- Candlestick Screen Complete ---")
    log.info(f"Found {patterns_found_count} potential setups across {len(stock_universe)} tickers.")

    # --- Step 3: Return ALL unique matches ---
    # Use dict.fromkeys to efficiently get unique tuples (preserves order in Python 3.7+)
    unique_matches = list(dict.fromkeys(all_pattern_matches))
    log.info(f"✅ Forwarding {len(unique_matches)} unique candidate(s) to AI Committee.")

    return unique_matches # Return the full list without ranking or filtering