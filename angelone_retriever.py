# angelone_retriever.py (Cleaned Version 2.0)
# DESIGN PHILOSOPHY:
# This module is the single, comprehensive gateway for all Angel One API data.
# It prioritizes data richness and completeness over efficiency. The goal is to
# extract every valuable piece of information available to power the Scrybe AI.

import pandas as pd
from logger_config import log
import config  # To get API keys and credentials
import time
from datetime import datetime
import sys
import os
import logging # Make sure logging is imported

print("--- STARTING SCRYBE-AI DEBUG ---")
print(f"Current Working Directory: {os.getcwd()}")
print("Python sys.path:")
for path in sys.path:
    print(f"  - {path}")
print("--- END OF DEBUG INFO ---\n")

# You must install the Angel One SDK first:
# pip install smartapi-python
try:
    print("Attempting to import SmartConnect...")
    from smartapi import SmartConnect
    print("Successfully imported SmartConnect.")
except ImportError as e:
    # This will log the TRUE, detailed error message to your console
    print(f"!!! DETAILED IMPORT ERROR: {e}")
    logging.getLogger("ScrybeAI").error(f"Detailed import error for smartapi", exc_info=True)
    raise e

# --- Module-Level API Session & Cache Management ---
smart_api_session = None
TOKEN_CACHE = {}  # In-memory cache for instrument tokens to improve speed.
INSTRUMENT_LIST = None # Cache for the entire instrument list

def initialize_angelone_session():
    """
    Initializes and authenticates the Angel One API session.
    This should be called once when the application starts. It is idempotent.
    """
    global smart_api_session, INSTRUMENT_LIST
    if smart_api_session and smart_api_session.feed_token:
        log.info("Angel One session already initialized and appears valid.")
        return True
    
    if smart_api_session and not smart_api_session.feed_token:
        log.warning("Angel One session found but may have expired. Re-initializing...")

    try:
        log.info("Initializing new Angel One API session...")
        api_key = config.ANGELONE_API_KEY
        client_id = config.ANGELONE_CLIENT_ID
        password = config.ANGELONE_PASSWORD
        totp = config.ANGELONE_TOTP

        smart_api_session = SmartConnect(api_key=api_key)
        session_data = smart_api_session.generateSession(client_id, password, totp)

        if session_data['status'] and session_data['data'].get('jwtToken'):
            log.info("✅ Angel One session successfully authenticated.")
            user_profile = smart_api_session.getProfile(session_data['data']['refreshToken'])
            log.info(f"Logged in as: {user_profile['data']['name']}")
            
            # Fetch and cache the instrument list upon successful login
            INSTRUMENT_LIST = smart_api_session.getInstrumentList()
            if isinstance(INSTRUMENT_LIST, list) and len(INSTRUMENT_LIST) > 0:
                log.info(f"Successfully fetched and cached {len(INSTRUMENT_LIST)} instruments.")
            else:
                 log.error("Failed to fetch instrument list from Angel One after login.")
                 INSTRUMENT_LIST = None

            return True
        else:
            log.error(f"❌ Angel One login failed: {session_data.get('message')}")
            smart_api_session = None
            return False

    except Exception as e:
        log.error(f"❌ CRITICAL ERROR during Angel One initialization: {e}", exc_info=True)
        smart_api_session = None
        return False

def _get_symbol_token(symbol: str, exchange: str = "NSE") -> str | None:
    """
    Finds the Angel One instrument token for a given ticker symbol.
    Uses an in-memory cache to avoid repeated lookups.
    """
    cache_key = f"{symbol}_{exchange}"
    if cache_key in TOKEN_CACHE:
        return TOKEN_CACHE[cache_key]

    if not INSTRUMENT_LIST:
        log.error("Cannot get symbol token, instrument list is not available.")
        return None

    try:
        search_symbol = symbol.replace('.NS', '')
        for instrument in INSTRUMENT_LIST:
            if isinstance(instrument, dict) and instrument.get('symbol') == search_symbol and instrument.get('exch_seg') == exchange:
                token = instrument['token']
                TOKEN_CACHE[cache_key] = token
                return token

        log.warning(f"Token not found for {symbol} in exchange {exchange}")
        return None
    except Exception as e:
        log.error(f"Error finding token for {symbol}: {e}")
        return None

# ===================================================================
# SECTION 1: COMPREHENSIVE TECHNICAL DATA
# ===================================================================

def get_historical_data(symbol: str, from_date: str, to_date: str, interval: str = "ONE_DAY") -> pd.DataFrame | None:
    """
    Fetches historical OHLCV data. Interval can be: ONE_MINUTE, THREE_MINUTE,
    FIVE_MINUTE, TEN_MINUTE, FIFTEEN_MINUTE, THIRTY_MINUTE, ONE_HOUR, ONE_DAY.
    """
    log.info(f"[AO] Fetching '{interval}' historical data for {symbol}...")
    if not smart_api_session and not initialize_angelone_session(): return None

    token = _get_symbol_token(symbol)
    if not token: return None

    try:
        params = {
            "exchange": "NSE",
            "symboltoken": token,
            "interval": interval,
            "fromdate": from_date,
            "todate": to_date
        }
        raw_data = smart_api_session.getCandleData(params)

        if not raw_data or raw_data.get('status') is False:
            log.error(f"[AO] API error for {symbol}: {raw_data.get('message')}")
            return None

        if not raw_data.get('data'):
             log.warning(f"No historical data returned for {symbol} in the given range.")
             return None

        df = pd.DataFrame(raw_data['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['timestamp'])
        df.set_index('date', inplace=True)
        log.info(f"Successfully fetched {len(df)} candles for {symbol}.")
        return df.drop(columns=['timestamp'])

    except Exception as e:
        log.error(f"Error in get_historical_data for {symbol}: {e}", exc_info=True)
        return None

def get_full_quote(symbol: str, exchange: str = "NSE") -> dict | None:
    """Fetches a full quote including LTP, open, high, low, close, and volume."""
    log.info(f"[AO] Fetching full quote for {symbol}...")
    if not smart_api_session and not initialize_angelone_session(): return None

    token = _get_symbol_token(symbol)
    if not token: return None

    try:
        quote_data = smart_api_session.ltpData(exchange, symbol.replace('.NS', ''), token)
        if quote_data and quote_data.get('status'):
            return quote_data['data']
        else:
            log.error(f"Error fetching quote for {symbol}: {quote_data.get('message')}")
            return None

    except Exception as e:
        log.error(f"Error in get_full_quote for {symbol}: {e}", exc_info=True)
        return None

# ===================================================================
# SECTION 2: COMPREHENSIVE OPTIONS DATA (IMPLEMENTED)
# ===================================================================

def get_option_chain_data(symbol: str, exchange: str = "NFO") -> dict | None:
    """
    Fetches the nearest expiry date and the full option chain for that date.
    This is now a single, robust function.
    """
    log.info(f"[AO] Getting full option chain for {symbol}...")
    if (not smart_api_session or not INSTRUMENT_LIST) and not initialize_angelone_session():
        return None

    underlying_symbol = symbol.replace('.NS', '')
    
    # Step 1: Find all expiry dates for the given stock from the cached instrument list
    try:
        expiry_dates = set()
        for inst in INSTRUMENT_LIST:
            if (isinstance(inst, dict) and 
                inst.get('instrumenttype') == 'OPTSTK' and 
                inst.get('name') == underlying_symbol and 
                inst.get('exch_seg') == exchange):
                expiry_dates.add(inst['expiry'])
        
        if not expiry_dates:
            log.warning(f"No option expiry dates found for {symbol} in the instrument list.")
            return None

        # Step 2: Find the nearest expiry date
        today = datetime.now()
        nearest_expiry = min(
            (datetime.strptime(d, '%d%b%Y') for d in expiry_dates if d),
            key=lambda d: abs(d - today)
        ).strftime('%d%b%Y').upper()
        
        log.info(f"Nearest expiry date for {symbol} is {nearest_expiry}.")

    except Exception as e:
        log.error(f"Error finding expiry date for {symbol}: {e}")
        return None

    # Step 3: Fetch the option chain for the nearest expiry
    try:
        chain_data = smart_api_session.getOptionChain(exchange, underlying_symbol, nearest_expiry)

        if chain_data and chain_data.get('status') and chain_data.get('data'):
            log.info(f"✅ Successfully fetched option chain for {symbol} on {nearest_expiry}.")
            # The structure of the response needs to be parsed into a more usable format.
            # This is a conceptual structure of the final, parsed object.
            return {
                "underlying_price": chain_data['data'].get('underlyingValue'),
                "expiry_date": nearest_expiry,
                "chain": chain_data['data'].get('options', [])
            }
        else:
            log.error(f"API error fetching option chain for {symbol}: {chain_data.get('message')}")
            return None

    except Exception as e:
        log.error(f"Error in get_full_option_chain for {symbol}: {e}", exc_info=True)
        return None

def get_option_greeks(symbol: str) -> dict | None:
    """
    Fetches option greeks (Delta, IV, etc.) for near-the-money (NTM) strikes.
    This provides deep insight into option market positioning.
    """
    log.info(f"[AO] Fetching Option Greeks for {symbol}...")
    try:
        # First, we need the option chain to find the strikes.
        chain_data = get_option_chain_data(symbol)
        if not chain_data or not chain_data.get('chain'):
            return None

        ltp = chain_data.get('underlying_price')
        if not ltp:
            log.warning(f"Could not get underlying LTP for {symbol} to find NTM strikes.")
            return None

        df = pd.DataFrame(chain_data['chain'])
        df['strikePrice'] = pd.to_numeric(df['strikePrice'])
        
        # Find the 3 nearest strikes to the current price
        ntm_strikes = df.iloc[(df['strikePrice'] - ltp).abs().argsort()[:3]]['strikePrice'].unique().tolist()
        
        greeks_data = {}
        # The Angel One SDK unfortunately does not have a direct greeks function.
        # A real implementation would require either calculating them manually (very complex)
        # or using a different library/API that provides them.
        # For now, we will create a placeholder and note this as a future enhancement.
        log.warning("Angel One SDK does not provide direct option greeks. This is a placeholder.")
        greeks_data = {
            "status": "Placeholder",
            "note": "Angel One SDK lacks a direct greeks endpoint. Manual calculation or another API is needed.",
            "ntm_strikes_analyzed": ntm_strikes,
        }
        return greeks_data

    except Exception as e:
        log.error(f"Error in get_option_greeks for {symbol}: {e}", exc_info=True)
        return None
    
# ===================================================================
# SECTION 3: ADVANCED MARKET DATA
# ===================================================================

def get_market_depth(symbol: str, exchange: str = "NSE") -> dict | None:
    """
    Fetches Level 2 market depth (top 5 bids and asks with orders).
    """
    log.info(f"[AO] Fetching market depth for {symbol}...")
    if not smart_api_session and not initialize_angelone_session(): return None

    token = _get_symbol_token(symbol)
    if not token: return None

    try:
        depth_data = smart_api_session.getMarketDepth('FULL', exchange, token)
        if depth_data and depth_data.get('status'):
            return depth_data['data']
        else:
            log.error(f"Error fetching market depth for {symbol}: {depth_data.get('message')}")
            return None

    except Exception as e:
        log.error(f"Error in get_market_depth for {symbol}: {e}", exc_info=True)
        return None