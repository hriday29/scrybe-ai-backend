# angelone_retriever.py
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
import logging  # Make sure logging is imported
import pyotp  # ADDED: For dynamic TOTP generation
import requests  # ADDED: For downloading instrument list
from functools import wraps
from py_vollib_vectorized import vectorized_implied_volatility, vectorized_delta, vectorized_gamma, vectorized_theta

# You must install the Angel One SDK first:
# pip install smartapi-python
from smartapi_client.smartConnect import SmartConnect

# --- Rate Limiting Decorator ---
API_CALL_DELAY = 0.35 # Delay of 350ms between calls

def rate_limited(func):
    """
    Decorator to enforce a delay between API calls to prevent hitting rate limits.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        time.sleep(API_CALL_DELAY)
        return func(*args, **kwargs)
    return wrapper

# --- Module-Level API Session & Cache Management ---
smart_api_session = None
TOKEN_CACHE = {}  # In-memory cache for instrument tokens to improve speed.
INSTRUMENT_LIST = None  # Cache for the entire instrument list

# Angel One master instrument list URL
INSTRUMENT_LIST_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

def _download_instrument_list() -> list | None:
    """
    Downloads the master instrument list from Angel One's CDN.
    Returns a list of instrument dictionaries or None on failure.
    """
    try:
        log.info("Downloading instrument list from Angel One...")
        response = requests.get(INSTRUMENT_LIST_URL, timeout=30)
        response.raise_for_status()
        
        instruments = response.json()
        if isinstance(instruments, list) and len(instruments) > 0:
            log.info(f"Successfully downloaded {len(instruments)} instruments.")
            return instruments
        else:
            log.error("Downloaded instrument list is empty or invalid format.")
            return None
            
    except requests.RequestException as e:
        log.error(f"Network error downloading instrument list: {e}", exc_info=True)
        return None
    except ValueError as e:
        log.error(f"JSON parsing error for instrument list: {e}", exc_info=True)
        return None
    except Exception as e:
        log.error(f"Unexpected error downloading instrument list: {e}", exc_info=True)
        return None

def initialize_angelone_session() -> bool:
    """
    Initializes and authenticates the Angel One API session.
    This should be called once when the application starts. It is idempotent.
    Returns True on successful initialization, False otherwise.
    """
    global smart_api_session, INSTRUMENT_LIST

    try:
        # If already initialized and looks valid, return early.
        if smart_api_session and getattr(smart_api_session, "feed_token", None):
            log.info("Angel One session already initialized and appears valid.")
            return True

        if smart_api_session and not getattr(smart_api_session, "feed_token", None):
            log.warning("Angel One session found but may have expired. Re-initializing...")

        log.info("Initializing new Angel One API session...")
        api_key = config.ANGELONE_API_KEY
        client_id = config.ANGELONE_CLIENT_ID
        password = config.ANGELONE_PASSWORD

        # --- Dynamic TOTP Generation ---
        # The TOTP secret key should be stored in your config/env, not the code itself.
        totp_secret = getattr(config, "ANGELONE_TOTP_SECRET", None)
        if not totp_secret:
            log.error("ANGELONE_TOTP_SECRET missing in config; cannot generate TOTP for login.")
            return False

        try:
            totp = pyotp.TOTP(totp_secret).now()
            log.info("Generated TOTP for Angel One login.")
        except Exception as e:
            log.error(f"Failed to generate TOTP: {e}", exc_info=True)
            return False

        smart_api_session = SmartConnect(api_key=api_key)
        session_data = smart_api_session.generateSession(client_id, password, totp)

        if session_data and session_data.get('status') and session_data['data'].get('jwtToken'):
            log.info("✅ Angel One session successfully authenticated.")
            try:
                user_profile = smart_api_session.getProfile(session_data['data']['refreshToken'])
                if user_profile and user_profile.get('data') and user_profile['data'].get('name'):
                    log.info(f"Logged in as: {user_profile['data']['name']}")
                else:
                    log.info("Logged in, but could not retrieve user profile name.")
            except Exception as e:
                log.warning(f"Could not fetch user profile: {e}")

            # --- Download and PROCESS the instrument list for fast lookups ---
            raw_instrument_list = _download_instrument_list()
            if raw_instrument_list:
                INSTRUMENT_MAP = {}
                for item in raw_instrument_list:
                    if not isinstance(item, dict): continue
                    symbol = item.get('symbol')
                    name = item.get('name')
                    exch = item.get('exch_seg')
                    token = item.get('token')
                    
                    if symbol and exch and token:
                        INSTRUMENT_MAP[f"{symbol.upper()}_{exch}"] = item
                    if name and exch and token:
                        INSTRUMENT_MAP[f"{name.upper()}_{exch}"] = item
                
                INSTRUMENT_LIST = INSTRUMENT_MAP
                log.info(f"✅ Successfully processed and cached {len(INSTRUMENT_LIST)} instruments for fast lookups.")
            else:
                log.warning("⚠️ Could not download instrument list. Token lookups will fail.")

            return True
        else:
            msg = session_data.get('message') if isinstance(session_data, dict) else "Unknown error"
            log.error(f"❌ Angel One login failed: {msg}")
            smart_api_session = None
            return False

    except Exception as e:
        log.error(f"❌ CRITICAL ERROR during Angel One initialization: {e}", exc_info=True)
        smart_api_session = None
        return False

def _get_symbol_token(symbol: str, exchange: str = "NSE") -> str | None:
    """
    Finds the instrument token using a pre-processed dictionary for O(1) lookups.
    """
    # Use the global map instead of the list
    if not isinstance(INSTRUMENT_LIST, dict):
        log.error("Instrument map is not initialized. Cannot look up token.")
        return None

    search_symbol = symbol.replace('.NS', '').strip().upper()
    cache_key = f"{search_symbol}_{exchange}"
    
    if cache_key in TOKEN_CACHE:
        return TOKEN_CACHE[cache_key]

    instrument = INSTRUMENT_LIST.get(cache_key)
    if instrument:
        token = instrument.get('token')
        if token:
            TOKEN_CACHE[cache_key] = token
            log.info(f"✅ Found token {token} for {symbol} via fast lookup.")
            return token

    log.warning(f"Token not found for {symbol} in exchange {exchange}")
    return None

# ===================================================================
# SECTION 1: COMPREHENSIVE TECHNICAL DATA
# ===================================================================

@rate_limited
def get_historical_data(symbol: str, from_date: datetime, to_date: datetime, interval: str = "ONE_DAY") -> pd.DataFrame | None:
    """
    Fetches historical OHLCV data. Now correctly handles datetime objects and
    formats them into the full timestamp string required by the Angel One API.
    """
    log.info(f"[AO] Fetching '{interval}' historical data for {symbol}...")
    if not smart_api_session and not initialize_angelone_session():
        return None

    token = _get_symbol_token(symbol)
    if not token:
        # For indices, Angel One uses names like "NIFTY 50", not tickers like "^NSEI"
        # Let's try a lookup on the symbol name itself if it's an index.
        if symbol.startswith('^'):
            clean_symbol = symbol.replace('^', '').replace('CNX', '') # e.g. ^CNXIT -> IT
            log.info(f"Retrying token lookup for index using cleaned name: {clean_symbol}")
            token = _get_symbol_token(clean_symbol)
            if not token and "BANK" in clean_symbol: # Handle Nifty Bank case
                 token = _get_symbol_token("NIFTY BANK")

        if not token:
             log.warning(f"Final token lookup failed for {symbol}")
             return None

    try:
        # Angel One API requires "YYYY-MM-DD HH:MM" format for all intervals.
        params = {
            "exchange": "NSE",
            "symboltoken": token,
            "interval": interval,
            "fromdate": from_date.strftime('%Y-%m-%d 09:15'),
            "todate": to_date.strftime('%Y-%m-%d 15:30')
        }

        raw_data = smart_api_session.getCandleData(params)

        if not raw_data or raw_data.get('status') is False:
            err_msg = raw_data.get('message', 'Unknown API error')
            log.error(f"[AO] API error for {symbol}: {err_msg}")
            return None

        if not raw_data.get('data'):
            log.warning(f"No historical data returned for {symbol} in the given range.")
            return None

        df = pd.DataFrame(raw_data['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['timestamp'])
        df.set_index('date', inplace=True)
        log.info(f"Successfully fetched {len(df)} candles for {symbol} from Angel One.")
        return df.drop(columns=['timestamp'])

    except Exception as e:
        log.error(f"Error in get_historical_data for {symbol}: {e}", exc_info=True)
        return None

@rate_limited
def get_full_quote(symbol: str, exchange: str = "NSE", token: str = None) -> dict | None:
    """
    Fetches a full quote including LTP, open, high, low, close, and volume.
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE', 'NIFTY')
        exchange: Exchange segment (default: 'NSE')
        token: Optional token to bypass symbol lookup (recommended for indices)
    """
    log.info(f"[AO] Fetching full quote for {symbol}...")
    if not smart_api_session and not initialize_angelone_session():
        return None

    # Use provided token or look it up
    if token:
        log.info(f"Using provided token: {token}")
        instrument_token = token
    else:
        instrument_token = _get_symbol_token(symbol, exchange)
        if not instrument_token:
            return None

    try:
        # Clean symbol for API call (remove exchange suffix)
        clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
        
        quote_data = smart_api_session.ltpData(exchange, clean_symbol, instrument_token)
        
        if quote_data and quote_data.get('status'):
            return quote_data.get('data')
        else:
            err_msg = quote_data.get('message') if isinstance(quote_data, dict) else "Unknown API error"
            log.error(f"Error fetching quote for {symbol}: {err_msg}")
            return None

    except Exception as e:
        log.error(f"Error in get_full_quote for {symbol}: {e}", exc_info=True)
        return None


# Usage examples:
# Option 1: Using symbol (relies on lookup)
# quote = get_full_quote("NIFTY")

# Option 2: Using token directly (fastest, most reliable for indices)
# quote = get_full_quote("Nifty 50", token="99926000")

# Option 3: Common indices with known tokens
KNOWN_INDEX_TOKENS = {
    "NIFTY": "99926000",
    "NIFTY 50": "99926000",
    "BANKNIFTY": "99926009",
    "NIFTY BANK": "99926009",
    "FINNIFTY": "99926037",
    "NIFTY FIN SERVICE": "99926037",
    "NIFTY IT": "99926008",
    "NIFTY MIDCAP 100": "99926011",
    "NIFTY 100": "99926012",
    "NIFTY NEXT 50": "99926013",
    "MIDCPNIFTY": "99926074",
}

def get_index_quote(index_name: str) -> dict | None:
    """
    Convenience function for fetching index quotes using known tokens.
    Supports: NIFTY, NIFTY 50, BANKNIFTY, FINNIFTY
    """
    index_upper = index_name.upper()
    
    if index_upper in KNOWN_INDEX_TOKENS:
        token = KNOWN_INDEX_TOKENS[index_upper]
        return get_full_quote(index_name, exchange="NSE", token=token)
    else:
        log.warning(f"Index {index_name} not in known tokens, falling back to lookup")
        return get_full_quote(index_name, exchange="NSE")

# ===================================================================
# SECTION 2: COMPREHENSIVE OPTIONS DATA (IMPLEMENTED)
# ===================================================================

@rate_limited
def get_option_chain_data(symbol: str, exchange: str = "NFO") -> dict | None:
    """
    Fetches the nearest expiry date and the full option chain for that date.
    Returns a dict with keys: underlying_price, expiry_date, chain
    """
    log.info(f"[AO] Getting full option chain for {symbol}...")
    if (not smart_api_session or not INSTRUMENT_LIST) and not initialize_angelone_session():
        return None

    underlying_symbol = symbol.replace('.NS', '')

    # Step 1: Find all expiry dates for the given stock from the cached instrument list
    try:
        expiry_dates = set()
        for inst in INSTRUMENT_LIST.values():
            if (isinstance(inst, dict) and
                inst.get('instrumenttype') == 'OPTSTK' and
                inst.get('name') == underlying_symbol and
                inst.get('exch_seg') == exchange):
                expiry_dates.add(inst.get('expiry'))

        if not expiry_dates:
            log.warning(f"No option expiry dates found for {symbol} in the instrument list.")
            return None

        # Step 2: Find the nearest expiry date
        today = datetime.now()
        parsed_dates = []
        for d in expiry_dates:
            try:
                parsed_dates.append(datetime.strptime(d, '%d%b%Y'))
            except Exception:
                # ignore unparsable expiry entries
                log.debug(f"Ignoring unparsable expiry date entry: {d}")

        if not parsed_dates:
            log.warning(f"No parsable expiry dates found for {symbol}.")
            return None

        nearest = min(parsed_dates, key=lambda dt: abs(dt - today))
        nearest_expiry = nearest.strftime('%d%b%Y').upper()

        log.info(f"Nearest expiry date for {symbol} is {nearest_expiry}.")

    except Exception as e:
        log.error(f"Error finding expiry date for {symbol}: {e}", exc_info=True)
        return None

    # Step 3: Fetch the option chain for the nearest expiry
    try:
        chain_data = smart_api_session.getOptionChain(exchange, underlying_symbol, nearest_expiry)

        if chain_data and chain_data.get('status') and chain_data.get('data'):
            log.info(f"✅ Successfully fetched option chain for {symbol} on {nearest_expiry}.")
            return {
                "underlying_price": chain_data['data'].get('underlyingValue'),
                "expiry_date": nearest_expiry,
                "chain": chain_data['data'].get('options', [])
            }
        else:
            err_msg = chain_data.get('message') if isinstance(chain_data, dict) else "Unknown API error"
            log.error(f"API error fetching option chain for {symbol}: {err_msg}")
            return None

    except Exception as e:
        log.error(f"Error in get_option_chain_data for {symbol}: {e}", exc_info=True)
        return None

def get_option_greeks(symbol: str) -> dict | None:
    """
    Fetches the option chain and calculates IV and Greeks for near-the-money strikes.
    This provides critical sentiment and volatility data for the AI.
    """
    log.info(f"[AO] Fetching and calculating Option Greeks for {symbol}...")
    try:
        chain_data = get_option_chain_data(symbol)
        if not chain_data or not chain_data.get('chain'):
            return None

        ltp = chain_data.get('underlying_price')
        if ltp is None:
            log.warning(f"Could not get underlying LTP for {symbol} to calculate greeks.")
            return None

        df = pd.DataFrame(chain_data['chain'])
        required_cols = ['strikePrice', 'optionType', 'lastPrice', 'expiryDate']
        if not all(col in df.columns for col in required_cols):
            log.warning(f"Option chain for {symbol} is missing required columns for greeks calculation.")
            return None

        # --- Data Cleaning and Preparation ---
        df['strikePrice'] = pd.to_numeric(df['strikePrice'], errors='coerce')
        df['lastPrice'] = pd.to_numeric(df['lastPrice'], errors='coerce')
        df = df.dropna(subset=['strikePrice', 'lastPrice'])
        if df.empty:
            log.warning(f"No valid option contracts with price data found for {symbol}.")
            return None

        # Calculate Time to Expiry in years
        expiry_dt = pd.to_datetime(df['expiryDate'].iloc[0], format='%d%b%Y')
        time_to_expiry = (expiry_dt - datetime.now()).days / 365.25
        if time_to_expiry <= 0:
            log.warning(f"Expiry for {symbol} is in the past or today. Cannot calculate greeks.")
            return None
        
        # --- Find 3 ATM Strikes ---
        atm_strikes_df = df.iloc[(df['strikePrice'] - ltp).abs().argsort()[:3]]

        # --- Vectorized Calculations ---
        price_array = atm_strikes_df['lastPrice'].values
        strike_array = atm_strikes_df['strikePrice'].values
        flag_array = atm_strikes_df['optionType'].apply(lambda x: 'c' if x == 'CE' else 'p').values
        
        # Calculate Implied Volatility
        iv_array = vectorized_implied_volatility(
            price=price_array, S=ltp, K=strike_array, t=time_to_expiry, r=0.05, flag=flag_array, return_as='numpy', on_error='warn'
        )
        
        # --- FIX: Calculate Greeks individually ---
        greeks = {
            'delta': vectorized_delta(flag=flag_array, S=ltp, K=strike_array, t=time_to_expiry, r=0.05, sigma=iv_array),
            'gamma': vectorized_gamma(flag=flag_array, S=ltp, K=strike_array, t=time_to_expiry, r=0.05, sigma=iv_array),
            'theta': vectorized_theta(flag=flag_array, S=ltp, K=strike_array, t=time_to_expiry, r=0.05, sigma=iv_array)
        }

        # --- Format Output ---
        results = {}
        for i, strike in enumerate(strike_array):
            # Check if IV calculation was successful for this strike
            iv_valid = iv_array[i] > 0 and not pd.isna(iv_array[i])
            
            results[f"strike_{strike}"] = {
                "optionType": atm_strikes_df.iloc[i]['optionType'],
                "implied_volatility": round(iv_array[i], 4) if iv_valid else "N/A",
                "delta": round(greeks['delta'][i], 4) if iv_valid else "N/A",
                "theta": round(greeks['theta'][i], 4) if iv_valid else "N/A",
                "gamma": round(greeks['gamma'][i], 4) if iv_valid else "N/A",
            }
        
        log.info(f"✅ Successfully calculated greeks for {len(results)} NTM strikes for {symbol}.")
        return results

    except Exception as e:
        log.error(f"Error in get_option_greeks for {symbol}: {e}", exc_info=True)
        return None

# ===================================================================
# SECTION 3: ADVANCED MARKET DATA
# ===================================================================

@rate_limited
def get_market_depth(symbol: str, exchange: str = "NSE") -> dict | None:
    """
    Fetches Level 2 market depth (top 5 bids and asks with orders).
    """
    log.info(f"[AO] Fetching market depth for {symbol}...")
    if not smart_api_session and not initialize_angelone_session():
        return None

    token = _get_symbol_token(symbol)
    if not token:
        return None

    try:
        depth_data = smart_api_session.getMarketDepth('FULL', exchange, token)
        if depth_data and depth_data.get('status'):
            return depth_data.get('data')
        else:
            err_msg = depth_data.get('message') if isinstance(depth_data, dict) else "Unknown API error"
            log.error(f"Error fetching market depth for {symbol}: {err_msg}")
            return None

    except Exception as e:
        log.error(f"Error in get_market_depth for {symbol}: {e}", exc_info=True)
        return None
