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
import os
import pyotp  # ADDED: For dynamic TOTP generation
import requests  # ADDED: For downloading instrument list
from functools import wraps
import json
from py_vollib_vectorized import vectorized_implied_volatility, vectorized_delta, vectorized_gamma, vectorized_theta  # type: ignore

# You must install the Angel One SDK first:
# pip install smartapi-python
from smartapi_client.smartConnect import SmartConnect  # type: ignore

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
INSTRUMENT_LIST = None  # Cache for the entire instrument map (was list)

# Cache for the master instrument list on disk
INSTRUMENT_CACHE_FILE = os.path.join(config.CACHE_DIR, 'angelone_instrument_master.json')
INSTRUMENT_CACHE_TTL_HOURS = 24  # Cache for 24 hours

# NEW: Pre-processed maps for O(1) expiry lookups
OPTION_EXPIRY_MAP = {"STK": {}, "IDX": {}}
FUTURES_EXPIRY_MAP = {"STK": {}, "IDX": {}}

# Angel One master instrument list URL
INSTRUMENT_LIST_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

def _load_instrument_list(cache_file: str, ttl_hours: int) -> list | None:
    """
    Downloads the master instrument list from Angel One's CDN or loads
    it from a local cache if it's not stale.
    """
    # 1. Check if a valid cache file exists
    if os.path.exists(cache_file):
        try:
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - file_mod_time).total_seconds() < ttl_hours * 3600:
                log.info(f"Loading instrument list from cache file: {cache_file}")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    instruments = json.load(f)
                    if isinstance(instruments, list) and len(instruments) > 0:
                        log.info(f"Successfully loaded {len(instruments)} instruments from cache.")
                        return instruments
            else:
                log.info("Instrument cache is stale. Re-downloading...")
        except (json.JSONDecodeError, IOError) as e:
            log.warning(f"Could not read cache file {cache_file}: {e}. Re-downloading...")
        except Exception as e:
            log.warning(f"Unexpected error reading cache file: {e}. Re-downloading...")
    else:
        log.info("No instrument cache found. Downloading new list...")

    # 2. If cache is invalid or missing, download from URL
    try:
        log.info(f"Downloading instrument list from Angel One CDN: {INSTRUMENT_LIST_URL}")
        response = requests.get(INSTRUMENT_LIST_URL, timeout=30)
        response.raise_for_status()
        
        instruments = response.json()
        if not (isinstance(instruments, list) and len(instruments) > 0):
            log.error("Downloaded instrument list is empty or invalid format.")
            return None

        log.info(f"Successfully downloaded {len(instruments)} instruments.")
        
        # 3. Save the new list to cache
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(instruments, f)
            log.info(f"Saved new instrument list to cache: {cache_file}")
        except IOError as e:
            log.error(f"Could not write to cache file {cache_file}: {e}")
        
        return instruments
            
    except requests.RequestException as e:
        log.error(f"Network error downloading instrument list: {e}", exc_info=True)
        return None
    except ValueError as e: # Renamed from json.JSONDecodeError
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

            # --- Download/Load and PROCESS the instrument list for fast lookups ---
            # Use the new caching function
            raw_instrument_list = _load_instrument_list(INSTRUMENT_CACHE_FILE, INSTRUMENT_CACHE_TTL_HOURS)
            
            if raw_instrument_list:
                INSTRUMENT_MAP = {}
                # Clear expiry maps for this new list
                global OPTION_EXPIRY_MAP, FUTURES_EXPIRY_MAP
                OPTION_EXPIRY_MAP = {"STK": {}, "IDX": {}}
                FUTURES_EXPIRY_MAP = {"STK": {}, "IDX": {}}

                for item in raw_instrument_list:
                    if not isinstance(item, dict): continue
                    
                    symbol = item.get('symbol')
                    name = item.get('name')
                    exch = item.get('exch_seg')
                    token = item.get('token')
                    expiry = item.get('expiry')
                    inst_type = item.get('instrumenttype')
                    
                    # 1. Build the fast token lookup map
                    if symbol and exch and token:
                        INSTRUMENT_MAP[f"{symbol.upper()}_{exch}"] = item
                    if name and exch and token:
                        # Use name lookup as a fallback (symbols are more unique)
                        key = f"{name.upper()}_{exch}"
                        if key not in INSTRUMENT_MAP:
                            INSTRUMENT_MAP[key] = item

                    # 2. Build the fast expiry lookup maps
                    if not (name and expiry and inst_type and exch == 'NFO'):
                        continue
                    
                    target_map = None

                    if inst_type == 'OPTSTK':
                        target_map = OPTION_EXPIRY_MAP["STK"]
                    elif inst_type == 'OPTIDX':
                        target_map = OPTION_EXPIRY_MAP["IDX"]
                    elif inst_type == 'FUTSTK':
                        target_map = FUTURES_EXPIRY_MAP["STK"]
                    elif inst_type == 'FUTIDX':
                        target_map = FUTURES_EXPIRY_MAP["IDX"]
                    
                    if target_map is not None:
                        if name not in target_map:
                            target_map[name] = set()
                        target_map[name].add(expiry)

                INSTRUMENT_LIST = INSTRUMENT_MAP # This is a map, not a list. Variable name is a bit confusing but we'll keep it.
                
                log.info(f"✅ Successfully processed and cached {len(INSTRUMENT_LIST)} instruments for fast lookups.")
                log.info(f"Pre-processed {len(OPTION_EXPIRY_MAP['STK'])} stocks with options.")
                log.info(f"Pre-processed {len(FUTURES_EXPIRY_MAP['STK'])} stocks with futures.")
                log.info(f"Pre-processed {len(OPTION_EXPIRY_MAP['IDX'])} indices with options.")
                log.info(f"Pre-processed {len(FUTURES_EXPIRY_MAP['IDX'])} indices with futures.")
            else:
                log.warning("⚠️ Could not download or load instrument list. Token lookups and derivative features will fail.")

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
    Fetches historical OHLCV data with a retry mechanism and improved
    index symbol mapping to handle API instability.
    """
    log.info(f"[AO] Fetching '{interval}' historical data for {symbol} from {from_date.date()} to {to_date.date()}...")
    if not smart_api_session and not initialize_angelone_session():
        return None

    # --- Improved Index and Symbol Handling ---
    token = _get_symbol_token(symbol)
    if not token:
        # Map common yfinance index tickers to Angel One names
        index_name_map = {
            "^NSEI": "NIFTY 50",
            "^NSEBANK": "NIFTY BANK",
            "^CNXIT": "NIFTY IT",
            "^CNXAUTO": "NIFTY AUTO",
            "^CNXPHARMA": "NIFTY PHARMA",
            "^CNXFMCG": "NIFTY FMCG",
            "^CNXMETAL": "NIFTY METAL",
            "^CNXENERGY": "NIFTY ENERGY",
            "^CNXREALTY": "NIFTY REALTY",
            "^CNXINFRA": "NIFTY INFRA"
        }
        if symbol in index_name_map:
            log.info(f"Retrying token lookup for index using mapped name: {index_name_map[symbol]}")
            token = _get_symbol_token(index_name_map[symbol])
        else:
            log.warning(f"Final token lookup failed for {symbol}")
            return None

    if not token:
        log.warning(f"Could not resolve token for {symbol} after all attempts.")
        return None

    # --- Retry Logic Implementation ---
    max_retries = 3
    backoff_factor = 2  # Delay will be 2s, 4s, 8s

    for attempt in range(max_retries):
        try:
            params = {
                "exchange": "NSE",
                "symboltoken": token,
                "interval": interval,
                "fromdate": from_date.strftime('%Y-%m-%d 09:15'),
                "todate": to_date.strftime('%Y-%m-%d 15:30')
            }
            raw_data = smart_api_session.getCandleData(params)

            if raw_data and raw_data.get('status') is True and raw_data.get('data'):
                df = pd.DataFrame(raw_data['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['date'] = pd.to_datetime(df['timestamp'])
                df.set_index('date', inplace=True)

                # --- Check for empty or insufficient data ---
                if df.empty or len(df) < 2:
                    log.warning(f"Received insufficient data for {symbol}.")
                    return None

                log.info(f"✅ Successfully fetched {len(df)} candles for {symbol} from Angel One.")
                return df.drop(columns=['timestamp'])

            # Handles cases where status is False or data is empty
            err_msg = raw_data.get('message', 'Unknown API error')
            log.warning(f"[AO Attempt {attempt + 1}/{max_retries}] API error for {symbol}: {err_msg}")

        except Exception as e:
            log.warning(
                f"[AO Attempt {attempt + 1}/{max_retries}] System error for {symbol}: {e}",
                exc_info=(attempt == max_retries - 1)  # Show full traceback only on final failure
            )

        # Wait before retrying if not last attempt
        if attempt < max_retries - 1:
            delay = backoff_factor ** (attempt + 1)
            log.info(f"Waiting for {delay} seconds before retrying...")
            time.sleep(delay)

    log.error(f"❌ Angel One fetch failed for {symbol} after {max_retries} attempts. Triggering fallback.")
    return None  # Return None to trigger the yfinance fallback in data_retriever.py

@rate_limited
def get_full_quote(symbol: str, exchange: str = "NSE", token: str | None = None) -> dict | None:
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

    underlying_symbol = symbol.replace('.NS', '').upper() # Use upper to match map key

    # Step 1: Find all expiry dates using the pre-processed O(1) map
    try:
        expiry_dates = OPTION_EXPIRY_MAP["STK"].get(underlying_symbol)

        if not expiry_dates:
            log.warning(f"No option expiry dates found for {symbol} in the pre-processed map.")
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
        
        # Get risk-free rate from config, default to 5% if not set
        risk_free_rate = getattr(config, 'RISK_FREE_RATE', 0.05)
        
        iv_array = vectorized_implied_volatility(
            price=price_array, S=ltp, K=strike_array, t=time_to_expiry, r=risk_free_rate, flag=flag_array, return_as='numpy', on_error='warn'
        )
        
        # --- FIX: Calculate Greeks individually ---
        greeks = {
            'delta': vectorized_delta(flag=flag_array, S=ltp, K=strike_array, t=time_to_expiry, r=risk_free_rate, sigma=iv_array),
            'gamma': vectorized_gamma(flag=flag_array, S=ltp, K=strike_array, t=time_to_expiry, r=risk_free_rate, sigma=iv_array),
            'theta': vectorized_theta(flag=flag_array, S=ltp, K=strike_array, t=time_to_expiry, r=risk_free_rate, sigma=iv_array)
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

# ===================================================================
# SECTION 4: FUTURES DATA
# ===================================================================

@rate_limited
def get_futures_contract_data(symbol: str, exchange: str = "NFO") -> dict | None:
    """
    Fetches key data (LTP, OI, Volume) for the nearest expiry futures contract.
    """
    log.info(f"[AO] Getting nearest futures contract data for {symbol}...")
    if (not smart_api_session or not INSTRUMENT_LIST) and not initialize_angelone_session():
        return None

    underlying_symbol = symbol.replace('.NS', '').upper() # Use upper to match map key

    # Step 1: Find all futures expiry dates using the pre-processed O(1) map
    try:
        futures_expiries = FUTURES_EXPIRY_MAP["STK"].get(underlying_symbol)

        if not futures_expiries:
            log.warning(f"No futures expiry dates found for {symbol} in the pre-processed map.")
            return None

        # Step 2: Find the nearest expiry date
        today = datetime.now()
        parsed_dates = []
        for d in futures_expiries:
            try:
                # AngelOne expiry format is DDMMMYYYY (e.g., 25JUL2024)
                parsed_dates.append(datetime.strptime(d, '%d%b%Y'))
            except Exception:
                log.debug(f"Ignoring unparsable futures expiry date entry: {d}")

        if not parsed_dates:
            log.warning(f"No parsable futures expiry dates found for {symbol}.")
            return None

        nearest = min(d for d in parsed_dates if d >= today) # Ensure expiry is not in the past
        nearest_expiry_str = nearest.strftime('%d%b%Y').upper()

        log.info(f"Nearest futures expiry date for {symbol} is {nearest_expiry_str}.")

        # Step 3: Find the specific instrument token for this future
        future_token = None
        future_trading_symbol = None
        search_key_prefix = f"{underlying_symbol.upper()}{nearest_expiry_str}" # e.g., RELIANCE25JUL2024

        # Search the INSTRUMENT_LIST dictionary (fast lookup)
        for key, inst in INSTRUMENT_LIST.items():
             # Check if the key starts with our target prefix and is a future
             if (isinstance(inst, dict) and
                 key.startswith(search_key_prefix) and
                 inst.get('instrumenttype') == 'FUTSTK' and
                 inst.get('exch_seg') == exchange and
                 inst.get('expiry') == nearest_expiry_str): # Double check expiry
                 future_token = inst.get('token')
                 future_trading_symbol = inst.get('symbol') # Get the exact symbol like RELIANCE24JULFUT
                 break # Found the first match (should be the only one)

        if not future_token or not future_trading_symbol:
            log.error(f"Could not find instrument token/symbol for {underlying_symbol} future expiring {nearest_expiry_str}.")
            return None

        log.info(f"Found future instrument: Symbol='{future_trading_symbol}', Token='{future_token}'")

        # Step 4: Fetch the quote data using the token
        # Use the specific future symbol for the ltpData call
        quote_data = smart_api_session.ltpData(exchange, future_trading_symbol, future_token)

        if quote_data and quote_data.get('status'):
             data = quote_data.get('data')
             if data:
                 log.info(f"✅ Successfully fetched futures quote for {future_trading_symbol}.")
                 # Extract relevant fields
                 return {
                     "symbol": future_trading_symbol,
                     "expiry_date": nearest_expiry_str,
                     "ltp": data.get('ltp'),
                     "open_interest": data.get('opnInterest'), # Note the capitalization from API
                     "volume": data.get('volume'),
                     "last_traded_time": data.get('exchFeedTime') # Useful for checking freshness
                 }
             else:
                  log.error(f"Empty data received for futures quote {future_trading_symbol}.")
                  return None
        else:
            err_msg = quote_data.get('message') if isinstance(quote_data, dict) else "Unknown API error"
            log.error(f"API error fetching futures quote for {future_trading_symbol}: {err_msg}")
            return None

    except Exception as e:
        log.error(f"Error getting futures contract data for {symbol}: {e}", exc_info=True)
        return None


def get_key_oi_levels(symbol: str, num_levels: int = 3) -> dict | None:
    """
    Fetches the option chain and identifies the top N Call and Put strikes
    with the highest Open Interest for the nearest expiry.
    """
    log.info(f"[AO] Getting Top {num_levels} High OI Strikes for {symbol}...")
    try:
        chain_data = get_option_chain_data(symbol) # Re-use the existing function
        if not chain_data or not chain_data.get('chain'):
            log.warning(f"No option chain data available for {symbol} to find high OI levels.")
            return None

        df = pd.DataFrame(chain_data['chain'])
        required_cols = ['strikePrice', 'optionType', 'openInterest']
        if not all(col in df.columns for col in required_cols):
            log.warning(f"Option chain for {symbol} missing columns for OI analysis.")
            return None

        # Convert to numeric, coercing errors
        df['strikePrice'] = pd.to_numeric(df['strikePrice'], errors='coerce')
        df['openInterest'] = pd.to_numeric(df['openInterest'], errors='coerce')
        df.dropna(subset=['strikePrice', 'openInterest'], inplace=True)

        calls = df[df['optionType'] == 'CE'].nlargest(num_levels, 'openInterest')
        puts = df[df['optionType'] == 'PE'].nlargest(num_levels, 'openInterest')

        high_oi_calls = calls[['strikePrice', 'openInterest']].to_dict('records')
        high_oi_puts = puts[['strikePrice', 'openInterest']].to_dict('records')

        log.info(f"✅ Identified High OI levels for {symbol}. Calls: {[c['strikePrice'] for c in high_oi_calls]}, Puts: {[p['strikePrice'] for p in high_oi_puts]}")
        return {
            "expiry_date": chain_data.get("expiry_date"),
            "underlying_price": chain_data.get("underlying_price"),
            "high_oi_calls": high_oi_calls,
            "high_oi_puts": high_oi_puts
        }

    except Exception as e:
        log.error(f"Error getting key OI levels for {symbol}: {e}", exc_info=True)
        return None

# Convenience function for Indices (Indices use FUTIDX/OPTIDX)
def get_index_futures_data(index_name: str) -> dict | None:
    """
    Gets futures data specifically for indices like NIFTY, BANKNIFTY.
    Uses 'FUTIDX' instrument type and the pre-processed index expiry map.
    """
    log.info(f"[AO] Getting nearest index futures contract data for {index_name}...")
    if (not smart_api_session or not INSTRUMENT_LIST) and not initialize_angelone_session():
        return None

    underlying_name = index_name.upper() # e.g., "NIFTY 50"
    exchange = "NFO"

    # Step 1: Find all futures expiry dates using the pre-processed O(1) map
    try:
        futures_expiries = FUTURES_EXPIRY_MAP["IDX"].get(underlying_name)
        if not futures_expiries:
            # Fallback for common aliases
            if underlying_name == "NIFTY":
                futures_expiries = FUTURES_EXPIRY_MAP["IDX"].get("NIFTY 50")
            elif underlying_name == "BANKNIFTY":
                futures_expiries = FUTURES_EXPIRY_MAP["IDX"].get("NIFTY BANK")

        if not futures_expiries:
            log.warning(f"No futures expiry dates found for index {index_name} in the pre-processed map.")
            return None

        # Step 2: Find the nearest expiry date
        today = datetime.now()
        parsed_dates = []
        for d in futures_expiries:
            try:
                parsed_dates.append(datetime.strptime(d, '%d%b%Y'))
            except Exception:
                log.debug(f"Ignoring unparsable index futures expiry date entry: {d}")

        if not parsed_dates:
            log.warning(f"No parsable index futures expiry dates found for {index_name}.")
            return None

        nearest = min(d for d in parsed_dates if d >= today) # Ensure expiry is not in the past
        nearest_expiry_str = nearest.strftime('%d%b%Y').upper()

        log.info(f"Nearest index futures expiry date for {index_name} is {nearest_expiry_str}.")

        # Step 3: Find the specific instrument token for this future
        future_token = None
        future_trading_symbol = None
        
        # We must iterate here, but only on matches for the *name*, which is still fast
        search_key_prefix = f"{underlying_name.upper()}{nearest_expiry_str}"

        for key, inst in INSTRUMENT_LIST.items():
            if (isinstance(inst, dict) and
                inst.get('name') == underlying_name and # Match the index name
                inst.get('instrumenttype') == 'FUTIDX' and # Match instrument type
                inst.get('exch_seg') == exchange and
                inst.get('expiry') == nearest_expiry_str):
                
                future_token = inst.get('token')
                future_trading_symbol = inst.get('symbol') # Get the exact symbol
                break

        if not future_token or not future_trading_symbol:
            log.error(f"Could not find instrument token/symbol for {underlying_name} index future expiring {nearest_expiry_str}.")
            return None

        log.info(f"Found index future instrument: Symbol='{future_trading_symbol}', Token='{future_token}'")

        # Step 4: Fetch the quote data using the token
        quote_data = smart_api_session.ltpData(exchange, future_trading_symbol, future_token)

        if quote_data and quote_data.get('status') and quote_data.get('data'):
            data = quote_data.get('data')
            log.info(f"✅ Successfully fetched index futures quote for {future_trading_symbol}.")
            return {
                "symbol": future_trading_symbol,
                "expiry_date": nearest_expiry_str,
                "ltp": data.get('ltp'),
                "open_interest": data.get('opnInterest'),
                "volume": data.get('volume'),
                "last_traded_time": data.get('exchFeedTime')
            }
        else:
            err_msg = quote_data.get('message') if isinstance(quote_data, dict) else "Unknown API error"
            log.error(f"API error fetching index futures quote for {future_trading_symbol}: {err_msg}")
            return None

    except Exception as e:
        log.error(f"Error getting index futures contract data for {index_name}: {e}", exc_info=True)
        return None

def get_index_option_chain_data(index_name: str, exchange: str = "NFO") -> dict | None:
    """
    Fetches the nearest expiry date and the full option chain for that date
    for an INDEX (e.g., NIFTY 50, NIFTY BANK).
    """
    log.info(f"[AO] Getting full option chain for INDEX: {index_name}...")
    if (not smart_api_session or not INSTRUMENT_LIST) and not initialize_angelone_session():
        return None

    underlying_name = index_name.upper() # e.g., "NIFTY 50"

    # Step 1: Find all expiry dates using the pre-processed O(1) map
    try:
        expiry_dates = OPTION_EXPIRY_MAP["IDX"].get(underlying_name)
        if not expiry_dates:
            # Fallback for common aliases
            if underlying_name == "NIFTY":
                expiry_dates = OPTION_EXPIRY_MAP["IDX"].get("NIFTY 50")
            elif underlying_name == "BANKNIFTY":
                expiry_dates = OPTION_EXPIRY_MAP["IDX"].get("NIFTY BANK")
        
        if not expiry_dates:
            log.warning(f"No option expiry dates found for index {index_name} in the pre-processed map.")
            return None

        # Step 2: Find the nearest expiry date
        today = datetime.now()
        parsed_dates = []
        for d in expiry_dates:
            try:
                parsed_dates.append(datetime.strptime(d, '%d%b%Y'))
            except Exception:
                log.debug(f"Ignoring unparsable index option expiry date entry: {d}")

        if not parsed_dates:
            log.warning(f"No parsable index option expiry dates found for {index_name}.")
            return None

        nearest = min(d for d in parsed_dates if d >= today.replace(hour=0, minute=0, second=0))
        nearest_expiry = nearest.strftime('%d%b%Y').upper()

        log.info(f"Nearest index option expiry date for {index_name} is {nearest_expiry}.")

    except Exception as e:
        log.error(f"Error finding expiry date for index {index_name}: {e}", exc_info=True)
        return None

    # Step 3: Fetch the option chain for the nearest expiry
    try:
        # Use the underlying_name for the API call
        chain_data = smart_api_session.getOptionChain(exchange, underlying_name, nearest_expiry)

        if chain_data and chain_data.get('status') and chain_data.get('data'):
            log.info(f"✅ Successfully fetched option chain for index {index_name} on {nearest_expiry}.")
            return {
                "underlying_price": chain_data['data'].get('underlyingValue'),
                "expiry_date": nearest_expiry,
                "chain": chain_data['data'].get('options', [])
            }
        else:
            err_msg = chain_data.get('message') if isinstance(chain_data, dict) else "Unknown API error"
            log.error(f"API error fetching option chain for index {index_name}: {err_msg}")
            return None

    except Exception as e:
        log.error(f"Error in get_index_option_chain_data for {index_name}: {e}", exc_info=True)
        return None

def get_index_key_oi_levels(index_name: str, num_levels: int = 5) -> dict | None:
    """
    Gets key OI levels specifically for indices by calling the
    new index-specific option chain function.
    """
    log.info(f"[AO] Getting Top {num_levels} High OI Strikes for INDEX: {index_name}...")
    try:
        # Call the new index-specific function
        chain_data = get_index_option_chain_data(index_name) 
        if not chain_data or not chain_data.get('chain'):
            log.warning(f"No option chain data available for index {index_name} to find high OI levels.")
            return None

        df = pd.DataFrame(chain_data['chain'])
        required_cols = ['strikePrice', 'optionType', 'openInterest']
        if not all(col in df.columns for col in required_cols):
            log.warning(f"Index option chain for {index_name} missing columns for OI analysis.")
            return None

        # Convert to numeric, coercing errors
        df['strikePrice'] = pd.to_numeric(df['strikePrice'], errors='coerce')
        df['openInterest'] = pd.to_numeric(df['openInterest'], errors='coerce')
        df.dropna(subset=['strikePrice', 'openInterest'], inplace=True)

        calls = df[df['optionType'] == 'CE'].nlargest(num_levels, 'openInterest')
        puts = df[df['optionType'] == 'PE'].nlargest(num_levels, 'openInterest')

        high_oi_calls = calls[['strikePrice', 'openInterest']].to_dict('records')
        high_oi_puts = puts[['strikePrice', 'openInterest']].to_dict('records')

        log.info(f"✅ Identified High OI levels for index {index_name}. Calls: {[c['strikePrice'] for c in high_oi_calls]}, Puts: {[p['strikePrice'] for p in high_oi_puts]}")
        return {
            "expiry_date": chain_data.get("expiry_date"),
            "underlying_price": chain_data.get("underlying_price"),
            "high_oi_calls": high_oi_calls,
            "high_oi_puts": high_oi_puts
        }

    except Exception as e:
        log.error(f"Error getting key OI levels for index {index_name}: {e}", exc_info=True)
        return None