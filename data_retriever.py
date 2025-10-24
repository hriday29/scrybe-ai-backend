# data_retriever.py 
# This file serves as the single, stable data gateway for the entire application.
# It uses a control switch in config.py to decide whether to fetch data from
# the new, high-fidelity Angel One API or fall back to the yfinance API.

import yfinance as yf
import pandas as pd
import requests
import index_manager
import json
import os
from datetime import datetime, timezone, timedelta
import database_manager
import time

from logger_config import log
import config  # Reads the new DATA_SOURCE flag

# --- NEW: Import the Angel One retriever module ---
# This module contains the specialized logic for the Angel One API.
import angelone_retriever

# --- Initialize Angel One Session on Startup ---
# This ensures the API is logged in and ready when the app starts.
if config.DATA_SOURCE == "angelone":
    log.info("Data source is Angel One. Initializing API session...")
    angelone_retriever.initialize_angelone_session()
# --------------------------------------------------

# --- Caching Setup (Remains the same) ---
CACHE_DIR = 'cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
CACHE_FILE = os.path.join(CACHE_DIR, 'sector_cache.json')
DATA_CACHE_DIR = 'data_cache'
if not os.path.exists(DATA_CACHE_DIR):
    os.makedirs(DATA_CACHE_DIR)
# -------------------------------------------

def load_sector_cache():
    """
    Safely loads the sector cache. If the file is corrupt or missing,
    it returns an empty dictionary. (Unchanged)
    """
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            log.warning("Could not decode sector_cache.json. It might be corrupt. Starting fresh.")
            return {}
    return {}

def save_sector_cache(cache):
    """
    Saves the sector cache, but only if it contains a reasonable number
    of tickers, preventing incomplete saves. (Unchanged)
    """
    if isinstance(cache, dict) and len(cache) > 40:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=4)
        log.info(f"Successfully saved a valid sector cache with {len(cache)} entries.")
    else:
        log.warning(f"Skipped saving sector cache because it was incomplete (had {len(cache)} entries).")

def get_historical_stock_data(ticker_symbol: str, end_date=None):
    """
    Fetches historical daily OHLC stock data with a robust file-caching system
    and a new Angel One -> yfinance fallback mechanism.
    """
    # --- MODIFIED: Normalize ticker for consistent caching and API calls ---
    clean_ticker = ticker_symbol.replace('.NS', '')
    
    source_suffix = "AO" if config.DATA_SOURCE == "angelone" else "YF"
    date_suffix = end_date.replace('-', '') if end_date else 'live'
    # Use the clean ticker for a consistent cache file name
    cache_file = os.path.join(DATA_CACHE_DIR, f"{clean_ticker}_{date_suffix}_{source_suffix}.feather")

    # --- Step 1: Check cache (No changes here) ---
    if os.path.exists(cache_file):
        try:
            log.info(f"CACHE HIT ({source_suffix}): Loading {ticker_symbol} from {cache_file}")
            df = pd.read_feather(cache_file)
            index_col = 'date' if 'date' in df.columns else 'Date'
            df[index_col] = pd.to_datetime(df[index_col])
            df.set_index(index_col, inplace=True)
            return df
        except Exception as e:
            log.warning(f"Could not read from cache file {cache_file}. Refetching. Error: {e}")

    # --- Step 2: If no cache, fetch data with new fallback logic ---
    log.info(f"CACHE MISS ({source_suffix}): Fetching data for {ticker_symbol} from API...")
    
    df = None
    data_source_used = "N/A"

    # --- MODIFIED: Implement Angel One with yfinance fallback ---
    if config.DATA_SOURCE == "angelone":
        try:
            log.info(f"Attempting to fetch {clean_ticker} from Angel One...")
            to_date_obj = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
            from_date_obj = to_date_obj - timedelta(days=5*365)
            to_date_str = to_date_obj.strftime('%Y-%m-%d')
            from_date_str = from_date_obj.strftime('%Y-%m-%d')
            
            # Use the clean_ticker for the API call
            df = angelone_retriever.get_historical_data(clean_ticker, from_date_obj, to_date_obj, "ONE_DAY")
            
            if df is not None and not df.empty:
                log.info(f"✅ Successfully fetched data for {clean_ticker} from Angel One.")
                data_source_used = "AngelOne"
            else:
                # This 'else' catches cases where the API call succeeds but returns no data.
                log.warning(f"Angel One returned no data for {clean_ticker}. Triggering fallback.")
                df = None # Ensure df is None to proceed to fallback
        except Exception as e:
            log.warning(f"Angel One API failed for {clean_ticker}: {e}. Triggering fallback to yfinance.")
            df = None # Ensure df is None to proceed to fallback

    # --- MODIFIED: This block now serves as the primary yfinance path AND the Angel One fallback ---
    if df is None:
        try:
            log.info(f"Attempting to fetch {ticker_symbol} from yfinance (as primary or fallback)...")
            ticker = yf.Ticker(ticker_symbol)
            # Fetch 5 years of data, which is a good standard
            df = ticker.history(period="5y", end=end_date)
            if df.empty:
                 log.warning(f"yfinance returned no data for {ticker_symbol}.")
            else:
                df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                df.index.name = 'Date' # Standardize index name
                log.info(f"✅ Successfully fetched data for {ticker_symbol} from yfinance.")
                data_source_used = "yfinance"
        except Exception as e:
            log.error(f"yfinance fetch also failed for {ticker_symbol}: {e}")
            return None

    # --- Step 3: Cache the successfully fetched data (No changes here) ---
    if df is not None and not df.empty:
        try:
            df.index = df.index.tz_localize(None)
            df.reset_index().to_feather(cache_file)
            log.info(f"CACHE WRITE ({data_source_used}): Saved {ticker_symbol} data to {cache_file}")
        except Exception as e:
            log.warning(f"Failed to write to cache file {cache_file}. Error: {e}")
    else:
        log.warning(f"No historical data could be retrieved for {ticker_symbol} from any source.")

    return df

def get_live_financial_data(ticker_symbol: str):
    """
    MODIFIED: Fetches curated live financial data using a hybrid approach.
    It gets the precise LTP from Angel One and enriches it with deep
    fundamental data from yfinance.
    """
    if config.DATA_SOURCE == "angelone":
        log.info(f"Fetching hybrid financial data for {ticker_symbol} (AO Price + YF Fundamentals)...")
        try:
            # 1. Get high-fidelity live price from Angel One
            quote = angelone_retriever.get_full_quote(ticker_symbol)
            live_price = quote.get("ltp") if quote else None

            # 2. Get rich fundamental data from yfinance
            raw_info = yf.Ticker(ticker_symbol).info

            # 3. Combine them for the best of both worlds
            curated_data = {
                "currentPrice": live_price or raw_info.get("currentPrice"), # Fallback to yf price if AO fails
                "trailingPE": raw_info.get("trailingPE"),
                "priceToBook": raw_info.get("priceToBook"),
                "profitMargins": raw_info.get("profitMargins"),
                "heldPercentInstitutions": raw_info.get("heldPercentInstitutions"),
                "returnOnEquity": raw_info.get("returnOnEquity"),
                "debtToEquity": raw_info.get("debtToEquity"),
                "totalCash": raw_info.get("totalCash"),
            }
            if curated_data["currentPrice"] is None: return None
            
            # The raw data sheet can be a combination of both sources
            raw_data_sheet = {**(quote or {}), **(raw_info or {})}
            return {"curatedData": curated_data, "rawDataSheet": raw_data_sheet}

        except Exception as e:
            log.error(f"Error fetching hybrid financial data for {ticker_symbol}: {e}")
            return None
    else:
        # --- Original yfinance logic (remains as the fallback) ---
        log.info(f"Fetching financial data for {ticker_symbol} from yfinance...")
        try:
            raw_info = yf.Ticker(ticker_symbol).info
            curated_data = {
                "currentPrice": raw_info.get("currentPrice", raw_info.get("previousClose")),
                "trailingPE": raw_info.get("trailingPE"), "priceToBook": raw_info.get("priceToBook"),
                "profitMargins": raw_info.get("profitMargins"), "heldPercentInstitutions": raw_info.get("heldPercentInstitutions"),
                "returnOnEquity": raw_info.get("returnOnEquity"), "debtToEquity": raw_info.get("debtToEquity"),
                "totalCash": raw_info.get("totalCash"),
            }
            if curated_data["currentPrice"] is None: return None
            return {"curatedData": curated_data, "rawDataSheet": raw_info}
        except Exception as e:
            log.error(f"Error fetching yfinance live financial data for {ticker_symbol}: {e}")
            return None

def get_options_data(ticker_symbol: str):
    """
    MODIFIED: Fetches and processes options chain data for a STOCK.
    Now also fetches and attaches option greeks from Angel One.
    """
    if config.DATA_SOURCE == "angelone":
        log.info(f"Fetching options data for {ticker_symbol} from Angel One...")
        try:
            # 1. Call our unified function from angelone_retriever
            chain_data = angelone_retriever.get_option_chain_data(ticker_symbol)
            if not chain_data or 'chain' not in chain_data or not chain_data['chain']:
                log.warning(f"No option chain data returned from Angel One for {ticker_symbol}.")
                return None

            # 2. Convert to DataFrame and ensure numeric columns
            df = pd.DataFrame(chain_data['chain'])
            for col in ['openInterest', 'volume', 'strikePrice']:
                if col not in df.columns:
                    log.error(f"Missing required column '{col}' in Angel One option chain data for {ticker_symbol}.")
                    return None
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(inplace=True)

            calls = df[df['optionType'] == 'CE']
            puts = df[df['optionType'] == 'PE']

            if calls.empty or puts.empty:
                log.warning(f"Incomplete option chain for {ticker_symbol} (missing calls or puts).")
                return None

            # 3. Perform calculations
            pcr_oi = puts['openInterest'].sum() / calls['openInterest'].sum() if calls['openInterest'].sum() > 0 else 0
            max_oi_call_strike = calls.loc[calls['openInterest'].idxmax()]['strikePrice']
            max_oi_put_strike = puts.loc[puts['openInterest'].idxmax()]['strikePrice']

            # --- NEW: Fetch and attach option greeks ---
            try:
                greeks = angelone_retriever.get_option_greeks(ticker_symbol)
            except Exception as ge:
                log.warning(f"Failed to fetch option greeks for {ticker_symbol}: {ge}")
                greeks = None

            return {
                "put_call_ratio_oi": round(pcr_oi, 2),
                "max_oi_call_strike": max_oi_call_strike,
                "max_oi_put_strike": max_oi_put_strike,
                "greeks_analysis": greeks  # <-- Attach the new data
            }

        except Exception as e:
            log.error(f"Error processing Angel One options data for {ticker_symbol}: {e}", exc_info=True)
            return None

    else:
        # --- Original yfinance logic (remains as fallback) ---
        try:
            log.warning("Options greeks are not available with the yfinance fallback.")
            ticker = yf.Ticker(ticker_symbol)
            if not ticker.options:
                return None

            opt = ticker.option_chain(ticker.options[0])
            calls, puts = opt.calls, opt.puts

            pcr_volume = puts['volume'].sum() / calls['volume'].sum() if calls['volume'].sum() > 0 else 0
            pcr_oi = puts['openInterest'].sum() / calls['openInterest'].sum() if calls['openInterest'].sum() > 0 else 0
            max_oi_call_strike = calls.loc[calls['openInterest'].idxmax()]['strike']
            max_oi_put_strike = puts.loc[puts['openInterest'].idxmax()]['strike']

            atm_calls = calls.iloc[(calls['strike'] - calls['lastPrice'].iloc[-1]).abs().argsort()[:5]]
            atm_puts = puts.iloc[(puts['strike'] - puts['lastPrice'].iloc[-1]).abs().argsort()[:5]]
            avg_iv = pd.concat([atm_calls['impliedVolatility'], atm_puts['impliedVolatility']]).mean()

            return {
                "put_call_ratio_volume": round(pcr_volume, 2),
                "put_call_ratio_oi": round(pcr_oi, 2),
                "max_oi_call_strike": max_oi_call_strike,
                "max_oi_put_strike": max_oi_put_strike,
                "average_iv": f"{avg_iv:.2%}"
            }

        except Exception as e:
            log.warning(f"Could not retrieve yfinance options data for {ticker_symbol}. Error: {e}")
            return None


def get_intraday_data(ticker_symbol: str):
    """MODIFIED: Fetches recent 15-minute intraday data for a stock."""
    if config.DATA_SOURCE == "angelone":
        to_date = datetime.now()
        from_date = to_date - timedelta(days=2) # Fetch last 2 days of data
        return angelone_retriever.get_historical_data(
                ticker_symbol,
                from_date,    # Pass the datetime object
                to_date,      # Pass the datetime object
                "FIFTEEN_MINUTE"
            )
    else:
        try:
            log.info(f"[yfinance] Getting 15-min intraday data for {ticker_symbol}...")
            df = yf.Ticker(ticker_symbol).history(period="2d", interval="15m")
            if df.empty: return None
            df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
            return df
        except Exception as e:
            log.error(f"Error fetching yfinance intraday data for {ticker_symbol}: {e}")
            return None

def get_stored_fundamentals(ticker: str, point_in_time: datetime = None) -> dict:
    """
    Fetches curated fundamental data from the 'fundamentals' MongoDB collection,
    ensuring it's the latest data available as of a specific point in time.
    """
    if database_manager.db is None:
        log.warning("Cannot fetch fundamentals, DB not initialized.")
        return {"error": "Database not initialized."}
    
    try:
        # If no date is provided, default to now (for live runs)
        if point_in_time is None:
            point_in_time = datetime.now(timezone.utc)
        
        # Ensure the timestamp is timezone-aware for correct comparison in MongoDB
        if point_in_time.tzinfo is None:
            point_in_time = point_in_time.replace(tzinfo=timezone.utc)

        fundamentals_collection = database_manager.db.fundamentals
        
        # Query for the most recent document ON OR BEFORE the point_in_time
        query = {
            'ticker': ticker,
            'asOfDate': {'$lte': point_in_time}
        }
        
        # Sort by date descending and get the first result
        data = fundamentals_collection.find_one(query, sort=[('asOfDate', -1)])
        
        if data:
            data.pop('_id', None) # Remove the MongoDB-specific ID
            return data
        else:
            return {"status": f"Data not found for {ticker} on or before {point_in_time.date()}."}
            
    except Exception as e:
        log.error(f"Error fetching stored fundamentals for {ticker}: {e}")
        return {"error": str(e)}

def get_live_market_depth(ticker: str) -> dict:
    """Fetches live market depth from Angel One and calculates Order Book Imbalance."""
    if config.DATA_SOURCE != "angelone":
        return {"status": "Market depth only available from Angel One."}
    
    try:
        depth_data = angelone_retriever.get_market_depth(ticker)
        if not depth_data or 'bids' not in depth_data or 'asks' not in depth_data:
            return {"status": "Incomplete market depth data."}

        total_bid_qty = sum(item['quantity'] for item in depth_data['bids'])
        total_ask_qty = sum(item['quantity'] for item in depth_data['asks'])

        obi_ratio = total_bid_qty / total_ask_qty if total_ask_qty > 0 else float('inf')

        return {
            "total_bid_quantity": total_bid_qty,
            "total_ask_quantity": total_ask_qty,
            "order_book_imbalance": round(obi_ratio, 2)
        }
    except Exception as e:
        log.error(f"Error fetching market depth for {ticker}: {e}")
        return {"error": str(e)}
    
# ===================================================================
# UNCHANGED FUNCTIONS
# These functions are either placeholders or are better served by yfinance/NewsAPI
# for broad market data, which is not the specialty of a broker API.
# ===================================================================

def get_nifty50_performance():
    """
    Fetches Nifty 50 performance using a resilient "Hybrid Fetch" strategy.
    This version includes robust logging and logic to prevent count anomalies.
    """
    log.info("Fetching Nifty 50 performance with Hybrid Fetch strategy...")
    nifty50_tickers = index_manager.get_nifty50_tickers()
    if not nifty50_tickers:
        return None

    try:
        log.info("Pass 1: Attempting fast batch download...")
        batch_data = yf.download(tickers=nifty50_tickers, period="5d", interval="1d", progress=False)
        
        if batch_data.empty:
            log.warning("The initial batch download returned an empty dataframe. This may happen on non-trading days.")
            return None

        successful_tickers = batch_data['Close'].dropna(how='all', axis=1).columns.tolist()
        missing_tickers = list(set(nifty50_tickers) - set(successful_tickers))
        all_data_frames = [batch_data]

        if missing_tickers:
            log.warning(f"Pass 2: {len(missing_tickers)} tickers failed in batch. Attempting individual fallback...")
            for ticker in missing_tickers:
                try:
                    individual_data = yf.download(tickers=ticker, period="2d", interval="1d", progress=False)
                    if not individual_data.empty:
                        all_data_frames.append(individual_data)
                        log.info(f"Rescued data for {ticker}")
                except Exception:
                    log.error(f"Failed to rescue data for {ticker}")
        
        if not all_data_frames:
            log.error("Failed to download any Nifty 50 price data.")
            return None

        combined_data = pd.concat(all_data_frames, axis=1)
        closing_prices = combined_data['Close'].dropna(how='all', axis=1).tail(2)

        if len(closing_prices) < 2:
            log.warning("Not enough historical data (less than 2 days) to calculate performance.")
            return None

        valid_tickers = closing_prices.columns[closing_prices.notna().all()]
        if not valid_tickers.any():
            log.warning("No stocks with valid 2-day price data found.")
            return None
        
        performance_df = closing_prices[valid_tickers].iloc[-1].to_frame('lastPrice').copy()
        performance_df['pctChange'] = ((closing_prices[valid_tickers].iloc[-1] - closing_prices[valid_tickers].iloc[0]) / closing_prices[valid_tickers].iloc[0]) * 100
        
        log.info(f"Successfully calculated performance for {len(performance_df)} stocks.")

        # --- CORRECTED SECTOR FETCHING LOGIC ---
        sector_cache = load_sector_cache()
        
        # Explicitly create the list of tickers to check from the performance dataframe
        tickers_in_scope = performance_df.index.tolist()
        log.info(f"Cross-referencing {len(tickers_in_scope)} tickers against the sector cache.")

        tickers_to_fetch = [ticker for ticker in tickers_in_scope if ticker not in sector_cache]
        
        if tickers_to_fetch:
            log.info(f"Found {len(tickers_to_fetch)} new/uncached tickers. Fetching their sectors...")
            
            fetched_sectors = {}
            for ticker in tickers_to_fetch:
                try:
                    info = yf.Ticker(ticker).info
                    sector = info.get('sector', 'Other')
                    fetched_sectors[ticker] = sector
                except Exception:
                    fetched_sectors[ticker] = 'Other'
            
            sector_cache.update(fetched_sectors)
            save_sector_cache(sector_cache)
        else:
            log.info("All ticker sectors were found in the local cache.")

        performance_df['sector'] = performance_df.index.map(sector_cache).fillna('Other')
        # --- END OF CORRECTION ---

        performance_df.dropna(inplace=True)
        sector_performance = performance_df.groupby('sector')['pctChange'].mean().to_dict()

        log.info(f"Successfully processed and categorized performance for {len(performance_df)} stocks.")
        return{"stock_performance": performance_df.to_dict('index'), "sector_performance": sector_performance}

    except Exception as e:
        log.error(f"An error occurred while fetching Nifty 50 performance: {e}")
        return None

def get_upcoming_events(ticker_symbol: str):
    """
    Fetches upcoming events from a more robust, manually curated list of
    recurring market-wide events.
    """
    log.info(f"Compiling upcoming events for {ticker_symbol}...")
    all_events = []
    today = pd.Timestamp.now()
    major_economic_events = config.MAJOR_ECONOMIC_EVENTS
    for evt in major_economic_events:
        event_date = pd.to_datetime(evt['date'])
        # The logic to check if the event is in the future remains the same
        if today <= event_date < today + pd.DateOffset(months=3):
            all_events.append(evt)

    # The yfinance part for company-specific events remains
    try:
        ticker = yf.Ticker(ticker_symbol)
        calendar_data = ticker.calendar
        if isinstance(calendar_data, pd.DataFrame) and not calendar_data.empty:
            if 'Earnings Date' in calendar_data.index:
                earnings_date_start = calendar_data.loc['Earnings Date', 0]
                if pd.notna(earnings_date_start):
                    event_text = f"Next Earnings Announcement around {earnings_date_start.strftime('%b %d, %Y')}"
                    all_events.append({"date": earnings_date_start.strftime('%Y-%m-%d'), "event": event_text})
    except Exception as e:
        log.warning(f"Could not fetch company-specific calendar events for {ticker_symbol}. Error: {e}")

    if all_events:
        all_events.sort(key=lambda x: x['date'])
        log.info(f"Successfully compiled a total of {len(all_events)} upcoming events.")
    else:
        log.warning(f"No upcoming events found for {ticker_symbol}.")
    
    return all_events

def get_benchmarks_data(period: str = "1y", end_date=None):
    """
    Fetches historical closing prices for all benchmark assets.
    Accepts an optional 'end_date' for historical point-in-time analysis.
    """
    log.info("Fetching historical data for all benchmarks...")
    if end_date:
        log.info(f"--- Using historical end_date: {end_date} ---")

    benchmark_tickers = {
        "Nifty50": "^NSEI", "USD-INR": "INR=X", "S&P 500": "^GSPC",
        "Dow Jones": "^DJI", "Nikkei 225": "^N225", "Crude Oil": "CL=F", "Gold": "GC=F"
    }
    try:
        data = yf.download(
            tickers=list(benchmark_tickers.values()), 
            period=period, 
            interval="1d", 
            progress=False,
            end=end_date
        )
        
        if data.empty:
            log.warning("Failed to download any benchmark data.")
            return None
            
        closing_prices = data['Close']
        closing_prices.rename(columns={v: k for k, v in benchmark_tickers.items()}, inplace=True)
        log.info("Successfully fetched benchmark data.")
        return closing_prices
        
    except Exception as e:
        log.error(f"An error occurred while fetching benchmark data: {e}")
        return None

def get_index_option_data(index_ticker: str) -> dict:
    """
    MODIFIED: Fetches key option chain data for a given index,
    now using Angel One as the primary source if configured.
    """
    log.info(f"Fetching option chain data for index {index_ticker}...")
    
    if config.DATA_SOURCE == "angelone":
        try:
            # Map the yfinance ticker (e.g., ^NSEI) to the Angel One name (e.g., NIFTY 50)
            index_name_map = {
                "^NSEI": "NIFTY 50",
                "^NSEBANK": "NIFTY BANK"
                # Add other mappings as needed
            }
            index_name = index_name_map.get(index_ticker, index_ticker)

            # Call the fast, pre-processed function from angelone_retriever
            chain_data = angelone_retriever.get_index_option_chain_data(index_name)
            
            if not chain_data or not chain_data.get('chain'):
                 log.warning(f"Angel One returned no chain data for index {index_name}.")
                 # Do not return None yet, try yfinance as a fallback
                 raise Exception("No chain data from Angel One")

            df = pd.DataFrame(chain_data['chain'])
            for col in ['openInterest', 'strikePrice', 'optionType']:
                if col not in df.columns:
                    log.error(f"Missing required column '{col}' in Angel One index option chain for {index_name}.")
                    raise Exception(f"Missing column {col} from Angel One")
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.dropna(subset=['openInterest', 'strikePrice'], inplace=True)

            calls = df[df['optionType'] == 'CE']
            puts = df[df['optionType'] == 'PE']

            if calls.empty or puts.empty or calls['openInterest'].sum() == 0:
                log.warning(f"Incomplete option chain for index {index_name} (missing calls, puts, or OI).")
                raise Exception("Incomplete chain data from Angel One")

            pcr = puts['openInterest'].sum() / calls['openInterest'].sum()
            max_oi_call_strike = calls.loc[calls['openInterest'].idxmax()]['strikePrice']
            max_oi_put_strike = puts.loc[puts['openInterest'].idxmax()]['strikePrice']

            log.info(f"✅ Successfully calculated option metrics for index {index_name} from Angel One.")
            return {
                "pcr": round(pcr, 2),
                "max_oi_call": max_oi_call_strike,
                "max_oi_put": max_oi_put_strike
            }

        except Exception as e:
            log.warning(f"Angel One index options failed ({e}). Falling back to yfinance...")
            # Fallthrough to yfinance logic
    
    # --- yfinance logic (as fallback or primary) ---
    try:
        log.info(f"Using yfinance for index option data for {index_ticker}...")
        ticker = yf.Ticker(index_ticker)
        
        if not ticker.options:
            log.warning(f"No options chain data found for {index_ticker} on yfinance.")
            return None

        opt_chain = ticker.option_chain(ticker.options[0])
        calls, puts = opt_chain.calls, opt_chain.puts
        
        if calls.empty or puts.empty or calls['openInterest'].sum() == 0:
            log.warning(f"Incomplete yfinance option chain for index {index_ticker}.")
            return None

        pcr = puts['openInterest'].sum() / calls['openInterest'].sum()
        max_oi_call_strike = calls.loc[calls['openInterest'].idxmax()]['strike']
        max_oi_put_strike = puts.loc[puts['openInterest'].idxmax()]['strike']

        log.info(f"Successfully calculated option metrics for {index_ticker} from yfinance.")
        return {
            "pcr": round(pcr, 2),
            "max_oi_call": max_oi_call_strike,
            "max_oi_put": max_oi_put_strike
        }
    except Exception as e:
        log.warning(f"yfinance could not fetch or process option chain data for {index_ticker}. Error: {e}")
        return None
    
def get_social_sentiment(ticker_symbol: str):
    """
    !! PLACEHOLDER FUNCTION !!
    Simulates fetching and analyzing sentiment from social media (e.g., X/Twitter).
    In a real system, this would use a library like snscrape or a dedicated API.
    """
    log.info(f"[FETCH] Getting social media sentiment for {ticker_symbol}...")
    # This mock response assumes neutral sentiment.
    # A real function would analyze recent posts for bearish/bullish keywords.
    return {
        "sentiment": "Neutral", 
        "source": "X/Twitter (Simulated)"
    }

def get_news_articles_for_ticker(ticker_symbol: str, company_info: dict = None) -> dict:
    """
    Fetches news with a revamped, precise query, a robust fallback system,
    and a crucial post-fetch filtering step to ensure relevance.
    """
    log.info(f"[FETCH] Running FOCUSED news fetch for {ticker_symbol}...")
    
    try:
        info = company_info if company_info is not None else yf.Ticker(ticker_symbol).info
        long_name = info.get('longName', ticker_symbol.split('.')[0])
        
        # 1. Sanitize the name to get the core company identity.
        # This removes generic suffixes for better queries and filtering.
        suffixes_to_remove = ['Limited', 'Ltd.', 'Ltd', 'Inc.', 'Inc', 'Corporation', 'Corp.']
        clean_name = long_name
        for suffix in suffixes_to_remove:
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)].strip() # "Hindalco Industries Limited" -> "Hindalco Industries"

        # 2. Get the primary name for filtering titles (most important word).
        primary_name_for_filter = clean_name.split(' ')[0].replace('.', '') # "Hindalco"
        log.info(f"Will filter news results for titles containing '{primary_name_for_filter}'.")

        # 3. Create a more robust query for NewsAPI using the cleaned name.
        precise_query = f'"{clean_name}"'

    except Exception as e:
        log.error(f"Failed to get company info for {ticker_symbol} via yfinance: {e}")
        # Fallback to a simple name if yfinance fails
        primary_name_for_filter = ticker_symbol.split('.')[0]
        precise_query = f'"{primary_name_for_filter}"'

    # --- Attempt 1: NewsAPI ---
    try:
        if not config.NEWSAPI_API_KEY:
            raise ValueError("NewsAPI key not configured.")
        
        log.info(f"Using precise query: {precise_query} for NewsAPI search.")
        
        url = (f"https://newsapi.org/v2/everything?q={precise_query}"
               "&language=en&sortBy=publishedAt&pageSize=20" # Fetch more (20) to have a better chance after filtering
               f"&apiKey={config.NEWSAPI_API_KEY}")
        
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        articles = data.get('articles', [])

        # --- CRUCIAL FILTERING STEP ---
        relevant_articles = [
            a for a in articles 
            if primary_name_for_filter.lower() in a.get('title', '').lower()
        ]
        
        if relevant_articles:
            log.info(f"✅ NewsAPI success: Found {len(articles)} raw articles, returning {len(relevant_articles)} after relevance filtering.")
            formatted_articles = [{
                "title": a.get("title"), "url": a.get("url"),
                "source_name": a.get("source", {}).get("name"),
                "published_at": a.get("publishedAt"), "description": a.get("description"),
            } for a in relevant_articles[:5]] # Return the top 5 relevant articles
            return {"type": "Market News (NewsAPI)", "articles": formatted_articles}
        
        log.warning(f"⚠️ NewsAPI returned 0 relevant articles for '{primary_name_for_filter}'. Attempting fallback...")

    except Exception as e:
        log.warning(f"❌ NewsAPI fetch failed for {ticker_symbol}. Attempting fallback... Error: {e}")

    # --- Attempt 2: yfinance Fallback with Filtering ---
    try:
        log.info(f"-> Fallback: Trying yfinance.news for {ticker_symbol}")
        yf_news = yf.Ticker(ticker_symbol).news
        if yf_news:
            # --- CRUCIAL FILTERING STEP FOR FALLBACK ---
            relevant_articles = [
                item for item in yf_news
                if primary_name_for_filter.lower() in item.get('title', '').lower()
            ]

            if relevant_articles:
                log.info(f"✅ yfinance Fallback success: Found {len(yf_news)} raw, returning {len(relevant_articles)} after filtering.")
                formatted_articles = [{
                    "title": item.get('title'), "url": item.get('link'),
                    "source_name": item.get('publisher'),
                    "published_at": datetime.fromtimestamp(item.get('providerPublishTime')).isoformat() if item.get('providerPublishTime') else None,
                    "description": None
                } for item in relevant_articles[:8]] # Return top 8 relevant articles
                return {"type": "Market News (Yahoo Finance)", "articles": formatted_articles}

        log.warning(f"⚠️ yfinance Fallback also found 0 relevant articles for {ticker_symbol}")
        return {"type": "No Relevant News Found", "articles": []}

    except Exception as e:
        log.error(f"❌ yfinance Fallback also failed for {ticker_symbol}: {e}")
        return {"type": "News Fetch Error", "articles": []}
    
def get_stock_sector_map(tickers: list[str]) -> dict:
    """
    Efficiently builds a stock-to-sector map using a cache-first approach.
    """
    log.info("Building stock-to-sector map...")
    sector_cache = load_sector_cache()
    stock_sector_map = {}
    
    # First, load all cached tickers into the map
    for ticker in tickers:
        if ticker in sector_cache:
            stock_sector_map[ticker] = sector_cache[ticker]

    # Then, identify which tickers are missing from the cache
    tickers_to_fetch = [t for t in tickers if t not in sector_cache]

    if tickers_to_fetch:
        log.info(f"Fetching sector info for {len(tickers_to_fetch)} new/uncached tickers...")
        for i, ticker in enumerate(tickers_to_fetch):
            try:
                info = yf.Ticker(ticker).info
                sector = info.get('sector', 'Other')
                stock_sector_map[ticker] = sector
                sector_cache[ticker] = sector # Update cache for next time
                log.info(f"({i+1}/{len(tickers_to_fetch)}) Fetched {ticker}: {sector}")
                time.sleep(0.5) # Small delay to be polite to yfinance API
            except Exception:
                stock_sector_map[ticker] = 'Other'
        save_sector_cache(sector_cache) # Save the updated cache to disk

    log.info("✅ Stock-to-sector map complete.")
    return stock_sector_map