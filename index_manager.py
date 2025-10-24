# index_manager.py
import pandas as pd
import requests
from io import StringIO
import config
from logger_config import log

def get_nifty50_tickers():
    """
    Fetches the Nifty 50 constituents list using a resilient hybrid strategy.
    It first attempts to fetch the live list from NSE. If it fails or returns
    invalid data, it falls back to the reliable, hardcoded list in the config.
    """
    log.info("Attempting to fetch live Nifty 50 constituents list...")
    
    url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Will raise an error for bad status codes (4xx or 5xx)

        csv_data = StringIO(response.content.decode('utf-8'))
        df = pd.read_csv(csv_data)
        
        # Ensure the 'Symbol' column exists before processing
        if 'Symbol' not in df.columns:
            raise ValueError("CSV format is invalid, 'Symbol' column not found.")
            
        tickers = (df['Symbol'] + '.NS').tolist()
        
        # Sanity check: If the list is unusually small, something is wrong.
        if len(tickers) > 40:
            log.info(f"✅ Successfully fetched {len(tickers)} live tickers from NSE.")
            return tickers
        else:
            log.warning(f"Live ticker fetch returned an incomplete list ({len(tickers)} tickers). Using fallback.")
            return config.NIFTY_50_TICKERS

    except Exception as e:
        log.error(f"Failed to fetch or parse live Nifty 50 list: {e}. Using reliable fallback from config file.")
        return config.NIFTY_50_TICKERS
    
def get_point_in_time_nifty50_tickers(point_in_time: pd.Timestamp) -> list[str]:
    """
    Returns Nifty 50 constituents valid at a given historical date.
    Handles inconsistent CSV columns and ensures fallback reliability.
    """
    try:
        df = pd.read_csv('nifty50_historical_constituents.csv')
        df.columns = [col.strip().lower() for col in df.columns]

        # --- FIX: Changed 'symbol' to 'ticker' to match your CSV file ---
        required_cols = {'date', 'ticker'} 
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in CSV: {', '.join(missing)}")

        df['date'] = pd.to_datetime(df['date'])
        historical_df = df.loc[df['date'] <= point_in_time]

        if historical_df.empty:
            log.warning(f"No historical Nifty 50 data on/before {point_in_time.date()}. Using fallback list.")
            return config.NIFTY_50_TICKERS

        latest_snapshot_date = historical_df['date'].max()
        point_in_time_df = historical_df.loc[historical_df['date'] == latest_snapshot_date]

        # --- FIX: Changed 'symbol' to 'ticker' here as well ---
        tickers = [s if s.endswith('.NS') else f"{s}.NS" for s in point_in_time_df['ticker']]
        
        log.info(f"Loaded {len(tickers)} Nifty 50 symbols for {point_in_time.date()} (snapshot: {latest_snapshot_date.date()}).")
        return tickers

    except FileNotFoundError:
        log.error("CRITICAL: nifty50_historical_constituents.csv not found — using fallback list.")
        return config.NIFTY_50_TICKERS
    except Exception as e:
        log.error(f"Error loading historical Nifty 50 list: {e}. Using fallback list.")
        return config.NIFTY_50_TICKERS

def get_nse_all_active_tickers():
    """
    Downloads the official list of all actively traded equity securities from NSE.
    Returns a list of ticker symbols formatted for yfinance (e.g., 'RELIANCE.NS').
    """
    log.info("Attempting to fetch the full list of active NSE equity tickers...")

    # URL for the CSV file on the new NSE website
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    # Alternative/Old URL (keep for reference if the primary one fails):
    # url = "https://www.nseindia.com/content/equities/EQUITY_L.csv"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive'
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status() # Check for HTTP errors

        # Read the CSV content into pandas
        csv_data = StringIO(response.content.decode('utf-8'))
        df = pd.read_csv(csv_data)

        # Basic validation
        if 'SYMBOL' not in df.columns:
            raise ValueError("CSV format unexpected: 'SYMBOL' column missing.")
        if df.empty:
             raise ValueError("Downloaded CSV is empty.")

        # Filter for standard equity series (optional but good practice)
        # Common equity series are 'EQ', 'BE', 'SM'. Adjust if needed.
        # df_filtered = df[df[' SERIES'].isin(['EQ', 'BE', 'SM'])] # Note the space in ' SERIES' often present in NSE CSVs
        # If filtering causes issues, use the unfiltered list first:
        df_filtered = df

        # Clean up column names (remove leading/trailing spaces)
        df_filtered.columns = df_filtered.columns.str.strip()

        tickers = (df_filtered['SYMBOL'] + '.NS').tolist()

        # Sanity check
        if len(tickers) < 1000: # Expecting well over 1000 active symbols
            log.warning(f"Fetched list seems small ({len(tickers)} tickers). Check NSE source/format.")
        else:
            log.info(f"✅ Successfully fetched and parsed {len(tickers)} active NSE tickers.")

        return tickers

    except requests.exceptions.RequestException as e:
        log.error(f"Network error fetching NSE ticker list: {e}")
        return [] # Return empty list on failure
    except pd.errors.ParserError as e:
        log.error(f"Error parsing NSE ticker CSV: {e}")
        return []
    except ValueError as e:
         log.error(f"Data validation error for NSE ticker list: {e}")
         return []
    except Exception as e:
        log.error(f"Unexpected error fetching NSE ticker list: {e}", exc_info=True)
        return []

def get_nifty_smallcap_250_tickers():
    """
    Fetches the Nifty Smallcap 250 constituents list from NSE India.
    Returns a list of ticker symbols formatted for yfinance (e.g., 'SYMBOL.NS').
    Uses the static NIFTY_50 list as a fallback (less ideal, but better than nothing).
    """
    log.info("Attempting to fetch live Nifty Smallcap 250 constituents list...")

    # NSE URL for Nifty Smallcap 250 constituents (Verify this URL is current)
    url = "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv"
    # Older/Alternative might be needed if the above fails
    # url = "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive'
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        csv_data = StringIO(response.content.decode('utf-8'))
        # Skip rows if needed - NSE CSVs sometimes have header info before the table
        # Try without skiprows first, add if necessary (e.g., skiprows=3)
        df = pd.read_csv(csv_data)

        # --- Find the correct symbol column ---
        # NSE CSVs can have inconsistent column names ('Symbol', 'SYMBOL', ' Symbol')
        symbol_col = None
        possible_cols = ['Symbol', 'SYMBOL', ' Symbol']
        for col in possible_cols:
            if col in df.columns:
                symbol_col = col
                break
        if symbol_col is None:
             raise ValueError(f"CSV format error: Could not find symbol column in {df.columns}")
        # --- End Find Symbol Column ---

        if df.empty:
            raise ValueError("Downloaded Smallcap 250 CSV is empty.")

        # Append '.NS' for yfinance compatibility
        tickers = (df[symbol_col].astype(str) + '.NS').tolist()

        # Sanity check (expecting around 250 tickers)
        if 200 <= len(tickers) <= 300:
            log.info(f"✅ Successfully fetched {len(tickers)} Smallcap 250 tickers from NSE.")
            return tickers
        else:
            log.warning(f"Live Smallcap 250 fetch returned an unusual number ({len(tickers)} tickers). Fetch failed.")
            # Return an empty list to signal failure clearly
            return [] 

    except Exception as e:
        log.error(f"Failed to fetch or parse live Nifty Smallcap 250 list: {e}. Fetch failed.")
        return [] 