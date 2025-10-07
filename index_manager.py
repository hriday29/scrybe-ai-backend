# index_manager.py (Hybrid Fetch Version)
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
    Fetches the Nifty 50 constituents for a specific point in time to avoid survivorship bias in backtesting.

    Args:
        point_in_time (pd.Timestamp): The date for which to get the constituents.

    Returns:
        list[str]: A list of ticker symbols for that date.
    """
    try:
        # Load the historical data
        df = pd.read_csv('nifty50_historical_constituents.csv')
        
        # Ensure the 'Date' column is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filter for all historical lists on or before the point_in_time
        historical_df = df[df['Date'] <= point_in_time].copy()
        
        if historical_df.empty:
            log.warning(f"No historical Nifty 50 data found on or before {point_in_time.date()}. Using fallback.")
            return config.NIFTY_50_TICKERS

        # Find the most recent snapshot date from the filtered data
        latest_snapshot_date = historical_df['Date'].max()
        
        # Get the final list of tickers from that specific snapshot date
        point_in_time_df = historical_df[historical_df['Date'] == latest_snapshot_date]
        
        tickers = (point_in_time_df['Symbol'] + '.NS').tolist()
        
        log.info(f"Loaded {len(tickers)} historical Nifty 50 constituents for backtest date {point_in_time.date()} (using list from {latest_snapshot_date.date()}).")
        return tickers
        
    except FileNotFoundError:
        log.error("CRITICAL: nifty50_historical_constituents.csv not found. Backtest will be inaccurate. Using fallback.")
        return config.NIFTY_50_TICKERS
    except Exception as e:
        log.error(f"Error getting point-in-time tickers: {e}. Using fallback.")
        return config.NIFTY_50_TICKERS
