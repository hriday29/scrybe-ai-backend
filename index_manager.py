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
    Returns Nifty 50 constituents valid at a given historical date.
    Handles inconsistent CSV columns and ensures fallback reliability.
    """
    try:
        df = pd.read_csv('nifty50_historical_constituents.csv')
        df.columns = [col.strip().lower() for col in df.columns]

        # Validate columns
        required_cols = {'date', 'symbol'}
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

        tickers = [s if s.endswith('.NS') else f"{s}.NS" for s in point_in_time_df['symbol']]
        log.info(f"Loaded {len(tickers)} Nifty 50 symbols for {point_in_time.date()} (snapshot: {latest_snapshot_date.date()}).")
        return tickers

    except FileNotFoundError:
        log.error("CRITICAL: nifty50_historical_constituents.csv not found — using fallback list.")
        return config.NIFTY_50_TICKERS
    except Exception as e:
        log.error(f"Error loading historical Nifty 50 list: {e}. Using fallback list.")
        return config.NIFTY_50_TICKERS
        
    except FileNotFoundError:
        log.error("CRITICAL: nifty50_historical_constituents.csv not found. Backtest will be inaccurate. Using fallback.")
        return config.NIFTY_50_TICKERS
    except Exception as e:
        log.error(f"Error getting point-in-time tickers: {e}. Using fallback.")
        return config.NIFTY_50_TICKERS
