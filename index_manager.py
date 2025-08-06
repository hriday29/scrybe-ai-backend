# import pandas as pd
# import requests
# from io import StringIO
# from logger_config import log

# def get_nifty50_tickers():
#     """
#     Fetches the current list of Nifty 50 constituents from the NSE India website.
#     """
#     log.info("Fetching live Nifty 50 constituents list...")
    
#     url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
    
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#     }

#     try:
#         response = requests.get(url, headers=headers, timeout=10)
#         response.raise_for_status() 

#         csv_data = StringIO(response.content.decode('utf-8'))
        
#         df = pd.read_csv(csv_data)
        
#         tickers = (df['Symbol'] + '.NS').tolist()
        
#         log.info(f"Successfully fetched {len(tickers)} tickers.")
#         return tickers

#     except requests.exceptions.RequestException as e:
#         log.error(f"Failed to fetch Nifty 50 list due to a network error: {e}")
#         return []
#     except Exception as e:
#         log.error(f"Failed to parse Nifty 50 list. The format may have changed. Error: {e}")
#         return []

# index_manager.py
import config
from logger_config import log

def get_nifty50_tickers():
    """
    Returns a static, hardcoded list of Nifty 50 constituents from the config file.
    This is a reliable method to avoid network errors from the NSE website.
    """
    log.info("Fetching Nifty 50 constituents list from config file...")
    # This now points to the list you need to add to your config.py file
    return config.NIFTY_50_TICKERS