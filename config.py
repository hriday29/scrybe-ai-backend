# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# --- Database & API Keys ---
ANALYSIS_DB_URI = os.getenv("ANALYSIS_DB_URI")
SCHEDULER_DB_URI = os.getenv("SCHEDULER_DB_URI")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
NEWSAPI_API_KEY = os.getenv('NEWSAPI_API_KEY')

GEMINI_API_KEY_POOL = [
    key for key in [
        os.getenv("GOOGLE_API_KEY_1"),
        os.getenv("GOOGLE_API_KEY_2"),
        os.getenv("GOOGLE_API_KEY_3"),
        os.getenv("GOOGLE_API_KEY_4"),
        os.getenv("GOOGLE_API_KEY_5"),
    ] if key
]

# --- Email Configuration ---
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

if not all([ANALYSIS_DB_URI, SCHEDULER_DB_URI, GEMINI_API_KEY, NEWSAPI_API_KEY]):
    print("⚠️  Warning: One or more environment variables (DB URIs, API Key) are missing.")
else:
    print("✅ Secure configuration loaded successfully for both databases.")

# --- Models ---
PRO_MODEL = 'gemini-2.5-pro'
FLASH_MODEL = 'gemini-2.5-flash'

VST_STRATEGY = {
    'name': 'VST',
    'horizon_text': 'Quick Flip (1-3 Days)',
    'holding_period': 3,
    'min_rr_ratio': 2.0
}

# --- Market Index Analysis Config ---
# Tickers are from Yahoo Finance
INDEX_LIST = {
    # --- Major Broad-Based Indices ---
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "NIFTY Next 50": "^CNXNXT",
    "NIFTY 100": "^CNX100",
    "NIFTY 500": "NIFTY500.NS",
    "NIFTY Midcap 100": "^CNXMIDCAP",
    "NIFTY Smallcap 100": "NIFTY_SMALCAP_100.NS",
    "NIFTY MidSmallcap 400": "NIFTY_MIDSML_400.NS",

    # --- Key Sectoral Indices ---
    "NIFTY Bank": "^NSEBANK",
    "NIFTY Financial Services": "NIFTY_FIN_SERVICE.NS",
    "NIFTY IT": "^CNXIT",
    "NIFTY Auto": "^CNXAUTO",
    "NIFTY Pharma": "^CNXPHARMA",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY Metal": "^CNXMETAL",
    "NIFTY PSU Bank": "^CNXPSUBANK",
    "NIFTY Private Bank": "NIFTY_PRIVATEBANK.NS",
    "NIFTY Oil & Gas": "NIFTY_OIL_AND_GAS.NS",

    # --- Thematic & Strategy Indices ---
    "NIFTY India Consumption": "^CNXCONSUM",
    "NIFTY50 Equal Weight": "NIFTY50_EQL_WGT.NS",
    "NIFTY Alpha 50": "NIFTY_ALPHA_50.NS",
    "NIFTY Low Volatility 50": "NIFTY_LOW_VOL_50.NS",
    "NIFTY High Beta 50": "NIFTY_HIGH_BETA_50.NS",
    "NIFTY Quality 30": "NIFTY_QUALITY_30.NS",

    # --- BSE Indices ---
    "BSE MidCap": "^BSEMD",
    "BSE SmallCap": "^BSESM",
    "BSE Healthcare": "^BSEHC",
    "BSE FMCG": "^BSEFMC",
}

VOLUME_SURGE_THRESHOLD = 1.6 #represents a 60% surge
ADX_THRESHOLD = 25
TRADE_EXPIRY_DAYS = 7

BACKTEST_CONFIG = {
    # Brokerage: e.g., 0.05% per side (buy and sell)
    "brokerage_pct": 0.05,
    # Slippage: Estimated price difference due to order execution speed
    "slippage_pct": 0.02,
    # STT: Securities Transaction Tax on delivery sell trades
    "stt_pct": 0.1
}

BACKTEST_PORTFOLIO_CONFIG = {
    "initial_capital": 100000.0,      # Start with 1 Lakh
    "position_size_pct_of_capital": 10.0 # Allocate 10% of initial capital to each trade
}

BETA_TESTER_EMAILS = ["vighriday@gmail.com"]

NIFTY_50_TICKERS = [
  'ADANIENT.NS',    'ADANIPORTS.NS', 'APOLLOHOSP.NS',
  'ASIANPAINT.NS', 'AXISBANK.NS',    'BAJAJ-AUTO.NS',
  'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BEL.NS',
  'BHARTIARTL.NS', 'CIPLA.NS',       'COALINDIA.NS',
  'DRREDDY.NS',    'EICHERMOT.NS',   'ETERNAL.NS',
  'GRASIM.NS',     'HCLTECH.NS',     'HDFCBANK.NS',
  'HDFCLIFE.NS',   'HEROMOTOCO.NS', 'HINDALCO.NS',
  'HINDUNILVR.NS', 'ICICIBANK.NS',   'INDUSINDBK.NS',
  'INFY.NS',       'ITC.NS',         'JIOFIN.NS',
  'JSWSTEEL.NS',   'KOTAKBANK.NS',   'LT.NS',
  'M&M.NS',        'MARUTI.NS',      'NESTLEIND.NS',
  'NTPC.NS',       'ONGC.NS',        'POWERGRID.NS',
  'RELIANCE.NS',   'SBILIFE.NS',     'SBIN.NS',
  'SHRIRAMFIN.NS', 'SUNPHARMA.NS',   'TATACONSUM.NS',
  'TATAMOTORS.NS', 'TATASTEEL.NS',   'TCS.NS',
  'TECHM.NS',      'TITAN.NS',       'TRENT.NS',
  'ULTRACEMCO.NS', 'WIPRO.NS'
]