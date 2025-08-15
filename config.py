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
        os.getenv("GOOGLE_API_KEY_6"),
        os.getenv("GOOGLE_API_KEY_7"),
        os.getenv("GOOGLE_API_KEY_8"),
        os.getenv("GOOGLE_API_KEY_9"),
        os.getenv("GOOGLE_API_KEY_10"),
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

# --- Strategy & Stock Personality Profiles ---

# This is our final, categorized map of the official Nifty 50 list.
# The system will use this to dynamically apply the best-fit strategy to each stock.

HIGH_BETA_CYCLICAL_TICKERS = {
    "ADANIENT.NS", "ADANIPORTS.NS", "AXISBANK.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS",
    "BPCL.NS", "EICHERMOT.NS", "GRASIM.NS", "HINDALCO.NS", "INDUSINDBK.NS",
    "JSWSTEEL.NS", "M&M.NS", "SBIN.NS", "SHRIRAMFIN.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS"
} # Total: 16 Stocks

STABLE_BLUE_CHIP_TICKERS = {
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "BHARTIARTL.NS",
    "KOTAKBANK.NS", "LT.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "BAJAJ-AUTO.NS",
    "MARUTI.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS", "COALINDIA.NS", "JIOFIN.NS",
    "ULTRACEMCO.NS", "TITAN.NS", "BEL.NS", "HEROMOTOCO.NS", "HDFCLIFE.NS", "SBILIFE.NS"
} # Total: 24 Stocks

LOW_VOLATILITY_COMPOUNDER_TICKERS = {
    "ASIANPAINT.NS", "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS",
    "CIPLA.NS", "SUNPHARMA.NS", "DRREDDY.NS", "APOLLOHOSP.NS", "DIVISLAB.NS",
    "TATACONSUM.NS"
} # Total: 11 Stocks

# --- Strategy Parameter Dictionaries ---

# Default strategy for High-Beta & Cyclical stocks
DEFAULT_SWING_STRATEGY = {
    'name': 'DefaultSwing',
    'horizon_text': 'Short-term Swing (3-7 Days)',
    'holding_period': 7,
    'min_rr_ratio': 2.0,
    'stop_loss_atr_multiplier': 2.0,
    'use_trailing_stop': True,
    'trailing_stop_pct': 1.5
}

# A more patient strategy for large, stable blue-chip companies
BLUE_CHIP_STRATEGY = {
    'name': 'BlueChip',
    'horizon_text': 'Positional Swing (10-20 Days)',
    'holding_period': 15,
    'min_rr_ratio': 1.5,
    'stop_loss_atr_multiplier': 2.5,
    'use_trailing_stop': True,
    'trailing_stop_pct': 2.0
}

# A fast, aggressive strategy for capturing breakout momentum
BREAKOUT_STRATEGY = {
    'name': 'Breakout',
    'horizon_text': 'Rapid Breakout (2-5 Days)',
    'holding_period': 5,
    'min_rr_ratio': 2.5,
    'stop_loss_atr_multiplier': 1.5,
    'use_trailing_stop': True,
    'trailing_stop_pct': 1.5
}

# A highly defensive, mean-reversion-focused strategy for low-volatility stocks
LOW_VOLATILITY_STRATEGY = {
    'name': 'LowVolatility',
    'horizon_text': 'Range-bound Swing (15-30 Days)',
    'holding_period': 20,
    'min_rr_ratio': 1.25,
    'stop_loss_atr_multiplier': 3.0,
    'use_trailing_stop': False
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
    'ADANIENT.NS',  'ADANIPORTS.NS', 'APOLLOHOSP.NS',
    'ASIANPAINT.NS', 'AXISBANK.NS',  'BAJAJ-AUTO.NS',
    'BAJAJFINSV.NS','BAJFINANCE.NS', 'BEL.NS',
    'BHARTIARTL.NS','BPCL.NS',       'BRITANNIA.NS',
    'CIPLA.NS',     'COALINDIA.NS',  'DIVISLAB.NS',
    'DRREDDY.NS',   'EICHERMOT.NS',  'GRASIM.NS',
    'HCLTECH.NS',   'HDFCBANK.NS',   'HDFCLIFE.NS',
    'HEROMOTOCO.NS','HINDALCO.NS',   'HINDUNILVR.NS',
    'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS',
    'ITC.NS',       'JIOFIN.NS',     'JSWSTEEL.NS',
    'KOTAKBANK.NS', 'LT.NS',         'M&M.NS',
    'MARUTI.NS',    'NESTLEIND.NS',  'NTPC.NS',
    'ONGC.NS',      'POWERGRID.NS',  'RELIANCE.NS',
    'SBILIFE.NS',   'SBIN.NS',       'SHRIRAMFIN.NS',
    'SUNPHARMA.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS',
    'TATASTEEL.NS', 'TCS.NS',        'TECHM.NS',
    'TITAN.NS',     'ULTRACEMCO.NS', 'WIPRO.NS'
]