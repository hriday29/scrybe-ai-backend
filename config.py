import os
from dotenv import load_dotenv

load_dotenv()

# --- Database & API Keys ---
ANALYSIS_DB_URI = os.getenv("ANALYSIS_DB_URI")
SCHEDULER_DB_URI = os.getenv("SCHEDULER_DB_URI")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")

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

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465  # SSL

# Beta fallback list
BETA_TESTER_EMAILS = ["vighriday@gmail.com"]

EMAIL_RECIPIENTS = os.getenv("EMAIL_RECIPIENTS", "")
EMAIL_RECIPIENTS = [email.strip() for email in EMAIL_RECIPIENTS.split(",") if email.strip()]

if not EMAIL_RECIPIENTS:
    EMAIL_RECIPIENTS = BETA_TESTER_EMAILS

# --- Models ---
PRO_MODEL = "gemini-2.5-flash"
FLASH_MODEL = "gemini-2.5-flash"

# --- Strategy Profile ---

APEX_SWING_STRATEGY = {
    "name": "ApexSwing",
    "holding_period": 10, # This won't be used, but we can leave it
    "stop_loss_atr_multiplier": 2.0,
    "profit_target_rr_multiple": 2.0,
    "min_conviction_score": 25,
    "use_trailing_stop": False, # CHANGE THIS TO FALSE
    "trailing_stop_atr_multiplier": 1.5,
}

# --- ADD THIS NEW CONFIGURATION ---
PORTFOLIO_CONSTRAINTS = {
    "max_concurrent_trades": 5
}

# --- Market Index Analysis Config ---
INDEX_LIST = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "NIFTY Next 50": "^CNXNXT",
    "NIFTY 100": "^CNX100",
    "NIFTY 500": "NIFTY500.NS",
    "NIFTY Midcap 100": "^CNXMIDCAP",
    "NIFTY Smallcap 100": "NIFTY_SMALLCAP_100.NS",
    "NIFTY MidSmallcap 400": "NIFTY_MIDSML_400.NS",
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
    "NIFTY India Consumption": "^CNXCONSUM",
    "NIFTY50 Equal Weight": "NIFTY50_EQL_WGT.NS",
    "NIFTY Alpha 50": "NIFTY_ALPHA_50.NS",
    "NIFTY Low Volatility 50": "NIFTY_LOW_VOL_50.NS",
    "NIFTY High Beta 50": "NIFTY_HIGH_BETA_50.NS",
    "NIFTY Quality 30": "NIFTY_QUALITY_30.NS",
    "BSE MidCap": "^BSEMD",
    "BSE SmallCap": "^BSESM",
    "BSE Healthcare": "^BSEHC",
    "BSE FMCG": "^BSEFMC",
}

VOLUME_SURGE_THRESHOLD = 1.6
ADX_THRESHOLD = 25

BACKTEST_CONFIG = {
    "brokerage_pct": 0.05,
    "slippage_pct": 0.02,
    "stt_pct": 0.1,
}

BACKTEST_PORTFOLIO_CONFIG = {
    "initial_capital": 100000.0,
    "position_size_pct_of_capital": 10.0,
}

NIFTY_50_TICKERS = [
    "ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS","AXISBANK.NS","BAJAJ-AUTO.NS",
    "BAJAJFINSV.NS","BAJFINANCE.NS","BEL.NS","BHARTIARTL.NS","BPCL.NS","BRITANNIA.NS","CIPLA.NS",
    "COALINDIA.NS","DIVISLAB.NS","DRREDDY.NS","EICHERMOT.NS","GRASIM.NS","HCLTECH.NS","HDFCBANK.NS",
    "HDFCLIFE.NS","HEROMOTOCO.NS","HINDALCO.NS","HINDUNILVR.NS","ICICIBANK.NS","INDUSINDBK.NS",
    "INFY.NS","ITC.NS","JIOFIN.NS","JSWSTEEL.NS","KOTAKBANK.NS","LT.NS","M&M.NS","MARUTI.NS",
    "NESTLEIND.NS","NTPC.NS","ONGC.NS","POWERGRID.NS","RELIANCE.NS","SBILIFE.NS","SBIN.NS",
    "SHRIRAMFIN.NS","SUNPHARMA.NS","TATACONSUM.NS","TATAMOTORS.NS","TATASTEEL.NS","TCS.NS",
    "TECHM.NS","TITAN.NS","ULTRACEMCO.NS","WIPRO.NS",
]

# The official, curated V1.0 A-List for Project Apex
LIVE_TRADING_UNIVERSE = [
    "LT.NS",
    "BAJFINANCE.NS",
    "TATAMOTORS.NS",
    "BAJAJFINSV.NS",
    "WIPRO.NS",
    "TITAN.NS",
    "SBIN.NS"
]

MAJOR_ECONOMIC_EVENTS = [
    {"date": "2025-07-12", "event": "CPI Inflation Data Release"},
    {"date": "2025-07-15", "event": "Start of Quarterly Results Season"},
    {"date": "2025-07-31", "event": "Monthly F&O Series Expiry"},
    {"date": "2025-08-01", "event": "Auto Sales Figures Release (Monthly)"},
    {"date": "2025-08-08", "event": "RBI Monetary Policy Meeting"},
    {"date": "2025-08-12", "event": "IIP Data Release (Monthly)"},
    {"date": "2025-08-28", "event": "Monthly F&O Series Expiry"},
    {"date": "2025-09-01", "event": "Auto Sales Figures Release (Monthly)"},
    {"date": "2025-09-12", "event": "CPI Inflation Data Release (Monthly)"},
]

LIVE_MACRO_CONTEXT = {
    "India GDP Growth (YoY)": "7.8% (Q1 2025 Estimate)",
    "RBI Policy Stance": "Neutral with a focus on inflation control",
    "Key Global Factor": "Monitoring global indices for signs of slowdown.",
    "Domestic Consumer Sentiment": "Cautiously Optimistic",
}

# --- Dynamic Risk Management Configuration ---
DYNAMIC_RISK_CONFIG = {
    'lookback_period': 3,                 # Number of past trades to consider for a stock
    'red_mode_threshold': 2,              # Number of CONSECUTIVE losses to enter Red Mode
    'yellow_mode_threshold': 2,           # Number of TOTAL losses in lookback to enter Yellow Mode
    'green_mode_position_size_pct': 100.0,
    'yellow_mode_position_size_pct': 50.0,
    'red_mode_position_size_pct': 0.0,    # The "Circuit Breaker" - no new trades
    'cooldown_period_days': 5             # Days a stock stays in Red Mode
}
HIGH_RISK_VIX_THRESHOLD = 28.0

BACKTEST_PORTFOLIO_CONFIG = {
    "initial_capital": 100000.0,
    "position_size_pct_of_capital": 10.0, # This will no longer be used directly
    "sizing_method": "risk_parity", # ADD THIS LINE
    "risk_per_trade_pct": 1.0 # ADD THIS LINE (means we risk 1% of capital per trade)
}