#config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Azure AI Credentials ---
AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT")
AZURE_AI_API_KEY = os.getenv("AZURE_AI_API_KEY")

# --- Data Source Control ---
# This is the master switch. Set to "angelone" or "yfinance".
DATA_SOURCE = os.getenv("DATA_SOURCE", "angelone")

# --- Database & API Keys ---
ANALYSIS_DB_URI = os.getenv("ANALYSIS_DB_URI")
SCHEDULER_DB_URI = os.getenv("SCHEDULER_DB_URI")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")

# --- Angel One Credentials ---
ANGELONE_API_KEY = os.getenv("ANGELONE_API_KEY")
ANGELONE_CLIENT_ID = os.getenv("ANGELONE_CLIENT_ID")
ANGELONE_PASSWORD = os.getenv("ANGELONE_PASSWORD")
ANGELONE_TOTP_SECRET = os.getenv("ANGELONE_TOTP_SECRET")

# --- Gemini API Key Pool (REMOVED) ---
# This is no longer needed as we use a single Azure endpoint and key.

# --- Email Configuration ---
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465  # SSL
BETA_TESTER_EMAILS = ["vighriday@gmail.com"]
EMAIL_RECIPIENTS = os.getenv("EMAIL_RECIPIENTS", "")
EMAIL_RECIPIENTS = [email.strip() for email in EMAIL_RECIPIENTS.split(",") if email.strip()]
if not EMAIL_RECIPIENTS:
    EMAIL_RECIPIENTS = BETA_TESTER_EMAILS

# --- Models (MODIFIED FOR AZURE) ---
# These now refer to your DEPLOYMENT NAMES in Azure AI Studio.
# Please set them in your .env file.
PRO_MODEL = os.getenv("AZURE_PRO_DEPLOYMENT", "gpt-4o-deployment")
FLASH_MODEL = os.getenv("AZURE_FLASH_DEPLOYMENT", "gpt-35-turbo-deployment")

# --- Strategy & Portfolio ---
APEX_SWING_STRATEGY = {
    "name": "AI_Analyst_v1",
    "allow_short_selling": True,
    "holding_period": 10,
    "stop_loss_atr_multiplier": 2.0,
    "profit_target_rr_multiple": 3.0,
    "min_conviction_score": 15,
    "use_trailing_stop": True,
    "trailing_stop_atr_multiplier": 1.5, 
    "trailing_stop_activation_r": 1.0,
}

PORTFOLIO_CONSTRAINTS = {
    "max_concurrent_trades": 5
}

BACKTEST_PORTFOLIO_CONFIG = {
    "initial_capital": 100000.0,
    "risk_per_trade_pct": 1.0 # Risk 1% of total capital per trade
}

# --- Market & Screener Config ---
HIGH_RISK_VIX_THRESHOLD = 22.0 # Lowered threshold for more cautious stance
ADX_THRESHOLD = 25
VOLUME_SURGE_THRESHOLD = 1.6 # Unused, but kept for reference

BACKTEST_CONFIG = {
    "brokerage_pct": 0.05,
    "slippage_pct": 0.02,
    "stt_pct": 0.1,
}

# --- Ticker Lists & Static Data ---
# This is a fallback and can be removed if index_manager is always reliable
NIFTY_50_TICKERS = [
    "ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS","AXISBANK.NS","BAJAJ-AUTO.NS",
    "BAJAJFINSV.NS","BAJFINANCE.NS","BHARTIARTL.NS","BPCL.NS","BRITANNIA.NS","CIPLA.NS",
    "COALINDIA.NS","DIVISLAB.NS","DRREDDY.NS","EICHERMOT.NS","GRASIM.NS","HCLTECH.NS","HDFCBANK.NS",
    "HDFCLIFE.NS","HEROMOTOCO.NS","HINDALCO.NS","HINDUNILVR.NS","ICICIBANK.NS","INDUSINDBK.NS",
    "INFY.NS","ITC.NS","JSWSTEEL.NS","KOTAKBANK.NS","LT.NS","LTIM.NS","M&M.NS","MARUTI.NS",
    "NESTLEIND.NS","NTPC.NS","ONGC.NS","POWERGRID.NS","RELIANCE.NS","SBILIFE.NS","SBIN.NS",
    "SHRIRAMFIN.NS","SUNPHARMA.NS","TATACONSUM.NS","TATAMOTORS.NS","TATASTEEL.NS","TCS.NS",
    "TECHM.NS","TITAN.NS","ULTRACEMCO.NS","WIPRO.NS",
]

MAJOR_ECONOMIC_EVENTS = [
    {"date": "2025-10-14", "event": "CPI Inflation Data Release (Monthly)"},
    {"date": "2025-10-15", "event": "Start of Q2 Earnings Season"},
    {"date": "2025-10-30", "event": "Monthly F&O Series Expiry"},
    {"date": "2025-11-01", "event": "Auto Sales Figures Release (Monthly)"},
    {"date": "2025-11-12", "event": "IIP Data Release (Monthly)"},
]

LIVE_MACRO_CONTEXT = {
    "India GDP Growth (YoY)": "7.8% (Q1 2025 Estimate)",
    "RBI Policy Stance": "Neutral, monitoring inflation closely.",
    "Key Global Factor": "Global market volatility influenced by US Federal Reserve policy.",
    "Domestic Consumer Sentiment": "Stable but cautious.",
}
