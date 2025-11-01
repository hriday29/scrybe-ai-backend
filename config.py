"""
config.py

Purpose
- Central configuration and environment variable resolution for AI providers, data sources,
    database URIs, email, directories, strategy/portfolio parameters, and static lists.

How it fits
- Consumed across the pipeline, API, and simulators to keep settings in one place with sensible
    defaults and directory bootstrapping.

Main role
- Normalize env var naming across Azure OpenAI and Azure AI Foundry, expose model selections,
    and provide constants used by analyzers, retrievers, and reporting modules.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- AI Provider & Credentials ---
# Select between Azure OpenAI and Azure AI Foundry (Models as a Service)
# Values: "azure-openai" (default) | "azure-foundry"
AI_PROVIDER = os.getenv("AI_PROVIDER", "azure-openai").lower()

# Azure OpenAI (classic) variables
AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_AI_API_KEY = os.getenv("AZURE_AI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# Azure AI Foundry Inference variables (for Grok and other non-OpenAI models)
# Endpoint can be the global endpoint below or a regional endpoint like https://<region>.models.ai.azure.com
AZURE_INFERENCE_ENDPOINT = os.getenv("AZURE_INFERENCE_ENDPOINT", "https://models.inference.ai.azure.com")
AZURE_INFERENCE_API_KEY = os.getenv("AZURE_INFERENCE_API_KEY")
AZURE_INFERENCE_API_VERSION = os.getenv("AZURE_INFERENCE_API_VERSION")  # optional

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

# --- Directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- Models (OpenAI-compatible names) ---
"""
Model selection has been renamed from PRO/FLASH to primary/secondary to be provider-agnostic
and consistent across Azure OpenAI and Azure AI Foundry.

Backward compatibility: we still honor old env vars if the new ones are not set.
Resolution order (first non-empty wins):
- PRIMARY_MODEL: AZURE_PRIMARY_DEPLOYMENT, PRIMARY_MODEL, AZURE_PRO_DEPLOYMENT, PRO_MODEL
- SECONDARY_MODEL: AZURE_SECONDARY_DEPLOYMENT, SECONDARY_MODEL, AZURE_FLASH_DEPLOYMENT, FLASH_MODEL

For AI_PROVIDER=azure-openai: these should be Azure deployment names.
For AI_PROVIDER=azure-foundry: these should be model IDs, e.g., "grok-2", "mistral-large", "gpt-4o-mini".
"""

PRIMARY_MODEL = (
    os.getenv("AZURE_PRIMARY_DEPLOYMENT")
    or os.getenv("PRIMARY_MODEL")
    or os.getenv("AZURE_PRO_DEPLOYMENT")
    or os.getenv("PRO_MODEL")
    or "gpt-4.1"
)

SECONDARY_MODEL = (
    os.getenv("AZURE_SECONDARY_DEPLOYMENT")
    or os.getenv("SECONDARY_MODEL")
    or os.getenv("AZURE_FLASH_DEPLOYMENT")
    or os.getenv("FLASH_MODEL")
    or "gpt-4.1"
)

# Optional explicit Grok model name when using Foundry
GROK_MODEL = os.getenv("GROK_MODEL", "grok-2")

# --- Strategy & Portfolio ---
APEX_SWING_STRATEGY = {
    "name": "AI_Candlestick_Swing_v1", # Renamed to reflect the new screener
    "allow_short_selling": True,
    "holding_period": 15,              # Keep 15 days for swing trades
    "stop_loss_atr_multiplier": 1.75,  # Keep initial stop relatively standard
    "profit_target_rr_multiple": 2.5,  # Keep target standard
    "min_conviction_score": 45,        # Slightly increased threshold for AI conviction -> to compensate for the broader screener funnel.
    "use_trailing_stop": True,
    "trailing_stop_atr_multiplier": 1.5, # Keep the trail distance standard for now
    "trailing_stop_activation_r": 1.5,   # Activate trailing stop a bit later (after 1.5R) -> to potentially give trades more room to run initially.
}

PORTFOLIO_CONSTRAINTS = {
    "max_concurrent_trades": 10              # Increased to allow more diversification.
}

BACKTEST_PORTFOLIO_CONFIG = {
    "initial_capital": 100000.0,
    "risk_per_trade_pct": 1.5               # Slightly increased risk per trade, balanced by tighter stops.
}

MAX_AI_CONCURRENCY = 10         # Max parallel AI calls in the pipeline

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

# Public indices to expose via API and analysis endpoints
INDEX_LIST = {
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
}

# Live trading universe for the app; can be overridden via env as a comma-separated list
_live_universe_env = os.getenv("LIVE_TRADING_UNIVERSE", "")
if _live_universe_env:
    LIVE_TRADING_UNIVERSE = [s.strip() for s in _live_universe_env.split(",") if s.strip()]
else:
    LIVE_TRADING_UNIVERSE = NIFTY_50_TICKERS

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

RISK_FREE_RATE = 0.05           # 5% risk-free rate for options greeks