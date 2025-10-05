# market_regime_analyzer.py

import data_retriever
from logger_config import log
import pandas as pd
import pandas_ta as ta

def calculate_regime_from_data(historical_data: pd.DataFrame) -> str:
    """
    Calculates the market regime from a given DataFrame of historical index data.
    Returns 'Bullish', 'Bearish', or 'Neutral'.
    """
    if historical_data is None or len(historical_data) < 100:
        return "Neutral" # Not enough data to determine

    try:
        # Use a copy to avoid SettingWithCopyWarning
        data = historical_data.copy()
        data.ta.ema(length=20, append=True)
        data.ta.ema(length=50, append=True)
        data.ta.ema(length=100, append=True)
        data.dropna(inplace=True)

        latest_emas = data.iloc[-1]
        ema_20 = latest_emas['EMA_20']
        ema_50 = latest_emas['EMA_50']
        ema_100 = latest_emas['EMA_100']

        if ema_20 > ema_50 > ema_100:
            return "Bullish"
        elif ema_20 < ema_50 < ema_100:
            return "Bearish"
        else:
            return "Neutral"
    except Exception:
        return "Neutral" # Default to Neutral on any error
    
def get_volatility_regime(historical_vix_data: pd.DataFrame) -> str:
    """
    Classify volatility environment based on India VIX.
    Returns one of: "High-Risk", "Low", "Normal"
    """
    if historical_vix_data is None or len(historical_vix_data) < 20:
        return "Normal"
    try:
        latest_vix = historical_vix_data['close'].iloc[-1]   # <-- lowercase
        vix_20_day_avg = historical_vix_data['close'].rolling(window=20).mean().iloc[-1]
        HIGH_VIX_THRESHOLD = 20.0
        if latest_vix > HIGH_VIX_THRESHOLD and latest_vix > (vix_20_day_avg * 1.15):
            return "High-Risk"
        elif latest_vix < 14:
            return "Low"
        else:
            return "Normal"
    except Exception:
        return "Normal"
    
def get_market_regime_context() -> dict:
    """
    Determines the current market regime context including trend and volatility.
    """
    log.info("--- Determining Market Regime Context ---")
    
    try:
        # Fetch the required historical data using the centralized retriever
        nifty_data = data_retriever.get_historical_stock_data("^NSEI")
        vix_data = data_retriever.get_historical_stock_data("^INDIAVIX")

        # Calculate the regimes using helper functions from data_retriever
        market_regime_status = calculate_regime_from_data(nifty_data)     
        volatility_regime_status = get_volatility_regime(vix_data)

        log.info(f"Market Regime: {market_regime_status}, Volatility Regime: {volatility_regime_status}")

        return {
            "market_regime": {
                "regime_status": market_regime_status,
                "analysis_source": "NIFTY 50 EMA Crossover (20, 50, 100)"
            },
            "volatility_regime": {
                "volatility_status": volatility_regime_status,
                "analysis_source": "India VIX Analysis"
            }
        }
    except Exception as e:
        log.error(f"Failed to determine market regime context: {e}")
        # Return a neutral/default state in case of error
        return {
            "market_regime": {"regime_status": "Neutral", "error": str(e)},
            "volatility_regime": {"volatility_status": "Normal", "error": str(e)}
        }