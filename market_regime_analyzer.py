# market_regime_analyzer.py

import data_retriever
from logger_config import log
import pandas as pd
import pandas_ta as ta
import config

def calculate_regime_from_data(historical_data: pd.DataFrame) -> str:
    """
    Calculates a more nuanced market regime from historical index data.
    This logic first identifies the primary trend and then the short-term state.
    """
    if historical_data is None or len(historical_data) < 100:
        return "Sideways" # Default to a safe, non-trending state

    try:
        data = historical_data.copy()
        data.ta.ema(length=20, append=True)
        data.ta.ema(length=50, append=True)
        data.ta.ema(length=100, append=True)
        data.dropna(inplace=True)

        latest = data.iloc[-1]
        close = latest['close']
        ema_20 = latest['EMA_20']
        ema_50 = latest['EMA_50']
        ema_100 = latest['EMA_100']

        # 1. Determine the Primary Trend (using the slower 50 and 100 EMAs)
        primary_trend_is_up = ema_50 > ema_100
        primary_trend_is_down = ema_50 < ema_100

        # 2. Determine the Short-Term State (using the faster 20 and 50 EMAs)
        short_term_is_up = ema_20 > ema_50
        short_term_is_down = ema_20 < ema_50

        # 3. Combine them into a nuanced regime
        if primary_trend_is_up:
            if short_term_is_up:
                # Both long and short term are aligned upwards. Strongest bull case.
                return "Uptrend"
            else:
                # Primary trend is up, but short term is pulling back. A potential dip-buying opportunity.
                return "Bullish Pullback"
        elif primary_trend_is_down:
            if short_term_is_down:
                # Both long and short term are aligned downwards. Strongest bear case.
                return "Downtrend"
            else:
                # Primary trend is down, but short term is rallying. A potential shorting opportunity.
                return "Bearish Rally"
        else:
            # The primary trend indicators are crossed or flat. The market is directionless.
            return "Sideways"
            
    except Exception as e:
        log.error(f"Error calculating regime: {e}")
        return "Sideways" # Default to a safe, non-trending state on any error
    
def get_volatility_regime(historical_vix_data: pd.DataFrame) -> str:
    """
    Classify volatility environment based on India VIX.
    Returns one of: "High-Risk", "Low", "Normal"
    """
    if historical_vix_data is None or len(historical_vix_data) < 20:
        return "Normal"
    try:
        latest_vix = historical_vix_data['close'].iloc[-1]
        vix_20_day_avg = historical_vix_data['close'].rolling(window=20).mean().iloc[-1]
        if latest_vix > config.HIGH_RISK_VIX_THRESHOLD and latest_vix > (vix_20_day_avg * 1.15):
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