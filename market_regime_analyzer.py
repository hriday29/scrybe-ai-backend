# market_regime_analyzer.py

import data_retriever
from logger_config import log
import pandas as pd

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
        market_regime_status = data_retriever.calculate_regime_from_data(nifty_data)
        volatility_regime_status = data_retriever.get_volatility_regime(vix_data)

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