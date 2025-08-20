# market_regime_analyzer.py
import data_retriever
from logger_config import log

def get_current_market_regime() -> str:
    """
    Determines the current market regime by calling the core function
    in the data_retriever. This acts as a clean interface for the orchestrator.
    
    Returns:
        str: 'Bullish', 'Bearish', or 'Neutral'
    """
    log.info("--- [Funnel Step 1] Analyzing Market Regime ---")
    try:
        regime = data_retriever.get_market_regime()
        log.info(f"✅ Market Regime determined as: '{regime}'")
        return regime
    except Exception as e:
        log.error(f"Failed to determine market regime: {e}", exc_info=True)
        # In case of any failure, we default to Neutral to be safe.
        return "Neutral"