# ai_sanity_check.py
import json
import time
from logger_config import log
import config
from ai_analyzer import AIAnalyzer

def run_ai_sanity_check():
    """
    A dedicated script to test-fire the AIAnalyzer on specific, handcrafted
    scenarios to validate its reasoning before running a full backtest.
    """
    log.info("--- 🚀 STARTING AI SANITY CHECK ---")
    
    try:
        # Initialize the AI Analyzer
        analyzer = AIAnalyzer(api_key=config.GEMINI_API_KEY)

        # --- TEST CASE 1: The "Perfect" Momentum Buy Setup ---
        # A fundamentally sound, technically strong stock in a bullish market.
        # EXPECTED OUTCOME: A 'BUY' signal with a high Scrybe Score (> 70).
        log.info("\n--- [TEST CASE 1/3]: PERFECT MOMENTUM BUY ---")
        case_1_ticker = "PERFECT_BUY.NS"
        case_1_context = {
            "layer_1_macro_context": {"nifty_50_regime": "Bullish"},
            "layer_2_relative_strength": {"relative_strength_vs_nifty50": "Outperforming"},
            "layer_3_fundamental_moat": {
                "valuation_proxy": "75.2% of 52-Week Range",
                "quality_proxy_volatility": "25.10%",
                "quality_score": 85
            },
            "layer_4_technicals": {"daily_close": 1500.00}
        }
        run_single_test(analyzer, case_1_ticker, case_1_context)
        time.sleep(30) # Pause to respect API rate limits

        # --- TEST CASE 2: The "Obvious" Contradiction Veto ---
        # A technically strong stock, but in a strongly Bearish market.
        # EXPECTED OUTCOME: A 'HOLD' signal, as the macro risk is too high.
        # The AI's verdict should explicitly mention the market regime conflict.
        log.info("\n--- [TEST CASE 2/3]: OBVIOUS CONTRADICTION (BEAR MARKET) ---")
        case_2_ticker = "CONTRADICTION.NS"
        case_2_context = {
            "layer_1_macro_context": {"nifty_50_regime": "Bearish"}, # The key conflict
            "layer_2_relative_strength": {"relative_strength_vs_nifty50": "Outperforming"},
            "layer_3_fundamental_moat": {
                "valuation_proxy": "80.5% of 52-Week Range",
                "quality_proxy_volatility": "22.00%",
                "quality_score": 90
            },
            "layer_4_technicals": {"daily_close": 2100.00}
        }
        run_single_test(analyzer, case_2_ticker, case_2_context)
        time.sleep(30)

        # --- TEST CASE 3: The "Weak & Choppy" Setup ---
        # A stock going nowhere, with mediocre stats and a neutral market.
        # EXPECTED OUTCOME: A 'HOLD' signal with a low Scrybe Score (near 0).
        log.info("\n--- [TEST CASE 3/3]: WEAK & CHOPPY (NO CLEAR SIGNAL) ---")
        case_3_ticker = "CHOPPY.NS"
        case_3_context = {
            "layer_1_macro_context": {"nifty_50_regime": "Neutral"},
            "layer_2_relative_strength": {"relative_strength_vs_nifty50": "Underperforming"},
            "layer_3_fundamental_moat": {
                "valuation_proxy": "45.0% of 52-Week Range",
                "quality_proxy_volatility": "40.50%",
                "quality_score": 40
            },
            "layer_4_technicals": {"daily_close": 500.00}
        }
        run_single_test(analyzer, case_3_ticker, case_3_context)

        log.info("\n--- ✅ AI SANITY CHECK COMPLETE ---")

    except Exception as e:
        log.error(f"--- ❌ AI SANITY CHECK FAILED ---")
        log.error(f"An exception occurred: {e}", exc_info=True)

def run_single_test(analyzer: AIAnalyzer, ticker: str, context: dict):
    """Helper function to run one test case and print the results."""
    try:
        log.info(f"Submitting data for {ticker}...")
        # Use placeholder history as we are testing the AI's reaction to the context
        placeholder_history = "No recent trade history for this test case."
        
        analysis = analyzer.get_apex_analysis(
            ticker=ticker,
            full_context=context,
            strategic_review=placeholder_history,
            tactical_lookback=placeholder_history,
            per_stock_history=placeholder_history
        )

        if analysis:
            log.info(f"--- ✅ AI Response Received for {ticker} ---")
            print(json.dumps(analysis, indent=2))
        else:
            log.error(f"--- ❌ AI returned a null response for {ticker} ---")

    except Exception as e:
        log.error(f"--- ❌ FAILED to get analysis for {ticker}. Error: {e} ---")

if __name__ == "__main__":
    run_ai_sanity_check()