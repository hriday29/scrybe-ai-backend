# test_prompt.py
import json
from ai_analyzer import AIAnalyzer
from logger_config import log
import config

def run_single_prompt_test():
    """
    A dedicated script to test the get_apex_analysis function with a specific
    data packet known to cause issues, allowing for rapid prompt engineering.
    """
    log.info("--- Starting Single Prompt Generation Test ---")

    try:
        # 1. Initialize the AI Analyzer with dynamic provider selection
        # Make sure your .env file is configured correctly
        analyzer = AIAnalyzer()

        # 2. Define the exact data that caused the block for CIPLA.NS
        ticker_to_test = "CIPLA.NS"
        context_that_failed = {
            "layer_1_macro_context": {
                "nifty_50_regime": "Bullish"
            },
            "layer_2_relative_strength": {
                "relative_strength_vs_nifty50": "Underperforming"
            },
            "layer_3_fundamental_moat": {
                "valuation_proxy": "52.8% of 52-Week Range",
                "quality_proxy_volatility": "22.39%",
                "quality_score": 55
            },
            "layer_4_technicals": {
                "daily_close": 1515.699951171875
            },
            "layer_5_options_sentiment": {
                "sentiment": "Unavailable in backtest"
            },
            "layer_6_news_catalyst": {
                "summary": "Unavailable in backtest"
            }
        }

        # 3. Use placeholder historical data, as it's not the cause of the issue
        placeholder_history = "No recent trade history for this stock."

        log.info(f"Attempting to generate analysis for {ticker_to_test}...")

        # 4. Call the analysis function
        final_analysis = analyzer.get_apex_analysis(
            ticker=ticker_to_test,
            full_context=context_that_failed,
            strategic_review=placeholder_history,
            tactical_lookback=placeholder_history,
            per_stock_history=placeholder_history
        )

        # 5. Print the result
        if final_analysis:
            log.info("--- ✅ TEST SUCCEEDED ---")
            log.info("Successfully generated analysis:")
            print(json.dumps(final_analysis, indent=2))
        else:
            # This case should ideally not be hit if an exception is raised
            log.error("--- ❌ TEST FAILED ---")
            log.error("Analysis returned None without an exception.")

    except Exception as e:
        log.error("--- ❌ TEST FAILED ---")
        log.error(f"An exception occurred during the test: {e}", exc_info=True)

if __name__ == "__main__":
    run_single_prompt_test()
