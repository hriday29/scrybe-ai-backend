# test_prompt.py
import pandas as pd
import config
from logger_config import log
import data_retriever
import technical_analyzer
from ai_analyzer import AIAnalyzer
from datetime import datetime

def run_single_prompt_test():
    """
    A fast, single-API-call test to validate a prompt change.
    """
    # --- 1. SETUP: Define our single test case ---
    TICKER_TO_TEST = "TATASTEEL.NS"
    DATE_TO_TEST = "2025-06-06"
    
    log.info(f"--- 🧪 Running Strategy Laboratory Test ---")
    log.info(f"Testing new prompt on {TICKER_TO_TEST} for date {DATE_TO_TEST}")

    # --- 2. PREPARATION: Recreate the exact data the AI saw on that day ---
    try:
        # Load the corrected AI Analyzer with the new prompt
        analyzer = AIAnalyzer(api_key=config.GEMINI_API_KEY)

        # Get all historical data up to our test date
        full_historical_data = data_retriever.get_historical_stock_data(TICKER_TO_TEST, end_date=DATE_TO_TEST)
        if full_historical_data is None or len(full_historical_data) < 100:
            log.error("Could not get enough historical data for the test.")
            return

        # The data slice is the data as it existed AT THE MOMENT of prediction
        data_slice = full_historical_data.loc[:DATE_TO_TEST].copy()

        # Re-calculate the exact same indicators
        data_slice.ta.bbands(length=20, append=True)
        data_slice.ta.rsi(length=14, append=True)
        data_slice.ta.macd(fast=12, slow=26, signal=9, append=True)
        data_slice.ta.adx(length=14, append=True)
        data_slice.ta.atr(length=14, append=True)
        data_slice.dropna(inplace=True)

        latest_row = data_slice.iloc[-1]
        latest_atr = latest_row['ATRr_14']
        price_at_prediction = latest_row['close']

        log.info(f"Price at prediction: {price_at_prediction:.2f}, ATR at prediction: {latest_atr:.2f}")

        # This data is passed to the AI, we are simulating it here
        live_financial_data = {"curatedData": {}, "rawDataSheet": {"symbol": TICKER_TO_TEST}}
        latest_indicators = {"ADX": f"{latest_row['ADX_14']:.2f}", "RSI": f"{latest_row['RSI_14']:.2f}"}
        market_context = {"CURRENT_MARKET_REGIME": "Neutral"}

    except Exception as e:
        log.error(f"Error during data preparation: {e}")
        return

    # --- 3. EXECUTION: Make one single API call with the refined prompt ---
    log.info("Making a single API call with the refined prompt...")
    analysis_result = analyzer.get_stock_analysis(
        live_financial_data=live_financial_data,
        latest_atr=latest_atr,
        model_name=config.PRO_MODEL,
        charts={},  # No charts needed for this logic test
        trading_horizon_text=config.VST_STRATEGY['horizon_text'],
        technical_indicators=latest_indicators,
        min_rr_ratio=config.VST_STRATEGY['min_rr_ratio'],
        market_context=market_context,
        options_data={}
    )

    # --- 4. VERIFICATION: Analyze the result ---
    if not analysis_result or 'tradePlan' not in analysis_result:
        log.error("AI analysis failed or returned an invalid structure.")
        return

    trade_plan = analysis_result.get('tradePlan', {})
    signal = analysis_result.get('signal')
    
    log.info(f"--- ✅ Test Complete. AI Analysis Result ---")
    log.info(f"Signal: {signal}")
    log.info(f"Trade Plan: {trade_plan}")

    if signal != 'HOLD':
        target = trade_plan.get('target', {}).get('price', 0)
        
        # The crucial check
        expected_reward = (config.VST_STRATEGY['min_rr_ratio'] * 2) * latest_atr
        actual_reward = abs(target - price_at_prediction)

        log.info(f"Expected Target Distance from Entry: ~{expected_reward:.2f}")
        log.info(f"  Actual Target Distance from Entry: ~{actual_reward:.2f}")
        
        # Allow for a small tolerance in the AI's math
        if abs(expected_reward - actual_reward) < 1.0:
            log.info("🎉 SUCCESS: The AI correctly calculated the target based on the new prompt.")
        else:
            log.warning("⚠️ WARNING: The AI's target calculation still deviates from the prompt's formula.")

if __name__ == "__main__":
    run_single_prompt_test()