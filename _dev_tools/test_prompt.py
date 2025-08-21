# test_prompt.py (MODIFIED FOR 'Technical Analyst' SPECIALIST)

import pandas as pd
import config
from logger_config import log
import data_retriever
from ai_analyzer import AIAnalyzer
import json
import pandas_ta as ta

def run_single_prompt_test():
    """
    A fast, single-API-call test to validate our new 'get_simple_momentum_signal'
    function, acting as our "Technical Analyst" specialist.
    """
    # --- 1. SETUP: Define our single test case ---
    TICKER_TO_TEST = "LT.NS"
    DATE_TO_TEST = "2025-07-15" # A day within your failing backtest period
    
    log.info(f"--- 🧪 Running Specialist Laboratory Test ---")
    log.info(f"Testing 'Technical Analyst' on {TICKER_TO_TEST} for date {DATE_TO_TEST}")

    # --- 2. PREPARATION: Recreate the exact data the AI will see ---
    try:
        analyzer = AIAnalyzer(api_key=config.GEMINI_API_KEY_POOL[0]) # Use a key from the pool

        full_historical_data = data_retriever.get_historical_stock_data(TICKER_TO_TEST, end_date=DATE_TO_TEST)
        if full_historical_data is None or len(full_historical_data) < 100:
            log.error("Could not get enough historical data for the test.")
            return

        data_slice = full_historical_data.loc[:DATE_TO_TEST].copy()

        # --- EDIT: Added EMA calculations for our new criteria ---
        data_slice.ta.ema(length=20, append=True)
        data_slice.ta.ema(length=50, append=True)
        data_slice.ta.rsi(length=14, append=True)
        data_slice.ta.adx(length=14, append=True)
        data_slice.ta.atr(length=14, append=True)
        data_slice.dropna(inplace=True)

        latest_row = data_slice.iloc[-1]
        price_at_prediction = latest_row['close']
        atr_at_prediction = latest_row['ATRr_14']

        # --- EDIT: Assembled a simple, focused dictionary for the new specialist ---
        technical_indicators_for_ai = {
            "ADX_14": round(latest_row['ADX_14'], 2),
            "RSI_14": round(latest_row['RSI_14'], 2),
            "close_price": round(latest_row['close'], 2),
            "EMA_20": round(latest_row['EMA_20'], 2),
            "EMA_50": round(latest_row['EMA_50'], 2)
        }
        log.info(f"Data being sent to AI: {json.dumps(technical_indicators_for_ai)}")

    except Exception as e:
        log.error(f"Error during data preparation: {e}")
        return

    # --- 3. EXECUTION: Make one single API call to the new function ---
    log.info("Making a single API call to the 'get_simple_momentum_signal'...")
    
    # --- EDIT: Changed the function call to our new specialist ---
    analysis_result = analyzer.get_simple_momentum_signal(
        ticker=TICKER_TO_TEST,
        technical_indicators=technical_indicators_for_ai
    )
    
    # --- 4. VERIFICATION & SIMULATION ---
    if not analysis_result or 'signal' not in analysis_result:
        log.error("AI analysis failed or returned an invalid structure.")
        return

    log.info("--- ✅ AI Analysis Received Successfully ---")
    log.info(f"Signal: {analysis_result.get('signal')}")
    log.info(f"Conviction Score: {analysis_result.get('convictionScore')}")
    log.info(f"Rationale: \"{analysis_result.get('rationale')}\"")

    # --- EDIT: Simulate the NEW deterministic trade plan calculation ---
    if analysis_result.get('signal') in ['BUY', 'SELL']:
        log.info("--- ⚙️ Simulating Deterministic Trade Plan Calculation ---")
        signal = analysis_result['signal']
        
        # Risk is defined as 2x ATR
        risk_per_share = 2 * atr_at_prediction
        
        # Stop loss is placed based on risk
        stop_loss_price = price_at_prediction - risk_per_share if signal == 'BUY' else price_at_prediction + risk_per_share
        
        # Target is calculated for a fixed 1.5 Risk-to-Reward Ratio
        reward_per_share = risk_per_share * 1.5
        target_price = price_at_prediction + reward_per_share if signal == 'BUY' else price_at_prediction - reward_per_share
        
        final_trade_plan = {
            "entryPrice": round(price_at_prediction, 2),
            "target": round(target_price, 2),
            "stopLoss": round(stop_loss_price, 2),
            "riskRewardRatio": 1.5
        }
        log.info(f"Final Calculated Trade Plan: {json.dumps(final_trade_plan)}")
    else:
        log.info("Signal is 'HOLD', no trade plan generated.")

    log.info("--- 🎉 Pre-Flight Check Successful! ---")


if __name__ == "__main__":
    run_single_prompt_test()