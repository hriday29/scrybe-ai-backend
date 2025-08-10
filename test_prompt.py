# In test_prompt.py

import pandas as pd
import config
from logger_config import log
import data_retriever
import technical_analyzer
from ai_analyzer import AIAnalyzer
from datetime import datetime

def run_single_prompt_test():
    """
    A fast, single-API-call test to validate the final, robust prompt and calculation architecture.
    """
    # --- 1. SETUP: Define our single test case ---
    TICKER_TO_TEST = "TATASTEEL.NS"
    DATE_TO_TEST = "2025-06-06"
    
    log.info(f"--- 🧪 Running Strategy Laboratory Test ---")
    log.info(f"Testing new prompt on {TICKER_TO_TEST} for date {DATE_TO_TEST}")

    # --- 2. PREPARATION: Recreate the exact data the AI will see ---
    try:
        analyzer = AIAnalyzer(api_key=config.GEMINI_API_KEY)

        full_historical_data = data_retriever.get_historical_stock_data(TICKER_TO_TEST, end_date=DATE_TO_TEST)
        if full_historical_data is None or len(full_historical_data) < 100:
            log.error("Could not get enough historical data for the test.")
            return

        data_slice = full_historical_data.loc[:DATE_TO_TEST].copy()

        data_slice.ta.bbands(length=20, append=True)
        data_slice.ta.rsi(length=14, append=True)
        data_slice.ta.macd(fast=12, slow=26, signal=9, append=True)
        data_slice.ta.adx(length=14, append=True)
        data_slice.ta.atr(length=14, append=True)
        data_slice.dropna(inplace=True)

        latest_row = data_slice.iloc[-1]
        price_at_prediction = latest_row['close']
        atr_at_prediction = latest_row['ATRr_14']

        log.info(f"Price at prediction: {price_at_prediction:.2f}, ATR at prediction: {atr_at_prediction:.2f}")

        live_financial_data = {"curatedData": {}, "rawDataSheet": {"symbol": TICKER_TO_TEST}}
        latest_indicators = {"ADX": f"{latest_row['ADX_14']:.2f}", "RSI": f"{latest_row['RSI_14']:.2f}", "Bollinger Band Width Percent": f"{(latest_row['BBU_20_2.0'] - latest_row['BBL_20_2.0']) / latest_row['BBM_20_2.0'] * 100:.2f}"}
        market_context = {"CURRENT_MARKET_REGIME": "Neutral"}

    except Exception as e:
        log.error(f"Error during data preparation: {e}")
        return

    # --- 3. EXECUTION: Make one single API call ---
    log.info("Making a single API call with the refined prompt...")
    analysis_result = analyzer.get_stock_analysis(
        live_financial_data=live_financial_data,
        latest_atr=atr_at_prediction,
        model_name=config.PRO_MODEL,
        charts={},
        trading_horizon_text=config.VST_STRATEGY['horizon_text'],
        technical_indicators=latest_indicators,
        min_rr_ratio=config.VST_STRATEGY['min_rr_ratio'],
        market_context=market_context,
        options_data={}
    )

    # --- 4. VERIFICATION & SIMULATION ---
    # NEW, CORRECT VALIDATION CHECK
    if not analysis_result or 'signal' not in analysis_result:
        log.error("AI analysis failed or returned an invalid structure.")
        return

    log.info("--- ✅ AI Analysis Received Successfully ---")
    log.info(f"Signal: {analysis_result.get('signal')}")
    log.info(f"Scrybe Score: {analysis_result.get('scrybeScore')}")
    log.info(f"Verdict: {analysis_result.get('analystVerdict')}")

    # SIMULATE the Python-based trade plan calculation
    if analysis_result.get('signal') in ['BUY', 'SELL']:
        log.info("--- ⚙️ Simulating Deterministic Trade Plan Calculation ---")
        signal = analysis_result['signal']
        rr_ratio = config.VST_STRATEGY['min_rr_ratio']
        
        stop_loss_price = price_at_prediction - (2 * atr_at_prediction) if signal == 'BUY' else price_at_prediction + (2 * atr_at_prediction)
        target_price = price_at_prediction + ((2 * atr_at_prediction) * rr_ratio) if signal == 'BUY' else price_at_prediction - ((2 * atr_at_prediction) * rr_ratio)
        
        final_trade_plan = {
            "entryPrice": round(price_at_prediction, 2),
            "target": round(target_price, 2),
            "stopLoss": round(stop_loss_price, 2),
            "riskRewardRatio": rr_ratio
        }
        log.info(f"Final Calculated Trade Plan: {final_trade_plan}")
    else:
        log.info("Signal is 'HOLD', no trade plan generated.")

    log.info("--- 🎉 Pre-Flight Check Successful! ---")


if __name__ == "__main__":
    run_single_prompt_test()