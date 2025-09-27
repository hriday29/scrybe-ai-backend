# orchestrator_pure_ai.py
# FINAL VERSION - FOR EXPERIMENT 1: PURE AI BASELINE TEST
# This script is a modified version of the main_orchestrator.
# PURPOSE: To test the raw predictive power of the APEX AI model.
# KEY MODIFICATIONS:
# 1. QUANTITATIVE SCREENER REMOVED: The AI analyzes EVERY stock in the universe, every day.
# 2. VETO LOGIC REMOVED: Trades are based ONLY on the AI's signal and a conviction score threshold.
# 3. HISTORICAL CONTEXT PROMPTS REMOVED: AI prompt is simplified to focus only on market data.

import pandas as pd
import time
import config
from logger_config import log
import database_manager
import json
import os
import data_retriever
import pandas_ta as ta
from ai_analyzer import AIAnalyzer
import uuid
from utils import APIKeyManager
import argparse
from sector_analyzer import CORE_SECTOR_INDICES, BENCHMARK_INDEX
import random

# --- CONFIGURATION FOR THIS EXPERIMENT ---
MIN_SCRYBE_SCORE_FOR_TRADE = 60 # Only trust the AI if its conviction is 60 or higher.

def _build_backtest_context(ticker: str, point_in_time_data: pd.DataFrame, market_regime: str, nifty_data: pd.DataFrame) -> dict:
    """
    Constructs the complete, point-in-time context dictionary for the AI.
    This function is kept from the original orchestrator.
    """
    failed_indicators = []
    technicals = {}
    try:
        data = point_in_time_data.copy()
        
        # Calculate indicators individually for robustness
        try: data = data.join(data.ta.macd())
        except Exception: failed_indicators.append("MACD")
        
        try: data = data.join(data.ta.bbands())
        except Exception: failed_indicators.append("Bollinger Bands")
        
        try: data = data.join(data.ta.supertrend())
        except Exception: failed_indicators.append("Supertrend")

        try: data = data.join(data.ta.rsi(length=14))
        except Exception: failed_indicators.append("RSI")
        
        try: data = data.join(data.ta.adx(length=14))
        except Exception: failed_indicators.append("ADX")
        
        latest_row = data.iloc[-1]
        
        technicals = {
            "daily_close": latest_row.get("close"),
            "RSI_14": f"{latest_row.get('RSI_14', 0):.2f}",
            "ADX_14_trend_strength": f"{latest_row.get('ADX_14', 0):.2f}",
            "MACD_status": {"value": f"{latest_row.get('MACD_12_26_9', 0):.2f}", "signal_line": f"{latest_row.get('MACDs_12_26_9', 0):.2f}"},
            "bollinger_bands": {"upper_band": f"{latest_row.get('BBU_20_2.0', 0):.2f}", "lower_band": f"{latest_row.get('BBL_20_2.0', 0):.2f}"},
            "supertrend_7_3": {"trend": "Uptrend" if latest_row.get('SUPERTd_7_3.0') == 1 else "Downtrend", "value": f"{latest_row.get('SUPERT_7_3.0', 0):.2f}"}
        }
        if failed_indicators:
            technicals["errors"] = f"Failed to calculate: {', '.join(failed_indicators)}"

    except Exception as e:
        log.error(f"Core indicator calculation failed for {ticker}: {e}")
        technicals = {"error": "Critical indicator calculation failure."}

    try:
        nifty_slice = nifty_data.loc[:point_in_time_data.index[-1]]
        nifty_5d_change = (nifty_slice['close'].iloc[-1] / nifty_slice['close'].iloc[-6] - 1) * 100
        stock_5d_change = (point_in_time_data['close'].iloc[-1] / point_in_time_data['close'].iloc[-6] - 1) * 100
        relative_strength = "Outperforming" if stock_5d_change > nifty_5d_change else "Underperforming"
    except (IndexError, KeyError):
        relative_strength = "Data Not Available"

    context = {
        "layer_1_macro_context": {"nifty_50_regime": market_regime},
        "layer_2_relative_strength": {"relative_strength_vs_nifty50": relative_strength},
        "layer_3_fundamental_moat": data_retriever.get_fundamental_proxies(point_in_time_data),
        "layer_4_technicals": technicals,
        "layer_5_options_sentiment": {"sentiment": "Unavailable in backtest"},
        "layer_6_news_catalyst": {"summary": "Unavailable in backtest"}
    }
    return context

def sanitize_context(context: dict) -> dict:
    """Sanitizes the context dictionary. Kept from the original."""
    return json.loads(json.dumps(context).replace('NaN', 'null'))

def run_pure_ai_simulation(batch_id: str, start_date: str, end_date: str, is_fresh_run: bool = False):
    log.info(f"### STARTING PURE AI SIMULATION FOR BATCH: {batch_id} ###")
    log.info(f"Period: {start_date} to {end_date}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'nifty50_historical_constituents.csv')
    constituents_df = pd.read_csv(csv_path)
    constituents_df['date'] = pd.to_datetime(constituents_df['date'])

    required_indices = list(CORE_SECTOR_INDICES.values()) + [BENCHMARK_INDEX]

    database_manager.init_db(purpose='scheduler')
    if is_fresh_run:
        log.warning("FRESH RUN ENABLED: Deleting all previous prediction and performance data.")
        database_manager.clear_scheduler_data()

    key_manager = APIKeyManager(api_keys=config.GEMINI_API_KEY_POOL)
    analyzer = AIAnalyzer(api_key=key_manager.get_key())
    point_in_time_data_cache = {}
    
    simulation_days = pd.bdate_range(start=start_date, end=end_date)
    
    for i, current_day in enumerate(simulation_days):
        day_str = current_day.strftime('%Y-%m-%d')
        log.info(f"\n--- Simulating Day {i+1}/{len(simulation_days)}: {day_str} ---")

        available_dates = constituents_df[constituents_df['date'] <= current_day]['date']
        if available_dates.empty: continue
        latest_constituent_date = available_dates.max()
        stock_universe_for_today = constituents_df[constituents_df['date'] == latest_constituent_date]['ticker'].tolist()
        
        tickers_to_load_today = stock_universe_for_today + required_indices
        for ticker in tickers_to_load_today:
            if ticker not in point_in_time_data_cache:
                data = data_retriever.get_historical_stock_data(ticker, end_date=end_date)
                if data is not None and len(data) > 252:
                    point_in_time_data_cache[ticker] = data

        nifty_data_slice = point_in_time_data_cache.get(BENCHMARK_INDEX)
        market_regime = data_retriever.calculate_regime_from_data(nifty_data_slice.loc[:current_day]) if nifty_data_slice is not None else "Neutral"
        
        # ==============================================================================
        # --- MODIFICATION 1: SCREENER REMOVED ---
        # Instead of screening, we analyze ALL stocks in the universe for that day.
        # ==============================================================================
        stocks_for_today = [(ticker, "Comprehensive Analysis") for ticker in stock_universe_for_today]
        log.info(f"PURE AI MODE: Analyzing all {len(stocks_for_today)} stocks in today's universe.")
        
        if not stocks_for_today: continue

        for ticker, screener_reason in stocks_for_today:
            point_in_time_data = point_in_time_data_cache.get(ticker)
            if point_in_time_data is None or len(point_in_time_data.loc[:day_str]) < 252:
                continue

            data_slice = point_in_time_data.loc[:day_str].copy()
            full_context = _build_backtest_context(ticker, data_slice, market_regime, nifty_data_slice)
            sanitized_context = sanitize_context(full_context)
            
            try:
                data_slice.ta.atr(length=14, append=True)
                atr_at_prediction = data_slice['ATRr_14'].iloc[-1]
            except Exception: atr_at_prediction = 0

            final_analysis = None
            try:
                # Simplified AI call, no historical feedback loops for this pure test
                final_analysis = analyzer.get_apex_analysis(
                    ticker=ticker, full_context=sanitized_context, screener_reason=screener_reason,
                    strategic_review=None, tactical_lookback=None, per_stock_history=None, model_name=config.PRO_MODEL
                )
            except Exception as e:
                log.error(f"AI analysis for {ticker} failed: {e}")
                continue

            time.sleep(10)

            if not final_analysis: continue
            
            original_signal = final_analysis.get('signal')
            scrybe_score = final_analysis.get('scrybeScore', 0)
            
            # ==============================================================================
            # --- MODIFICATION 2: VETO LOGIC REMOVED ---
            # The signal is based ONLY on the AI's output and our score threshold.
            # ==============================================================================
            final_signal = 'HOLD'
            if original_signal == 'BUY' and scrybe_score >= MIN_SCRYBE_SCORE_FOR_TRADE:
                final_signal = 'BUY'
            
            prediction_doc = {**final_analysis, 
                'signal': final_signal, 'ticker': ticker, 
                'prediction_date': current_day.to_pydatetime(), 
                'price_at_prediction': data_slice['close'].iloc[-1],
                'strategy': "Apex_PureAI_v1"
            }

            if final_signal == 'BUY' and atr_at_prediction > 0:
                entry_price = prediction_doc['price_at_prediction']
                active_strategy = config.APEX_SWING_STRATEGY
                risk_per_share = active_strategy['stop_loss_atr_multiplier'] * atr_at_prediction
                reward_per_share = risk_per_share * active_strategy['profit_target_rr_multiple']
                
                prediction_doc['tradePlan'] = {
                    "entryPrice": round(entry_price, 2),
                    "target": round(entry_price + reward_per_share, 2),
                    "stopLoss": round(entry_price - risk_per_share, 2)
                }
                prediction_doc['status'] = 'open'
                log.info(f"✅ PURE AI SIGNAL: Found actionable BUY for {ticker} with score {scrybe_score}.")
            else:
                prediction_doc['status'] = 'hold'

            database_manager.save_prediction_for_backtesting(prediction_doc, batch_id)

    log.info("\n--- ✅ PURE AI Dynamic Signal Generation Finished! ---")
    database_manager.close_db_connection()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PURE AI historical backtest signal generator.")
    parser.add_argument('--batch_id', required=True, help="A unique ID for this backtest run (e.g., 'PureAI_Q3_2025').")
    parser.add_argument('--start_date', required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument('--end_date', required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument('--fresh_run', type=lambda x: (str(x).lower() == 'true'), default=False, help="Set to True to clear old data before running.")
    args = parser.parse_args()
    
    run_pure_ai_simulation(
        batch_id=args.batch_id, 
        start_date=args.start_date, 
        end_date=args.end_date, 
        is_fresh_run=args.fresh_run
    )