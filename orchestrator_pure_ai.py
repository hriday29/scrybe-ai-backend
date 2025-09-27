# orchestrator_hybrid_screener_ai.py
# EXPERIMENT 2: HYBRID APPROACH (SCREENER + AI)
# This script is a modified version of the main_orchestrator.
# PURPOSE: To test the performance of the Quantitative Screener combined with the AI's analysis,
# but WITHOUT the final hard-coded veto rules (VIX, market regime checks, etc.).
# KEY MODIFICATIONS:
# 1. QUANTITATIVE SCREENER: REMAINS ACTIVE.
# 2. VETO LOGIC REMOVED: Replaced with a simple Scrybe Score threshold check.

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
import market_regime_analyzer
import sector_analyzer
import quantitative_screener
from config import PORTFOLIO_CONSTRAINTS
from collections import deque
from sector_analyzer import CORE_SECTOR_INDICES, BENCHMARK_INDEX
import random

# --- CONFIGURATION FOR THIS EXPERIMENT ---
MIN_SCRYBE_SCORE_FOR_TRADE = 60 # Only trust the AI if its conviction is 60 or higher.

# (All helper functions from main_orchestrator.py are included and unchanged)
def _build_backtest_context(ticker: str, point_in_time_data: pd.DataFrame, market_regime: str, nifty_data: pd.DataFrame) -> dict:
    # This function is identical to the one in main_orchestrator.py
    failed_indicators = []
    technicals = {}
    try:
        data = point_in_time_data.copy()
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
            "daily_close": latest_row.get("close"), "RSI_14": f"{latest_row.get('RSI_14', 0):.2f}",
            "ADX_14_trend_strength": f"{latest_row.get('ADX_14', 0):.2f}",
            "MACD_status": {"value": f"{latest_row.get('MACD_12_26_9', 0):.2f}", "signal_line": f"{latest_row.get('MACDs_12_26_9', 0):.2f}"},
            "bollinger_bands": {"upper_band": f"{latest_row.get('BBU_20_2.0', 0):.2f}", "lower_band": f"{latest_row.get('BBL_20_2.0', 0):.2f}"},
            "supertrend_7_3": {"trend": "Uptrend" if latest_row.get('SUPERTd_7_3.0') == 1 else "Downtrend", "value": f"{latest_row.get('SUPERT_7_3.0', 0):.2f}"}
        }
        if failed_indicators: technicals["errors"] = f"Failed to calculate: {', '.join(failed_indicators)}"
    except Exception as e:
        log.error(f"Core indicator calculation failed for {ticker}: {e}")
        technicals = {"error": "Critical indicator calculation failure."}
    try:
        nifty_slice = nifty_data.loc[:point_in_time_data.index[-1]]
        nifty_5d_change = (nifty_slice['close'].iloc[-1] / nifty_slice['close'].iloc[-6] - 1) * 100
        stock_5d_change = (point_in_time_data['close'].iloc[-1] / point_in_time_data['close'].iloc[-6] - 1) * 100
        relative_strength = "Outperforming" if stock_5d_change > nifty_5d_change else "Underperforming"
    except (IndexError, KeyError): relative_strength = "Data Not Available"
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
    return json.loads(json.dumps(context).replace('NaN', 'null'))

# (The rest of the script is identical to main_orchestrator until the Veto Logic section)

def run_hybrid_simulation(batch_id: str, start_date: str, end_date: str, is_fresh_run: bool = False):
    log.info(f"### STARTING HYBRID (SCREENER + AI) SIMULATION FOR BATCH: {batch_id} ###")
    # --- This setup part is identical to main_orchestrator.py ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'nifty50_historical_constituents.csv')
    constituents_df = pd.read_csv(csv_path)
    constituents_df['date'] = pd.to_datetime(constituents_df['date'])
    required_indices = list(CORE_SECTOR_INDICES.values()) + [BENCHMARK_INDEX]
    database_manager.init_db(purpose='scheduler')
    if is_fresh_run:
        database_manager.clear_scheduler_data()
    key_manager = APIKeyManager(api_keys=config.GEMINI_API_KEY_POOL)
    analyzer = AIAnalyzer(api_key=key_manager.get_key())
    point_in_time_data_cache = {}
    vix_data = data_retriever.get_historical_stock_data("^INDIAVIX", end_date=end_date)
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

        approved_stock_data_cache = {t: point_in_time_data_cache[t] for t in stock_universe_for_today if t in point_in_time_data_cache}
        full_context_data_cache = approved_stock_data_cache.copy()
        for ticker in required_indices:
            if ticker in point_in_time_data_cache:
                full_context_data_cache[ticker] = point_in_time_data_cache[ticker]
        
        nifty_data_slice = full_context_data_cache.get(BENCHMARK_INDEX)
        market_regime = data_retriever.calculate_regime_from_data(nifty_data_slice.loc[:current_day]) if nifty_data_slice is not None else "Neutral"
        strong_sectors = sector_analyzer.get_top_performing_sectors(full_context_data_cache, current_day)
        vix_slice = vix_data.loc[:current_day] if vix_data is not None else None
        volatility_regime = data_retriever.get_volatility_regime(vix_slice)

        # ==============================================================================
        # --- SCREENER LOGIC REMAINS ACTIVE ---
        # We are still using the screener to filter the universe down to a manageable list.
        # ==============================================================================
        stocks_for_today = []
        if market_regime == "Bearish":
            stocks_for_today = quantitative_screener.screen_for_mean_reversion(strong_sectors, approved_stock_data_cache, current_day)
        elif market_regime == "Bullish":
            if volatility_regime == "High-Risk":
                stocks_for_today = quantitative_screener.screen_for_pullbacks(strong_sectors, approved_stock_data_cache, current_day)
            else:
                stocks_for_today = quantitative_screener.screen_for_momentum(strong_sectors, approved_stock_data_cache, current_day)
        elif market_regime == "Neutral":
            stocks_for_today = quantitative_screener.screen_for_mean_reversion(strong_sectors, approved_stock_data_cache, current_day)
        
        if not stocks_for_today:
            log.warning("No stocks passed the screener today. Skipping.")
            continue
        log.info(f"Screener found {len(stocks_for_today)} potential candidates.")

        for ticker, screener_reason in stocks_for_today:
            point_in_time_data = approved_stock_data_cache.get(ticker).loc[:day_str].copy()
            if len(point_in_time_data) < 252: continue

            full_context = _build_backtest_context(ticker, point_in_time_data, market_regime, nifty_data_slice)
            sanitized_full_context = sanitize_context(full_context)
            
            point_in_time_data.ta.atr(length=14, append=True)
            atr_at_prediction = point_in_time_data['ATRr_14'].iloc[-1]

            final_analysis = None
            try:
                final_analysis = analyzer.get_apex_analysis(
                    ticker, sanitized_full_context, strategic_review=None, tactical_lookback=None,
                    per_stock_history=None, model_name=config.PRO_MODEL, screener_reason=screener_reason
                )
            except Exception as e:
                log.error(f"AI analysis for {ticker} failed: {e}")
                continue

            time.sleep(10) 

            if not final_analysis: continue
            
            original_signal = final_analysis.get('signal')
            scrybe_score = final_analysis.get('scrybeScore', 0)
            
            # ==============================================================================
            # --- MODIFICATION: VETO LOGIC IS REMOVED ---
            # The complex checks on VIX, regime, etc., are gone. We now only check the AI's signal and score.
            # ==============================================================================
            final_signal = 'HOLD' # Default to HOLD
            
            if original_signal == 'BUY' and scrybe_score >= MIN_SCRYBE_SCORE_FOR_TRADE:
                final_signal = 'BUY'
            
            prediction_doc = {**final_analysis, 
                'signal': final_signal, 'ticker': ticker, 
                'prediction_date': current_day.to_pydatetime(), 
                'price_at_prediction': point_in_time_data['close'].iloc[-1],
                'strategy': "Apex_Hybrid_ScreenerAI_v1"
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
                log.info(f"✅ HYBRID SIGNAL: Found actionable BUY for {ticker} with score {scrybe_score}.")
            else:
                prediction_doc['status'] = 'hold'

            database_manager.save_prediction_for_backtesting(prediction_doc, batch_id)

    log.info("\n--- ✅ HYBRID (SCREENER + AI) Simulation Finished! ---")
    database_manager.close_db_connection()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the HYBRID (Screener + AI) historical backtest.")
    parser.add_argument('--batch_id', required=True, help="A unique ID for this backtest run (e.g., 'Hybrid_Q3_2025').")
    parser.add_argument('--start_date', required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument('--end_date', required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument('--fresh_run', type=lambda x: (str(x).lower() == 'true'), default=False, help="Set to True to clear old data before running.")
    args = parser.parse_args()
    
    run_hybrid_simulation(
        batch_id=args.batch_id, 
        start_date=args.start_date, 
        end_date=args.end_date, 
        is_fresh_run=args.fresh_run
    )