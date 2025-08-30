# main_orchestrator.py (FINAL APEX-AWARE VERSION)
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
from quantitative_screener import generate_dynamic_watchlist, _passes_fundamental_health_check
from config import PORTFOLIO_CONSTRAINTS
from collections import deque

def sanitize_context(context: dict) -> dict:
    """Sanitizes the context dictionary to replace null-like values."""
    sanitized_context = {}
    for layer, details in context.items():
        if isinstance(details, dict):
            sanitized_details = {}
            for k, v in details.items():
                # Check for various null-like or placeholder strings
                if not v or str(v).strip().lower() in ["unavailable", "n/a", "none", "null", "unavailable in backtest"]:
                    sanitized_details[k] = "Data Not Available"
                else:
                    sanitized_details[k] = v
            sanitized_context[layer] = sanitized_details
        else:
            # If the layer's value is not a dictionary, keep it as is
            sanitized_context[layer] = details
    return sanitized_context

def _get_30_day_performance_review(current_day: pd.Timestamp, batch_id: str) -> str:
    # ... [This function remains unchanged, no need to copy it again] ...
    thirty_days_prior = current_day - pd.Timedelta(days=30)
    query = {"batch_id": batch_id, "close_date": {"$gte": thirty_days_prior.to_pydatetime(), "$lt": current_day.to_pydatetime()}}
    recent_trades = list(database_manager.performance_collection.find(query))
    if not recent_trades: return "No trading history in the last 30 days."
    df = pd.DataFrame(recent_trades)
    total_signals = len(df)
    win_rate = (df['net_return_pct'] > 0).mean() * 100
    gross_profit = df[df['net_return_pct'] > 0]['net_return_pct'].sum()
    gross_loss = abs(df[df['net_return_pct'] < 0]['net_return_pct'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    review = (f"30-Day Performance Review:\n- Total Signals: {total_signals}\n- Win Rate: {win_rate:.1f}%\n- Profit Factor: {profit_factor:.2f}")
    return review

def _get_1_day_tactical_lookback(current_day: pd.Timestamp, ticker: str, batch_id: str) -> str:
    # ... [This function remains unchanged, no need to copy it again] ...
    previous_trading_day = current_day - pd.Timedelta(days=3) # Look back 3 days to be safe
    query = {"batch_id": batch_id, "ticker": ticker, "prediction_date": {"$gte": previous_trading_day.to_pydatetime(), "$lt": current_day.to_pydatetime()}}
    last_analysis = database_manager.predictions_collection.find_one(query, sort=[("prediction_date", -1)])
    if not last_analysis: return "No analysis for this stock on the previous trading day."
    lookback = (f"Previous Day's Note ({ticker}):\n- Signal: {last_analysis.get('signal', 'N/A')}\n- Scrybe Score: {last_analysis.get('scrybeScore', 0)}\n"
                f"- Key Insight: \"{last_analysis.get('keyInsight', 'N/A')}\"")
    return lookback

def _get_per_stock_trade_history(ticker: str, batch_id: str, current_day: pd.Timestamp) -> str:
    # ... [This function remains unchanged, no need to copy it again] ...
    query = { "batch_id": batch_id, "ticker": ticker, "close_date": {"$lt": current_day.to_pydatetime()} }
    recent_trades = list(database_manager.performance_collection.find(query).sort("close_date", -1).limit(3))
    if not recent_trades: return "No recent trade history for this stock."
    history_lines = [f"{i+1}. Signal: {t.get('signal')}, Outcome: {t.get('net_return_pct'):.2f}% ({t.get('closing_reason')})" for i, t in enumerate(recent_trades)]
    return "\n".join(history_lines)

def run_simulation(batch_id: str, start_date: str, end_date: str, stock_universe: list, is_fresh_run: bool = False):
    log.info(f"### STARTING APEX SIMULATION FOR BATCH: {batch_id} ###")
    log.info(f"Original Period Requested: {start_date} to {end_date}")
    database_manager.init_db(purpose='scheduler')

    effective_start_date = start_date

    if is_fresh_run:
        log.warning("FRESH RUN ENABLED: Deleting all previous data.")
        database_manager.clear_scheduler_data()
    else:
        # --- NEW: AUTOMATIC RESUME LOGIC ---
        log.info("Checking for previous progress for this batch (Fresh Run is OFF)...")
        last_prediction = database_manager.predictions_collection.find_one(
            {"batch_id": batch_id},
            sort=[("prediction_date", -1)]
        )

        if last_prediction:
            last_date = pd.to_datetime(last_prediction['prediction_date'])
            # Start from the day AFTER the last saved date
            next_day_to_run = last_date + pd.Timedelta(days=1)
            effective_start_date = next_day_to_run.strftime('%Y-%m-%d')
            log.warning(f"✅ Previous progress found. Resuming simulation from {effective_start_date}")
        else:
            log.info("No previous progress found for this batch. Starting from the beginning.")
        # --- END OF NEW LOGIC ---

    try:
        key_manager = APIKeyManager(api_keys=config.GEMINI_API_KEY_POOL)
        analyzer = AIAnalyzer(api_key=key_manager.get_key())
        log.info("Pre-loading all historical data...")
        full_historical_data_cache = {ticker: data for ticker in stock_universe if (data := data_retriever.get_historical_stock_data(ticker, end_date=end_date)) is not None and len(data) > 252}
        nifty_data = data_retriever.get_historical_stock_data("^NSEI", end_date=end_date)
        vix_data = data_retriever.get_historical_stock_data("^INDIAVIX", end_date=end_date)
        log.info("✅ All data pre-loading complete.")

        # --- START: NEW FUNDAMENTAL PRE-FLIGHT CHECK ---
        log.info("--- Running Fundamental Pre-Flight Check on Full Universe ---")
        fundamentally_approved_stocks = {} # CHANGE: Use a dictionary to store data
        for ticker in stock_universe:
            log.info(f"  -> Checking fundamentals for {ticker}...")
            is_approved, fundamentals = _passes_fundamental_health_check(ticker)
            if is_approved:
                fundamentally_approved_stocks[ticker] = fundamentals # Store the metrics
                log.info(f"    - ✅ PASS: {ticker} is fundamentally approved.")

        log.info(f"✅ Fundamental Pre-Flight Check complete. {len(fundamentally_approved_stocks)}/{len(stock_universe)} stocks passed.")
        if not fundamentally_approved_stocks:
            log.fatal("No stocks passed the fundamental pre-flight check. Halting simulation.")
            return
        # --- END: NEW FUNDAMENTAL PRE-FLIGHT CHECK ---

    except Exception as e:
        log.fatal(f"Failed during pre-run initialization. Error: {e}")
        return
    
    simulation_days = pd.bdate_range(start=effective_start_date, end=end_date)

    total_stocks_processed_by_screener = 0
    total_stocks_passed_screener = 0
    total_ai_buy_sell_signals = 0
    total_trades_executed = 0

    if not simulation_days.empty:
        log.info(f"\n--- Starting simulation for {len(simulation_days)} trading days ---")
        for i, current_day in enumerate(simulation_days):
            day_str = current_day.strftime('%Y-%m-%d')
            log.info(f"\n--- Simulating Day {i+1}/{len(simulation_days)}: {day_str} ---")
            
            try:
                latest_vix = vix_data.loc[:day_str].iloc[-1]['close']
                market_is_high_risk = latest_vix > config.HIGH_RISK_VIX_THRESHOLD
                if market_is_high_risk:
                    log.warning(f"!! MASTER RISK OVERLAY ENGAGED !! VIX is at {latest_vix:.2f}. No new BUY or SELL signals will be processed today.")
            except (KeyError, IndexError):
                market_is_high_risk = False
            
            market_regime = data_retriever.calculate_regime_from_data(nifty_data.loc[:day_str])
            strong_sectors = sector_analyzer.get_top_performing_sectors(point_in_time=current_day)
            approved_stock_data_cache = {ticker: full_historical_data_cache[ticker] for ticker in fundamentally_approved_stocks if ticker in full_historical_data_cache}
            stocks_for_today = generate_dynamic_watchlist(strong_sectors, approved_stock_data_cache, current_day)
            total_stocks_processed_by_screener += len(fundamentally_approved_stocks) # Or whatever list goes into the screener
            total_stocks_passed_screener += len(stocks_for_today)
            
            if not stocks_for_today:
                log.warning("The dynamic funnel returned no stocks for today. Proceeding to next day.")
                continue

            # --- PHASE 1: COLLECT ALL POTENTIAL TRADES FOR THE DAY ---
            potential_trades_today = []
            for ticker, screener_reason in stocks_for_today:

                final_analysis = None
                max_attempts = len(config.GEMINI_API_KEY_POOL)
                current_attempt = 0
                # --- Prepare context once, outside the retry loops ---
                point_in_time_data = full_historical_data_cache.get(ticker).loc[:day_str].copy()
                if len(point_in_time_data) < 252:
                    log.warning(f"Skipping {ticker} due to insufficient historical data on {day_str}.")
                    continue

                strategic_review = _get_30_day_performance_review(current_day, batch_id)
                tactical_lookback = _get_1_day_tactical_lookback(current_day, ticker, batch_id)
                per_stock_history = _get_per_stock_trade_history(ticker, batch_id, current_day)
                latest_row = point_in_time_data.iloc[-1]
                point_in_time_data.ta.atr(length=14, append=True)
                atr_at_prediction = point_in_time_data['ATRr_14'].iloc[-1]
                point_in_time_data.ta.adx(length=14, append=True)
                adx_at_prediction = point_in_time_data['ADX_14'].iloc[-1]
                nifty_5d_change = (nifty_data.loc[:day_str]['close'].iloc[-1] / nifty_data.loc[:day_str]['close'].iloc[-6] - 1) * 100
                stock_5d_change = (latest_row['close'] / point_in_time_data['close'].iloc[-6] - 1) * 100
                point_in_time_data.ta.ema(length=20, append=True)
                point_in_time_data.ta.rsi(length=14, append=True)
                point_in_time_data.ta.bbands(length=20, append=True)
                point_in_time_data.ta.macd(append=True)
                point_in_time_data.ta.supertrend(append=True) # Using default settings
                point_in_time_data.dropna(inplace=True)
                latest_row = point_in_time_data.iloc[-1] # Refresh latest_row after ALL calculations
                rsi_value = latest_row.get('RSI_14', 50)
                macd_histogram = latest_row.get('MACDh_12_26_9', 0)
                # Trend and Volatility
                ema_20 = latest_row.get('EMA_20')
                bbp_value = latest_row.get('BBP_20_2.0', 0.5)
                price_vs_ema20_pct = ((latest_row['close'] - ema_20) / ema_20) * 100 if ema_20 else 0
                supertrend_direction = "Uptrend" if latest_row.get('SUPERTd_7_3.0') == 1 else "Downtrend"
                                
                technical_health_metrics = data_retriever.get_technical_health_metrics(point_in_time_data)

                fundamental_metrics = fundamentally_approved_stocks.get(ticker, {})
                roe_val = fundamental_metrics.get('returnOnEquity')
                roe_str = f"{roe_val:.2%}" if isinstance(roe_val, (int, float)) else "N/A"

                # Define the fundamental summary string
                fundamental_summary = (
                    f"Trailing P/E: {fundamental_metrics.get('trailingPE', 'N/A')}, "
                    f"Debt/Equity: {fundamental_metrics.get('debtToEquity', 'N/A')}, "
                    f"Return on Equity: {roe_str}"
                )

                full_context = {
                    "layer_1_macro_context": {"nifty_50_regime": market_regime},
                    "layer_2_relative_strength": {"relative_strength_vs_nifty50": "Outperforming" if stock_5d_change > nifty_5d_change else "Underperforming"},
                    "layer_3_fundamental_moat": {"summary": fundamental_summary},
                    "layer_4_technicals": {
                        # Section 1: High-level health and stability metrics
                        "overview_and_health": technical_health_metrics,

                        # Section 2: Primary, longer-term trend and momentum signals
                        "primary_trend_and_momentum": {
                            "adx_14_trend_strength": f"{adx_at_prediction:.2f}",
                            "supertrend_7_3_direction": supertrend_direction,
                            "macd_histogram_momentum": f"{macd_histogram:.2f}"
                        },
                        
                        # Section 3: Short-term price action, oscillators, and volatility
                        "short_term_oscillators_and_volatility": {
                            "daily_close": latest_row['close'],
                            "rsi_14": f"{rsi_value:.2f}",
                            "price_position_vs_20ema_pct": f"{price_vs_ema20_pct:.2f}%",
                            "bollinger_band_percentage": f"{bbp_value:.2f}"
                        }
                    },
                    "layer_5_options_sentiment": {"sentiment": "Unavailable in backtest"},
                    "layer_6_news_catalyst": {"summary": "Unavailable in backtest"}
                }
                
                full_context["layer_4_technicals"].update(technical_health_metrics)

                sanitized_full_context = sanitize_context(full_context)
                # --- End of context preparation ---

                # --- Primary Model Loop (Pro with Key Rotation) ---
                log.info(f"--- Analyzing Ticker: {ticker} with Primary Model ({config.PRO_MODEL}) ---")
                while current_attempt < max_attempts:
                    try:
                        final_analysis = analyzer.get_apex_analysis(
                            ticker, sanitized_full_context, strategic_review, tactical_lookback, per_stock_history, model_name=config.PRO_MODEL, screener_reason=screener_reason
                        )
                        if final_analysis:
                            log.info(f"✅ Successfully got analysis for {ticker} using PRIMARY model on key #{key_manager.current_index + 1}.")
                            final_analysis['modelUsed'] = 'pro'
                            break # Success, exit the loop
                            
                    except Exception as e:
                        log.error(f"Primary model attempt #{current_attempt + 1} for {ticker} failed. Error: {e}")
                        # Check for specific, recoverable errors
                        error_str = str(e).lower()
                        if "429" in error_str or "quota" in error_str or "500" in error_str or isinstance(e, ValueError):
                            log.warning("Error is recoverable, rotating API key and retrying...")
                            analyzer = AIAnalyzer(api_key=key_manager.rotate_key())
                            current_attempt += 1
                            time.sleep(5) # A small delay before retrying with new key
                        else:
                            log.error("Encountered a non-recoverable error. Breaking loop.")
                            break # Break on unrecoverable errors

                # --- Fallback Model (Flash) if all Primary attempts failed ---
                if not final_analysis:
                    log.warning(f"All primary model attempts failed. Switching to FALLBACK model ({config.FLASH_MODEL}) for {ticker}.")
                    try:
                        # We can try the fallback with the current key
                        final_analysis = analyzer.get_apex_analysis(
                            ticker, sanitized_full_context, strategic_review, tactical_lookback, per_stock_history, model_name=config.FLASH_MODEL, screener_reason=screener_reason
                        )
                        if final_analysis:
                            log.info(f"✅ Successfully got analysis for {ticker} using FALLBACK model.")
                            final_analysis['modelUsed'] = 'flash'
                    except Exception as e:
                        log.error(f"CRITICAL FAILURE: Fallback model also failed for {ticker}. Error: {e}")
                        final_analysis = None

                # --- Pacing Delay (remains the same) ---
                log.info("Pacing API calls with a 35-second delay to respect rate limits.")
                time.sleep(35)

                # If all attempts failed, skip this ticker
                if not final_analysis:
                    log.warning(f"Skipping {ticker} for day {day_str} after all attempts failed.")
                    continue

                original_signal = final_analysis.get('signal')
                if original_signal in ['BUY', 'SELL']:
                    total_ai_buy_sell_signals += 1
                    # --- VETO BLOCK ---
                    # If the market is in a high-risk state, veto ALL new signals (BUY or SELL).
                    if original_signal in ['BUY', 'SELL'] and market_is_high_risk:
                        log.warning(
                            f"VETOED SIGNAL for {ticker}: A '{original_signal}' signal was generated, "
                            f"but the Master Risk Overlay (VIX) is engaged. Skipping."
                        )
                        continue # Skip the rest of the logic for this ticker
                    # --- END OF VETO BLOCK ---
                    # This section remains largely the same, calculating all possible SL/TP levels
                    entry_price = latest_row['close']
                    ai_predicted_gain = final_analysis.get('predicted_gain_pct', 5.0)
                    ai_target_price = entry_price * (1 + (ai_predicted_gain / 100.0))
                    atr_risk_per_share_A = config.STRATEGY_ATR_3R['stop_loss_atr_multiplier'] * atr_at_prediction
                    atr_stop_loss_A = entry_price - atr_risk_per_share_A
                    atr_target_A = entry_price + (atr_risk_per_share_A * config.STRATEGY_ATR_3R['profit_target_rr_multiple'])
                    atr_risk_per_share_B = config.STRATEGY_AI_ATR['stop_loss_atr_multiplier'] * atr_at_prediction
                    atr_stop_loss_B = entry_price - atr_risk_per_share_B
                    lookback = config.STRATEGY_AI_STRUCTURE['swing_low_lookback_period']
                    buffer_mult = 1 - (config.STRATEGY_AI_STRUCTURE['structure_stop_buffer_pct'] / 100.0)
                    recent_low = data_retriever.get_recent_swing_low(point_in_time_data, lookback)
                    structure_stop_loss_C = recent_low * buffer_mult if recent_low > 0 else 0

                    # Create the base prediction document from the AI analysis
                    prediction_doc = final_analysis.copy()

                    # Save ALL calculated data points to the trade plan
                    prediction_doc['tradePlan'] = {
                        "entryPrice": round(entry_price, 2),
                        "ai_target": round(ai_target_price, 2),
                        "atr_target_A": round(atr_target_A, 2),
                        "atr_stop_loss_A": round(atr_stop_loss_A, 2),
                        "atr_stop_loss_B": round(atr_stop_loss_B, 2),
                        "structure_stop_loss_C": round(structure_stop_loss_C, 2),
                        "halfway_profit_price": entry_price + (ai_target_price - entry_price) * 0.5
                    }

                    # Add all necessary context for the backtester's veto logic
                    prediction_doc.update({
                        'ticker': ticker,
                        'prediction_date': current_day.to_pydatetime(),
                        'price_at_prediction': entry_price,
                        'status': 'open', # Status is always open now
                        'strategy': 'ApexPredator_Multi_v2',
                        # --- CRITICAL ADDITION ---
                        'market_regime_at_prediction': market_regime 
                    })
                    potential_trades_today.append(prediction_doc)

                else: # This handles 'HOLD' signals from the AI
                    prediction_doc = final_analysis.copy()
                    prediction_doc.update({
                        'ticker': ticker, 
                        'prediction_date': current_day.to_pydatetime(),
                        'price_at_prediction': latest_row['close'], 
                        'status': 'hold', # Status is now just 'hold'
                        'strategy': 'ApexPredator_Multi', 
                        'veto_reason': None # Veto reason is no longer applicable here
                    })
                    database_manager.save_prediction_for_backtesting(prediction_doc, batch_id)
            # --- END OF LOOP FOR EACH TICKER ---

            # --- PHASE 2 & 3: RANK AND SELECT THE BEST TRADES ---
            if potential_trades_today:
                log.info(f"Found {len(potential_trades_today)} potential trades for today. Ranking by conviction score...")
                
                sorted_trades = sorted(potential_trades_today, key=lambda x: x.get('scrybeScore', 0), reverse=True)
                
                trades_to_execute = sorted_trades[:PORTFOLIO_CONSTRAINTS['max_concurrent_trades']]
                total_trades_executed += len(trades_to_execute)
                
                log.info(f"Selecting top {len(trades_to_execute)} trades to execute and saving to database.")
                for trade_doc in trades_to_execute:
                    trade_doc['analysis_id'] = str(uuid.uuid4()) # Assign final ID before saving
                    database_manager.save_prediction_for_backtesting(trade_doc, batch_id)
            
    log.info("\n--- ✅ APEX Dynamic Simulation Finished! ---")
    log.info("\n" + "="*50)
    log.info("### FUNNEL ANALYSIS REPORT ###")
    log.info(f"Total Stocks Processed by Screener: {total_stocks_processed_by_screener}")
    log.info(f"Total Stocks that Passed Screener: {total_stocks_passed_screener} ({ (total_stocks_passed_screener/total_stocks_processed_by_screener*100) if total_stocks_processed_by_screener > 0 else 0 :.2f}%)")
    log.info(f"Total Raw BUY/SELL Signals from AI: {total_ai_buy_sell_signals}")
    log.info(f"Total Trades Executed & Saved: {total_trades_executed}")
    log.info("="*50)
    database_manager.close_db_connection()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the APEX Dynamic Funnel historical backtest.")
    parser.add_argument('--batch_id', required=True)
    parser.add_argument('--start_date', required=True)
    parser.add_argument('--end_date', required=True)
    parser.add_argument('--stock_universe', required=True, help="Comma-separated list defining the total stock UNIVERSE to screen from")
    parser.add_argument('--fresh_run', type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()
    
    stocks_list = [stock.strip() for stock in args.stock_universe.split(',')]
    run_simulation(batch_id=args.batch_id, start_date=args.start_date, end_date=args.end_date, stock_universe=stocks_list, is_fresh_run=args.fresh_run)