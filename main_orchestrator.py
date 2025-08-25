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

STATE_FILE = 'simulation_state.json'

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

def save_state(next_day_to_run):
    with open(STATE_FILE, 'w') as f: json.dump({'next_start_date': next_day_to_run.strftime('%Y-%m-%d')}, f)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            log.warning(f"Resuming from saved state. Next day to run: {state['next_start_date']}")
            return state['next_start_date']
    return None

def run_simulation(batch_id: str, start_date: str, end_date: str, stock_universe: list, is_fresh_run: bool = False):
    log.info(f"### STARTING APEX SIMULATION FOR BATCH: {batch_id} ###")
    log.info(f"Period: {start_date} to {end_date}")
    database_manager.init_db(purpose='scheduler')
    if is_fresh_run:
        log.warning("FRESH RUN ENABLED: Deleting all previous data.")
        database_manager.clear_scheduler_data()
    
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
        fundamentally_approved_stocks = []
        for ticker in stock_universe:
            log.info(f"  -> Checking fundamentals for {ticker}...")
            # Note: We are now calling the function we imported at the top of the file
            if _passes_fundamental_health_check(ticker):
                fundamentally_approved_stocks.append(ticker)
                log.info(f"    - ✅ PASS: {ticker} is fundamentally approved.")

        log.info(f"✅ Fundamental Pre-Flight Check complete. {len(fundamentally_approved_stocks)}/{len(stock_universe)} stocks passed.")
        if not fundamentally_approved_stocks:
            log.fatal("No stocks passed the fundamental pre-flight check. Halting simulation.")
            return
        # --- END: NEW FUNDAMENTAL PRE-FLIGHT CHECK ---

    except Exception as e:
        log.fatal(f"Failed during pre-run initialization. Error: {e}")
        return

    simulation_days = pd.bdate_range(start=start_date, end=end_date)
    portfolio_config = config.BACKTEST_PORTFOLIO_CONFIG
    current_equity = portfolio_config['initial_capital']
    active_strategy = config.APEX_SWING_STRATEGY

    if not simulation_days.empty:
        log.info(f"\n--- Starting simulation for {len(simulation_days)} trading days ---")
        for i, current_day in enumerate(simulation_days):
            day_str = current_day.strftime('%Y-%m-%d')
            log.info(f"\n--- Simulating Day {i+1}/{len(simulation_days)}: {day_str} ---")
            
            try:
                latest_vix = vix_data.loc[:day_str].iloc[-1]['close']
                market_is_high_risk = latest_vix > config.HIGH_RISK_VIX_THRESHOLD
                if market_is_high_risk:
                    log.warning(f"!! MASTER RISK OVERLAY ENGAGED !! VIX is at {latest_vix:.2f}. No new BUY signals will be processed today.")
            except (KeyError, IndexError):
                market_is_high_risk = False
            
            market_regime = data_retriever.calculate_regime_from_data(nifty_data.loc[:day_str])
            strong_sectors = sector_analyzer.get_top_performing_sectors()
            approved_stock_data_cache = {ticker: full_historical_data_cache[ticker] for ticker in fundamentally_approved_stocks if ticker in full_historical_data_cache}
            stocks_for_today = generate_dynamic_watchlist(strong_sectors, approved_stock_data_cache, current_day)
            
            if not stocks_for_today:
                log.warning("The dynamic funnel returned no stocks for today. Proceeding to next day.")
                continue

            # --- PHASE 1: COLLECT ALL POTENTIAL TRADES FOR THE DAY ---
            potential_trades_today = []
            for ticker in stocks_for_today:
                # --- START: RETRY LOGIC ---
                max_retries = len(config.GEMINI_API_KEY_POOL)
                retries = 0
                final_analysis = None
                
                while retries < max_retries:
                    try:
                        log.info(f"--- Analyzing Ticker: {ticker} (Attempt {retries + 1}/{max_retries}) ---")
                        point_in_time_data = full_historical_data_cache.get(ticker).loc[:day_str].copy()
                        if len(point_in_time_data) < 252:
                            log.warning(f"Skipping {ticker} due to insufficient historical data on {day_str}.")
                            final_analysis = None
                            break 
                        
                        strategic_review = _get_30_day_performance_review(current_day, batch_id)
                        tactical_lookback = _get_1_day_tactical_lookback(current_day, ticker, batch_id)
                        per_stock_history = _get_per_stock_trade_history(ticker, batch_id, current_day)
                        latest_row = point_in_time_data.iloc[-1]
                        point_in_time_data.ta.atr(length=14, append=True)
                        atr_at_prediction = point_in_time_data['ATRr_14'].iloc[-1]
                        nifty_5d_change = (nifty_data.loc[:day_str]['close'].iloc[-1] / nifty_data.loc[:day_str]['close'].iloc[-6] - 1) * 100
                        stock_5d_change = (latest_row['close'] / point_in_time_data['close'].iloc[-6] - 1) * 100
                        full_context = {
                            "layer_1_macro_context": {"nifty_50_regime": market_regime},
                            "layer_2_relative_strength": {"relative_strength_vs_nifty50": "Outperforming" if stock_5d_change > nifty_5d_change else "Underperforming"},
                            "layer_3_fundamental_moat": data_retriever.get_fundamental_proxies(point_in_time_data),
                            "layer_4_technicals": {"daily_close": latest_row['close']}, "layer_5_options_sentiment": {"sentiment": "Unavailable in backtest"},
                            "layer_6_news_catalyst": {"summary": "Unavailable in backtest"}
                        }
                        sanitized_full_context = sanitize_context(full_context)
                        final_analysis = analyzer.get_apex_analysis(ticker, sanitized_full_context, strategic_review, tactical_lookback, per_stock_history)
                        if final_analysis:
                            log.info(f"Successfully got analysis for {ticker}.")
                            break
                    except Exception as e:
                        log.error(f"Attempt {retries + 1} for {ticker} failed. Error: {e}")
                        
                        # If it's a blocking error, log the context for debugging
                        if "blocked response" in str(e).lower():
                            log.error(f"--- PROMPT CONTEXT THAT CAUSED BLOCK FOR {ticker} ---")
                            log.error(json.dumps(full_context, indent=2))
                        
                        retries += 1 # Increment retry counter on ANY failure
                        if retries < max_retries:
                            log.warning(f"Rotating API key and retrying ({retries}/{max_retries})...")
                            analyzer = AIAnalyzer(api_key=key_manager.rotate_key())
                            log.info("Pausing for 35 seconds before next attempt...")
                            time.sleep(35)
                        else:
                            log.error(f"All {max_retries} retries failed for {ticker}. Marking as final failure.")
                            final_analysis = None
                            # No 'break' here, let the while loop condition handle the exit
                # --- END: RETRY LOGIC ---

                if not final_analysis:
                    log.warning(f"Skipping {ticker} for day {day_str} after all retries failed or a critical error occurred.")
                    continue
                
                original_signal = final_analysis.get('signal')
                scrybe_score = final_analysis.get('scrybeScore', 0)
                final_signal = original_signal
                veto_reason = None
                
                if original_signal in ['BUY', 'SELL']:
                    is_conviction_ok = abs(scrybe_score) >= active_strategy['min_conviction_score']
                    is_regime_ok = (original_signal == 'BUY' and market_regime != 'Bearish') or (original_signal == 'SELL' and market_regime != 'Bullish')
                    entry_price = latest_row['close']
                    potential_risk_per_share = active_strategy['stop_loss_atr_multiplier'] * atr_at_prediction
                    potential_reward_per_share = potential_risk_per_share * active_strategy['profit_target_rr_multiple']
                    risk_reward_ratio = potential_reward_per_share / potential_risk_per_share if potential_risk_per_share > 0 else 0
                    is_rr_ok = risk_reward_ratio >= 1.5
                    if original_signal == 'BUY' and market_is_high_risk: final_signal, veto_reason = 'HOLD', f"VETOED BUY: High market VIX ({latest_vix:.2f})"
                    elif not is_regime_ok: final_signal, veto_reason = 'HOLD', f"VETOED {original_signal}: Signal contradicts Market Regime ({market_regime})"
                    elif not is_conviction_ok: final_signal, veto_reason = 'HOLD', f"VETOED {original_signal}: Conviction Score ({scrybe_score}) is below threshold of {active_strategy['min_conviction_score']}"
                    elif not is_rr_ok: final_signal, veto_reason = 'HOLD', f"VETOED {original_signal}: Poor Risk/Reward Ratio ({risk_reward_ratio:.2f}R)"

                prediction_doc = final_analysis.copy()
                prediction_doc['signal'] = final_signal

                if final_signal in ['BUY', 'SELL']:
                    num_shares_to_trade = int((current_equity * (portfolio_config['risk_per_trade_pct'] / 100.0)) / potential_risk_per_share) if potential_risk_per_share > 0 else 0
                    position_size_pct = (num_shares_to_trade * entry_price / current_equity) * 100
                    stop_loss_price = (entry_price - potential_risk_per_share) if final_signal == 'BUY' else (entry_price + potential_risk_per_share)
                    target_price = (entry_price + potential_reward_per_share) if final_signal == 'BUY' else (entry_price - potential_reward_per_share)
                    
                    prediction_doc['tradePlan'] = {"entryPrice": round(entry_price, 2), "target": round(target_price, 2), "stopLoss": round(stop_loss_price, 2)}
                    prediction_doc.update({'ticker': ticker, 'prediction_date': current_day.to_pydatetime(), 'price_at_prediction': entry_price, 'status': 'open', 'strategy': "ApexSwing_v5_HighConviction", 'atr_at_prediction': atr_at_prediction, 'position_size_pct': position_size_pct})
                    potential_trades_today.append(prediction_doc) # ADD TO LIST, DO NOT SAVE
                else:
                    prediction_doc.update({'ticker': ticker, 'prediction_date': current_day.to_pydatetime(),'price_at_prediction': latest_row['close'], 'status': 'vetoed' if veto_reason else 'hold', 'strategy': "ApexSwing_v5_HighConviction", 'atr_at_prediction': atr_at_prediction, 'veto_reason': veto_reason})
                    database_manager.save_prediction_for_backtesting(prediction_doc, batch_id) # SAVE NON-TRADES IMMEDIATELY
                
                log.info("Pausing for 35 seconds to respect API rate limits...")
                time.sleep(35)
            # --- END OF LOOP FOR EACH TICKER ---

            # --- PHASE 2 & 3: RANK AND SELECT THE BEST TRADES ---
            if potential_trades_today:
                log.info(f"Found {len(potential_trades_today)} potential trades for today. Ranking by conviction score...")
                
                sorted_trades = sorted(potential_trades_today, key=lambda x: x.get('scrybeScore', 0), reverse=True)
                
                trades_to_execute = sorted_trades[:PORTFOLIO_CONSTRAINTS['max_concurrent_trades']]
                
                log.info(f"Selecting top {len(trades_to_execute)} trades to execute and saving to database.")
                for trade_doc in trades_to_execute:
                    trade_doc['analysis_id'] = str(uuid.uuid4()) # Assign final ID before saving
                    database_manager.save_prediction_for_backtesting(trade_doc, batch_id)
            
    log.info("\n--- ✅ APEX Dynamic Simulation Finished! ---")
    database_manager.close_db_connection()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the APEX Dynamic Funnel historical backtest.")
    parser.add_argument('--batch_id', required=True)
    parser.add_argument('--start_date', required=True)
    parser.add_argument('--resume_date', required=False, default=None)
    parser.add_argument('--end_date', required=True)
    parser.add_argument('--stock_universe', required=True, help="Comma-separated list defining the total stock UNIVERSE to screen from")
    parser.add_argument('--fresh_run', type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()
    
    effective_start_date = args.start_date
    if args.resume_date and not args.fresh_run:
        log.warning(f"RESUME DATE PROVIDED. Overriding start date. The simulation will resume from: {args.resume_date}")
        effective_start_date = args.resume_date
    
    stocks_list = [stock.strip() for stock in args.stock_universe.split(',')]
    run_simulation(batch_id=args.batch_id, start_date=effective_start_date, end_date=args.end_date, stock_universe=stocks_list, is_fresh_run=args.fresh_run)