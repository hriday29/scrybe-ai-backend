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
import quantitative_screener

STATE_FILE = 'simulation_state.json'

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
        if os.path.exists(STATE_FILE): os.remove(STATE_FILE)
    
    try:
        key_manager = APIKeyManager(api_keys=config.GEMINI_API_KEY_POOL)
        analyzer = AIAnalyzer(api_key=key_manager.get_key())
        log.info("Pre-loading all historical data for the stock universe...")
        full_historical_data_cache = {ticker: data for ticker in stock_universe if (data := data_retriever.get_historical_stock_data(ticker, end_date=end_date)) is not None and len(data) > 252}
        nifty_data = data_retriever.get_historical_stock_data("^NSEI", end_date=end_date)
        vix_data = data_retriever.get_historical_stock_data("^INDIAVIX", end_date=end_date)
        log.info("✅ All data pre-loading complete.")
    except Exception as e:
        log.fatal(f"Failed during pre-run initialization. Error: {e}")
        return

    resumed_start_date = load_state()
    if resumed_start_date: start_date = resumed_start_date
    simulation_days = pd.bdate_range(start=start_date, end=end_date)
    
    portfolio_config = config.BACKTEST_PORTFOLIO_CONFIG
    current_equity = portfolio_config['initial_capital']

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
                log.warning("Could not retrieve VIX for today. Proceeding with caution.")
                market_is_high_risk = False
            
            market_regime = data_retriever.calculate_regime_from_data(nifty_data.loc[:day_str])
            strong_sectors = sector_analyzer.get_top_performing_sectors()
            stocks_for_today = quantitative_screener.generate_dynamic_watchlist(strong_sectors)
            
            if not stocks_for_today:
                log.warning("The dynamic funnel returned no stocks for today. Proceeding to next day.")
                if i + 1 < len(simulation_days): save_state(simulation_days[i+1])
                continue

            for ticker in stocks_for_today:
                try:
                    point_in_time_data = full_historical_data_cache.get(ticker).loc[:day_str].copy()
                    if len(point_in_time_data) < 252: continue
                    
                    log.info(f"--- Analyzing Ticker: {ticker} from Dynamic Watchlist ---")
                    
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
                    
                    final_analysis = analyzer.get_apex_analysis(ticker, full_context, strategic_review, tactical_lookback, per_stock_history)
                    
                    if not final_analysis:
                        log.warning(f"Apex AI analysis failed for {ticker}, skipping.")
                        continue

                    active_strategy = config.APEX_SWING_STRATEGY
                    original_signal = final_analysis.get('signal')
                    scrybe_score = final_analysis.get('scrybeScore', 0)
                    final_signal = original_signal
                    veto_reason = None
                    
                    if original_signal in ['BUY', 'SELL']:
                        is_conviction_ok = abs(scrybe_score) >= active_strategy['min_conviction_score']
                        is_regime_ok = (original_signal == 'BUY' and market_regime != 'Bearish') or (original_signal == 'SELL' and market_regime != 'Bullish')

                        if original_signal == 'BUY' and market_is_high_risk:
                            final_signal, veto_reason = 'HOLD', f"VETOED BUY: High market VIX ({latest_vix:.2f})"
                        elif not is_regime_ok:
                            final_signal, veto_reason = 'HOLD', f"VETOED {original_signal}: Signal contradicts Market Regime ({market_regime})"
                        elif not is_conviction_ok:
                            final_signal, veto_reason = 'HOLD', f"VETOED {original_signal}: Conviction Score ({scrybe_score}) is below threshold"
                    
                    prediction_doc = final_analysis
                    prediction_doc['signal'] = final_signal

                    if final_signal in ['BUY', 'SELL']:
                        entry_price = latest_row['close']
                        potential_risk_per_share = active_strategy['stop_loss_atr_multiplier'] * atr_at_prediction
                        potential_reward_per_share = potential_risk_per_share * active_strategy['profit_target_rr_multiple']
                        
                        rupee_risk_per_trade = current_equity * (portfolio_config['risk_per_trade_pct'] / 100.0)
                        num_shares_to_trade = int(rupee_risk_per_trade / potential_risk_per_share) if potential_risk_per_share > 0 else 0
                        position_size_rupees = num_shares_to_trade * entry_price
                        position_size_pct = (position_size_rupees / current_equity) * 100
                        
                        log.info(f"RISK CALC: Portfolio Risk: ₹{rupee_risk_per_trade:.2f}. Per-Share Risk: ₹{potential_risk_per_share:.2f}. ==> Position Size: {num_shares_to_trade} shares ({position_size_pct:.2f}%).")
                        
                        stop_loss_price = (entry_price - potential_risk_per_share) if final_signal == 'BUY' else (entry_price + potential_risk_per_share)
                        target_price = (entry_price + potential_reward_per_share) if final_signal == 'BUY' else (entry_price - potential_reward_per_share)
                        
                        prediction_doc['tradePlan'] = {"entryPrice": round(entry_price, 2), "target": round(target_price, 2), "stopLoss": round(stop_loss_price, 2)}
                        prediction_doc.update({
                            'analysis_id': str(uuid.uuid4()), 'ticker': ticker, 'prediction_date': current_day.to_pydatetime(),
                            'price_at_prediction': entry_price, 'status': 'open', 'strategy': "ApexSwing_v4_Full",
                            'atr_at_prediction': atr_at_prediction, 'position_size_pct': position_size_pct,
                        })
                        database_manager.save_prediction_for_backtesting(prediction_doc, batch_id)
                        time.sleep(20)
                    else:
                        prediction_doc['tradePlan'] = None
                        prediction_doc.update({
                            'analysis_id': str(uuid.uuid4()), 'ticker': ticker, 'prediction_date': current_day.to_pydatetime(),
                            'price_at_prediction': latest_row['close'], 'status': 'vetoed' if veto_reason else 'hold', 
                            'strategy': "ApexSwing_v4_Full", 'atr_at_prediction': atr_at_prediction, 'veto_reason': veto_reason
                        })
                        database_manager.save_prediction_for_backtesting(prediction_doc, batch_id)
                        time.sleep(5)

                except Exception as e:
                    log.error(f"CRITICAL FAILURE on day {day_str} for {ticker}: {e}", exc_info=True)
                    if "429" in str(e):
                        analyzer = AIAnalyzer(api_key=key_manager.rotate_key())
                    continue

                log.info("Pausing for 33 seconds to respect API rate limit...")
                time.sleep(33)

            if i + 1 < len(simulation_days): save_state(simulation_days[i+1])
    
    if os.path.exists(STATE_FILE): os.remove(STATE_FILE)
    log.info("\n--- ✅ APEX Dynamic Simulation Finished! ---")
    database_manager.close_db_connection()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the APEX Dynamic Funnel historical backtest.")
    parser.add_argument('--batch_id', required=True)
    parser.add_argument('--start_date', required=True)
    parser.add_argument('--resume_date', required=False, default=None)
    parser.add_argument('--end_date', required=True)
    parser.add_argument('--stock_universe', required=True, help="Comma-separated list defining the total stock UNIVERSE to screen from (e.g., all Nifty 50 stocks)")
    parser.add_argument('--fresh_run', type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()
    effective_start_date = args.start_date
    if args.resume_date:
        log.warning(f"RESUME DATE PROVIDED. Overriding start date. The simulation will resume from: {args.resume_date}")
        effective_start_date = args.resume_date
    stocks_list = [stock.strip() for stock in args.stock_universe.split(',')]
    run_simulation(batch_id=args.batch_id, start_date=args.start_date, end_date=args.end_date, stock_universe=stocks_list, is_fresh_run=args.fresh_run)