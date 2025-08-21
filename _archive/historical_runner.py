# historical_runner.py (APEX VERSION - FINAL, CORRECTED)
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

STATE_FILE = 'simulation_state.json'


def _synthesize_dvm_from_apex(apex_analysis: dict, fundamental_proxies: dict) -> dict:
    """
    Derives a simple DVM score from the rich Apex output for frontend display.
    """
    # M-Score is directly from the Apex AI's main conviction score
    momentum_score = apex_analysis.get('scrybeScore', 0)
    
    # D-Score is from our 'quality_score' proxy, which is based on volatility
    durability_score = fundamental_proxies.get('quality_score', 50)
    
    # V-Score is derived from our 'valuation_proxy' (position in 52-week range)
    # We invert it because a low position (e.g., 10%) implies better value (score of 90).
    valuation_proxy_str = fundamental_proxies.get('valuation_proxy', '50.0%')
    try:
        valuation_pct = float(valuation_proxy_str.split('%')[0])
        valuation_score = 100 - valuation_pct
    except:
        valuation_score = 50 # Default to a neutral score on any error

    return {
        "d_score": int(durability_score),
        "v_score": int(valuation_score),
        "m_score": int(momentum_score)
    }

## START: Helper Functions for Omni-Context ##
def _get_30_day_performance_review(current_day: pd.Timestamp, batch_id: str) -> str:
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
    high_conv_ids = database_manager.predictions_collection.distinct("_id", {"batch_id": batch_id, "scrybeScore": {"$gt": 80}})
    high_conviction_trades = df[df['prediction_id'].isin(high_conv_ids)]
    hc_win_rate = (high_conviction_trades['net_return_pct'] > 0).mean() * 100 if not high_conviction_trades.empty else 0
    hc_avg_gain = high_conviction_trades[high_conviction_trades['net_return_pct'] > 0]['net_return_pct'].mean() if not high_conviction_trades[high_conviction_trades['net_return_pct'] > 0].empty else 0
    recent_losers = df[df['net_return_pct'] < 0].sort_values(by="close_date", ascending=False).head(2)
    loser_strings = [f"[{row['signal']}] {row['ticker']} ({row['net_return_pct']:.2f}%)" for _, row in recent_losers.iterrows()]
    review = (f"30-Day Performance Review:\n- Total Signals: {total_signals}\n- Win Rate: {win_rate:.1f}%\n- Profit Factor: {profit_factor:.2f}\n"
              f"- High-Conviction (Score > 80): {len(high_conviction_trades)} trades, {hc_win_rate:.1f}% win rate, Avg Gain: +{hc_avg_gain:.2f}%\n"
              f"- Recent Losing Trades: {', '.join(loser_strings) if loser_strings else 'None'}")
    return review

def _get_1_day_tactical_lookback(current_day: pd.Timestamp, ticker: str, batch_id: str) -> str:
    previous_trading_day = current_day - pd.Timedelta(days=1)
    query = {"batch_id": batch_id, "ticker": ticker, "prediction_date": {"$gte": previous_trading_day.to_pydatetime(), "$lt": current_day.to_pydatetime()}}
    last_analysis = database_manager.predictions_collection.find_one(query)
    if not last_analysis: return "No analysis for this stock on the previous trading day."
    lookback = (f"Previous Day's Note ({ticker}):\n- Signal: {last_analysis.get('signal', 'N/A')}\n- Scrybe Score: {last_analysis.get('scrybeScore', 0)}\n"
                f"- Key Insight: \"{last_analysis.get('keyInsight', 'N/A')}\"")
    return lookback
## END: Helper Functions for Omni-Context ##

def save_state(next_day_to_run):
    with open(STATE_FILE, 'w') as f: json.dump({'next_start_date': next_day_to_run.strftime('%Y-%m-%d')}, f)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            log.warning(f"Resuming from saved state. Next day to run: {state['next_start_date']}")
            return state['next_start_date']
    return None

def run_historical_test(batch_id: str, start_date: str, end_date: str, stocks_to_test: list, is_fresh_run: bool = False, generate_charts: bool = False):
    log.info(f"### STARTING HISTORICAL RUN FOR BATCH: {batch_id} ###")
    log.info(f"Period: {start_date} to {end_date}")
    database_manager.init_db(purpose='scheduler')
    if is_fresh_run:
        log.warning("FRESH RUN ENABLED: Deleting all previous data.")
        database_manager.clear_scheduler_data()
        if os.path.exists(STATE_FILE): os.remove(STATE_FILE)
    try:
        key_manager = APIKeyManager(api_keys=config.GEMINI_API_KEY_POOL)
        analyzer = AIAnalyzer(api_key=key_manager.get_key())
        log.info("Pre-loading all historical data...")
        
        full_historical_data_cache = {ticker: data for ticker in stocks_to_test if (data := data_retriever.get_historical_stock_data(ticker, end_date=end_date)) is not None and len(data) > 252}
        log.info("✅ All data pre-loading complete.")
    except Exception as e:
        log.fatal(f"Failed during pre-run initialization. Error: {e}")
        return
    resumed_start_date = load_state()
    if resumed_start_date: start_date = resumed_start_date
    simulation_days = pd.bdate_range(start=start_date, end=end_date)
    if not simulation_days.empty:
        log.info(f"\n--- Starting simulation for {len(simulation_days)} trading days ---")
        for i, current_day in enumerate(simulation_days):
            day_str = current_day.strftime('%Y-%m-%d')
            log.info(f"\n--- Simulating Day {i+1}/{len(simulation_days)}: {day_str} ---")
            
            for ticker in stocks_to_test:
                if ticker not in full_historical_data_cache: continue
                try:
                    data_slice = full_historical_data_cache[ticker].loc[:day_str].copy()
                    if len(data_slice) < 252: continue
                    
                    log.info(f"--- Analyzing Ticker: {ticker} for {day_str} ---")
                    
                    data_slice.ta.bbands(length=20, append=True); data_slice.ta.rsi(length=14, append=True)
                    data_slice.ta.adx(length=14, append=True); data_slice.ta.atr(length=14, append=True)

                    latest_row = data_slice.iloc[-1]
                    required_cols = ['BBU_20_2.0', 'BBL_20_2.0', 'RSI_14', 'ADX_14', 'ATRr_14']
                    if latest_row[required_cols].isnull().any():
                        log.warning(f"Skipping {ticker} on {day_str} due to NaN indicators.")
                        continue
                    
                    technical_indicators_for_ai = {
                        "ADX_14": round(latest_row['ADX_14'], 2),
                        "RSI_14": round(latest_row['RSI_14'], 2),
                        "close_price": round(latest_row['close'], 2),
                        "Lower_Bollinger_Band": round(latest_row['BBL_20_2.0'], 2),
                        "Upper_Bollinger_Band": round(latest_row['BBU_20_2.0'], 2)
                    }
                    
                    # --- START: CORRECTED KEY ROTATION LOGIC BLOCK ---
                    final_analysis = None
                    for attempt in range(len(key_manager.api_keys)):
                        try:
                            # IMPORTANT: Calling the new mean reversion function
                            final_analysis = analyzer.get_mean_reversion_signal(ticker, technical_indicators_for_ai)
                            if final_analysis:
                                break 
                        except Exception as e:
                            if "429" in str(e):
                                log.warning(f"Quota error on attempt {attempt + 1}. Rotating key...")
                                if attempt < len(key_manager.api_keys) - 1:
                                    analyzer = AIAnalyzer(api_key=key_manager.rotate_key())
                                else:
                                    log.error("All API keys exhausted. Unable to get analysis for this day.")
                                    break
                            else:
                                log.error(f"A non-quota error occurred during AI analysis: {e}")
                                break 
                    # --- END: CORRECTED KEY ROTATION LOGIC BLOCK ---

                    if not final_analysis:
                        log.warning(f"AI analysis failed for {ticker} on {day_str} after all retries, skipping.")
                        continue

                    final_signal = final_analysis.get('signal')
                    prediction_doc = final_analysis.copy()

                    if final_signal in ['BUY', 'SELL']:
                        entry_price = latest_row['close']
                        atr = latest_row['ATRr_14']
                        risk_per_share = 2 * atr
                        stop_loss_price = entry_price - risk_per_share if final_signal == 'BUY' else entry_price + risk_per_share
                        reward_per_share = risk_per_share * 1.5
                        target_price = entry_price + reward_per_share if final_signal == 'BUY' else entry_price - reward_per_share
                        
                        prediction_doc['tradePlan'] = {
                            "entryPrice": round(entry_price, 2), 
                            "target": round(target_price, 2), 
                            "stopLoss": round(stop_loss_price, 2)
                        }
                    
                    prediction_doc.update({
                        'analysis_id': str(uuid.uuid4()), 'ticker': ticker, 'prediction_date': current_day.to_pydatetime(),
                        'price_at_prediction': latest_row['close'], 'status': 'open', 'strategy': "MeanReversion_v1",
                        'atr_at_prediction': latest_row['ATRr_14'],
                        'position_size_pct': 100.0
                    })
                    database_manager.save_prediction_for_backtesting(prediction_doc, batch_id)
                    time.sleep(5) # Keeping the sleep to be respectful to the API
                except Exception as e:
                    log.error(f"CRITICAL FAILURE on day {day_str} for {ticker}: {e}", exc_info=True)
            if i + 1 < len(simulation_days): save_state(simulation_days[i+1])
    if os.path.exists(STATE_FILE): os.remove(STATE_FILE)
    log.info("\n--- ✅ Historical Run Finished! ---")
    database_manager.close_db_connection()

def _get_per_stock_trade_history(ticker: str, batch_id: str, current_day: pd.Timestamp) -> str:
    """Gets the results of the last 3 closed trades for a specific stock."""
    query = { "batch_id": batch_id, "ticker": ticker, "close_date": {"$lt": current_day.to_pydatetime()} }
    # Sort by close_date descending and limit to the last 3 trades
    recent_trades = list(database_manager.performance_collection.find(query).sort("close_date", -1).limit(3))
    if not recent_trades:
        return "No recent trade history for this stock."
    
    history_lines = []
    for i, trade in enumerate(recent_trades):
        line = (f"{i+1}. Signal: {trade.get('signal')}, "
                f"Outcome: {trade.get('net_return_pct'):.2f}% "
                f"({trade.get('closing_reason')})")
        history_lines.append(line)
    return "\n".join(history_lines)

def _get_dynamic_risk_mode(ticker: str, batch_id: str, current_day: pd.Timestamp, cooldown_tracker: dict) -> tuple[str, float]:
    """
    Determines the risk mode (Green, Yellow, Red) and position size for a stock
    based on its recent performance and any active cool-down periods.
    """
    cfg = config.DYNAMIC_RISK_CONFIG
    
    # Check if the stock is in a cool-down period
    if ticker in cooldown_tracker and current_day <= cooldown_tracker[ticker]:
        log.warning(f"CIRCUIT BREAKER: {ticker} is in a cool-down period. No trades allowed.")
        return "Red", cfg['red_mode_position_size_pct']

    # Fetch recent trade history
    query = {"batch_id": batch_id, "ticker": ticker, "close_date": {"$lt": current_day.to_pydatetime()}}
    recent_trades = list(database_manager.performance_collection.find(query).sort("close_date", -1).limit(cfg['lookback_period']))
    
    if len(recent_trades) < cfg['lookback_period']:
        return "Green", cfg['green_mode_position_size_pct'] # Not enough history, trade normally

    # Check for consecutive losses
    consecutive_losses = 0
    for trade in recent_trades:
        if trade.get('net_return_pct', 0) < 0:
            consecutive_losses += 1
        else:
            break # Streak is broken
    
    if consecutive_losses >= cfg['red_mode_threshold']:
        cooldown_end_date = current_day + pd.Timedelta(days=cfg['cooldown_period_days'])
        cooldown_tracker[ticker] = cooldown_end_date
        log.warning(f"CIRCUIT BREAKER ENGAGED: {ticker} has {consecutive_losses} consecutive losses. Entering cool-down until {cooldown_end_date.strftime('%Y-%m-%d')}.")
        return "Red", cfg['red_mode_position_size_pct']

    # Check for total losses
    total_losses = sum(1 for trade in recent_trades if trade.get('net_return_pct', 0) < 0)
    if total_losses >= cfg['yellow_mode_threshold']:
        log.warning(f"RISK MODE YELLOW: {ticker} has {total_losses} losses in its last {len(recent_trades)} trades. Reducing position size.")
        return "Yellow", cfg['yellow_mode_position_size_pct']

    return "Green", cfg['green_mode_position_size_pct']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Apex historical backtest.")
    parser.add_argument('--batch_id', required=True)
    parser.add_argument('--start_date', required=True)
    parser.add_argument('--end_date', required=True)
    parser.add_argument('--stocks', required=True)
    parser.add_argument('--fresh_run', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--generate_charts', type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()
    stocks_list = [stock.strip() for stock in args.stocks.split(',')]
    run_historical_test(batch_id=args.batch_id, start_date=args.start_date, end_date=args.end_date, stocks_to_test=stocks_list, is_fresh_run=args.fresh_run, generate_charts=args.generate_charts)