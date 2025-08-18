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
    log.info(f"### STARTING APEX HISTORICAL RUN FOR BATCH: {batch_id} ###")
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
        
        # --- START: DEFINITIVE FIX ---
        # Load stock data (already returns lowercase from our fixed retriever)
        full_historical_data_cache = {ticker: data for ticker in stocks_to_test if (data := data_retriever.get_historical_stock_data(ticker, end_date=end_date)) is not None and len(data) > 252}
        
        # Load Nifty data
        nifty_data_cache_raw = data_retriever.get_historical_stock_data("^NSEI", end_date=end_date)
        nifty_data_cache = nifty_data_cache_raw.rename(columns={
            'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
        })
        # Load India VIX data
        vix_data_cache = data_retriever.get_historical_stock_data("^INDIAVIX", end_date=end_date)
        
        benchmarks_data_cache = data_retriever.get_benchmarks_data(end_date=end_date)
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
            vix_slice = vix_data_cache.loc[:day_str] if vix_data_cache is not None else None
            volatility_regime = data_retriever.get_volatility_regime(vix_slice)

            if volatility_regime == "High-Risk":
                log.warning(f"VOLATILITY FILTER ENGAGED: Market is in 'High-Risk' state. Standing aside for day {day_str}.")
                if i + 1 < len(simulation_days): save_state(simulation_days[i+1])
                continue # Skip all trading for this day
            log.info(f"\n--- Simulating Day {i+1}/{len(simulation_days)}: {day_str} ---")
            nifty_slice = nifty_data_cache.loc[:day_str] if nifty_data_cache is not None else None
            current_regime = data_retriever.calculate_regime_from_data(nifty_slice)
            
            for ticker in stocks_to_test:
                if ticker not in full_historical_data_cache: continue
                try:
                    data_slice = full_historical_data_cache[ticker].loc[:day_str].copy()
                    if len(data_slice) < 252: continue
                    log.info(f"--- Analyzing Ticker: {ticker} for {day_str} ---")
                    
                    # Safe Indicator Calculation
                    data_slice.ta.bbands(length=20, append=True); data_slice.ta.rsi(length=14, append=True)
                    data_slice.ta.macd(fast=12, slow=26, signal=9, append=True); data_slice.ta.adx(length=14, append=True)
                    data_slice.ta.atr(length=14, append=True)
                    latest_row = data_slice.iloc[-1]
                    required_cols = ['BBU_20_2.0', 'RSI_14', 'MACD_12_26_9', 'ADX_14', 'ATRr_14']
                    if latest_row[required_cols].isnull().any():
                        log.warning(f"Skipping {ticker} on {day_str} due to NaN indicators.")
                        continue
                    
                    # Assemble Full Context for AI
                    benchmarks_slice = benchmarks_data_cache.loc[:day_str] if benchmarks_data_cache is not None else pd.DataFrame()
                    nifty_5d_change = (nifty_slice['Close'].iloc[-1] / nifty_slice['Close'].iloc[-6] - 1) * 100 if nifty_slice is not None and len(nifty_slice) > 5 else 0
                    stock_5d_change = (latest_row['Close'] / data_slice['Close'].iloc[-6] - 1) * 100 if len(data_slice) > 5 else 0
                    relative_strength = "Outperforming" if stock_5d_change > nifty_5d_change else "Underperforming"
                    weekly_data = data_slice['Close'].resample('W').last()
                    weekly_trend = "Bullish" if len(weekly_data) > 2 and weekly_data.iloc[-1] > weekly_data.iloc[-2] else "Bearish"
                    fundamental_proxies = data_retriever.get_fundamental_proxies(data_slice)

                    full_context_for_ai = {
                        "layer_1_macro_context": {"nifty_50_regime": current_regime},
                        "layer_2_relative_strength": {"stock_5d_change_pct": f"{stock_5d_change:.2f}%", "relative_strength_vs_nifty50": relative_strength},
                        "layer_3_fundamental_moat": fundamental_proxies,
                        "layer_4_technicals": {"daily_indicators": {"ADX": f"{latest_row['ADX_14']:.2f}", "RSI": f"{latest_row['RSI_14']:.2f}"}, "weekly_trend": weekly_trend},
                        "layer_5_options_sentiment": {"sentiment": "N/A in backtest"},
                        "layer_6_news_catalyst": {"summary": "N/A in backtest"}
                    }
                    
                    strategic_review_30d = _get_30_day_performance_review(current_day, batch_id)
                    tactical_lookback_1d = _get_1_day_tactical_lookback(current_day, ticker, batch_id)
                    per_stock_history = _get_per_stock_trade_history(ticker, batch_id, current_day)
                    
                    final_analysis = None
                    for attempt in range(len(key_manager.api_keys)):
                        try:
                            final_analysis = analyzer.get_apex_analysis(
                                ticker, full_context_for_ai, 
                                strategic_review_30d, tactical_lookback_1d, per_stock_history
                            )
                            break
                        
                        except Exception as e:
                            if "429" in str(e):
                                log.warning(f"Quota error on attempt {attempt + 1}. Rotating key...")
                                if attempt < len(key_manager.api_keys) - 1: analyzer = AIAnalyzer(api_key=key_manager.rotate_key())
                                else: log.error("All API keys exhausted."); break
                            else: log.error(f"A non-quota error occurred: {e}"); break
                    
                    if not final_analysis: continue

                    original_signal = final_analysis.get('signal')
                    scrybe_score = final_analysis.get('scrybeScore', 0)
                    
                    is_quality_ok = True if original_signal != 'BUY' else fundamental_proxies.get("quality_score", 0) >= 40
                    is_conviction_ok = abs(scrybe_score) >= config.APEX_SWING_STRATEGY['min_conviction_score']
                    is_regime_ok = (original_signal == 'BUY' and current_regime == 'Bullish') or (original_signal == 'SELL' and current_regime == 'Bearish')
                    
                    final_signal = original_signal
                    filter_reason, reason_code = None, None
                    if original_signal in ['BUY', 'SELL']:
                        if not is_regime_ok: final_signal, filter_reason, reason_code = 'HOLD', f"Vetoed: Signal '{original_signal}' contradicts regime '{current_regime}'.", "REGIME_VETO"
                        elif not is_conviction_ok: final_signal, filter_reason, reason_code = 'HOLD', f"Vetoed: Conviction score ({scrybe_score}) is below threshold.", "LOW_CONVICTION"
                        elif not is_quality_ok: final_signal, filter_reason, reason_code = 'HOLD', f"Vetoed: Poor fundamental quality (Proxy Score: {fundamental_proxies.get('quality_score', 0)}).", "QUALITY_VETO"
                    
                    prediction_doc = final_analysis.copy()
                    prediction_doc['signal'] = final_signal
                    if filter_reason:
                        log.info(filter_reason)
                        prediction_doc['analystVerdict'] = filter_reason; prediction_doc['reason_code'] = reason_code
                    
                    if final_signal in ['BUY', 'SELL']:
                        entry_price = latest_row['Close']
                        predicted_gain = prediction_doc.get('predicted_gain_pct', 0)
                        stop_multiplier = config.APEX_SWING_STRATEGY['stop_loss_atr_multiplier']
                        atr = latest_row['ATRr_14']
                        stop_loss_price = entry_price - (stop_multiplier * atr) if final_signal == 'BUY' else entry_price + (stop_multiplier * atr)
                        target_price = entry_price * (1 + (predicted_gain / 100.0)) if final_signal == 'BUY' else entry_price * (1 - (predicted_gain / 100.0))
                        prediction_doc['tradePlan'] = { "entryPrice": round(entry_price, 2), "target": round(target_price, 2), "stopLoss": round(stop_loss_price, 2) }
                    
                    prediction_doc.update({
                        'analysis_id': str(uuid.uuid4()), 'ticker': ticker, 'prediction_date': current_day.to_pydatetime(),
                        'price_at_prediction': latest_row['Close'], 'status': 'open', 'strategy': config.APEX_SWING_STRATEGY['name'],
                        'atr_at_prediction': latest_row['ATRr_14']
                    })
                    database_manager.save_prediction_for_backtesting(prediction_doc, batch_id)
                    time.sleep(5)
                except Exception as e:
                    log.error(f"CRITICAL FAILURE on day {day_str} for {ticker}: {e}", exc_info=True)
            if i + 1 < len(simulation_days): save_state(simulation_days[i+1])
    if os.path.exists(STATE_FILE): os.remove(STATE_FILE)
    log.info("\n--- ✅ APEX Historical Run Finished! ---")
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