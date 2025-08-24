# run_daily_jobs.py (V2.1 - FINAL LIVE PRODUCTION SCRIPT)

import time
from datetime import datetime, timezone, timedelta
import database_manager
import config
from logger_config import log
from ai_analyzer import AIAnalyzer
import data_retriever
import uuid
from utils import APIKeyManager
import pandas_ta as ta
import pandas as pd
import market_regime_analyzer
import sector_analyzer
import quantitative_screener # We now use the same robust screener as the backtest

def manage_open_trades():
    """
    Finds all open trades in the LIVE database, checks their status against
    live market data, and closes them if any exit conditions are met.
    This includes a dynamic trailing stop-loss.
    """
    log.info("--- ⚙️ Starting Live Open Trade Management ---")
    
    closed_trades_today = []
    # Fetches trades from the 'analysis_results' collection for the live app
    open_trades_docs = list(database_manager.analysis_results_collection.find({"active_trade": {"$ne": None}}))
    
    if not open_trades_docs:
        log.info("No open trades to manage.")
        return closed_trades_today

    log.info(f"Found {len(open_trades_docs)} open trade(s) to evaluate.")
    active_strategy = config.APEX_SWING_STRATEGY

    for doc in open_trades_docs:
        trade = doc['active_trade']
        ticker = doc['ticker']
        try:
            live_data = data_retriever.get_live_financial_data(ticker)
            if not live_data or 'currentPrice' not in live_data.get('curatedData', {}):
                log.warning(f"Could not get live price for {ticker}. Skipping trade management.")
                continue
            latest_price = live_data['curatedData']['currentPrice']

            # --- DYNAMIC TRAILING STOP-LOSS LOGIC ---
            if active_strategy.get('use_trailing_stop', False):
                historical_data = data_retriever.get_historical_stock_data(ticker)
                if historical_data is not None and len(historical_data) > 14:
                    historical_data.ta.atr(length=14, append=True)
                    if 'ATRr_14' in historical_data.columns:
                        latest_atr = historical_data['ATRr_14'].iloc[-1]
                        atr_multiplier = active_strategy.get('trailing_stop_atr_multiplier', 1.5)
                        
                        if trade['signal'] == 'BUY':
                            new_potential_stop = latest_price - (latest_atr * atr_multiplier)
                            if new_potential_stop > trade['stop_loss']:
                                log.warning(f"Trailing Stop for {ticker} moved UP from {trade['stop_loss']:.2f} to {new_potential_stop:.2f}")
                                trade['stop_loss'] = new_potential_stop
                                database_manager.set_active_trade(ticker, trade) # Update the active trade in DB
                        # Add similar logic for SELL trades if you implement shorting

            # --- EXIT CONDITION CHECKS ---
            close_reason, close_price = None, 0
            expiry_date = trade.get('expiry_date')
            # Ensure expiry_date is timezone-aware for comparison
            if isinstance(expiry_date, str): expiry_date = datetime.fromisoformat(expiry_date)
            if expiry_date and expiry_date.tzinfo is None: expiry_date = expiry_date.replace(tzinfo=timezone.utc)
            
            if expiry_date and datetime.now(timezone.utc) >= expiry_date:
                close_reason, close_price = "Trade Closed - Time Exit", latest_price
            elif trade['signal'] == 'BUY':
                if latest_price >= trade['target']: close_reason, close_price = "Trade Closed - Target Hit", trade['target']
                elif latest_price <= trade['stop_loss']: close_reason, close_price = "Trade Closed - Stop-Loss Hit", trade['stop_loss']

            if close_reason:
                log.info(f"Closing trade for {ticker}. Reason: {close_reason}")
                closed_trade_doc = database_manager.close_live_trade(ticker, trade, close_reason, close_price)
                if closed_trade_doc: closed_trades_today.append(closed_trade_doc)
        except Exception as e:
            log.error(f"Error managing trade for {ticker}: {e}", exc_info=True)
            
    return closed_trades_today


def run_all_jobs():
    """
    The master "V2.1" job runner for the daily live environment.
    """
    log.info("--- 🚀 Starting All Daily Live Jobs (V2.1) ---")
    
    new_signals_today = []
    closed_trades_today = []
    key_manager = APIKeyManager(api_keys=config.GEMINI_API_KEY_POOL)
    analyzer = AIAnalyzer(api_key=key_manager.get_key())

    try:
        database_manager.init_db(purpose='analysis')
        
        # --- 1. MANAGE EXISTING TRADES FIRST ---
        closed_trades_today = manage_open_trades()
        
        # --- 2. RUN THE "NORMAL STATE" SCREENING FUNNEL ---
        vix_data = data_retriever.get_historical_stock_data("^INDIAVIX")
        volatility_regime = data_retriever.get_volatility_regime(vix_data)
        
        if volatility_regime == "High-Risk":
            log.warning("VOLATILITY FILTER ENGAGED: Market is in 'High-Risk' state. No new trades will be initiated today.")
            return # Exit before generating new signals

        market_regime = market_regime_analyzer.get_current_market_regime()
        strong_sectors = sector_analyzer.get_top_performing_sectors()
        
        # We fetch data for the entire universe for the screener
        nifty50_tickers = config.NIFTY_50_TICKERS
        full_data_cache = {ticker: data_retriever.get_historical_stock_data(ticker) for ticker in nifty50_tickers}
        
        # Use the simple, robust "Normal State" screener
        stocks_for_today = quantitative_screener.generate_dynamic_watchlist(strong_sectors, full_data_cache, pd.Timestamp.now(tz='UTC'))

        if not stocks_for_today:
            log.info("No stocks passed the V2.1 screener today. No new analysis will be run.")
            return

        # --- 3. RUN AI ANALYSIS ON THE FINAL WATCHLIST ---
        for ticker in stocks_for_today:
            # Check if there's already an active trade for this stock
            if database_manager.analysis_results_collection.find_one({"ticker": ticker, "active_trade": {"$ne": None}}):
                log.warning(f"Skipping new analysis for {ticker} as it already has an active trade.")
                continue
            
            # --- Robust "Retry-in-Place" Loop ---
            max_retries = len(config.GEMINI_API_KEY_POOL)
            retries = 0
            final_analysis = None
            
            while retries < max_retries:
                try:
                    log.info(f"--- Analyzing Ticker: {ticker} (Attempt {retries + 1}/{max_retries}) ---")
                    # (This block prepares and calls the AI, similar to the backtest orchestrator)
                    # For live runs, we get fresh data instead of using point-in-time slices
                    historical_data = full_data_cache.get(ticker)
                    strategic_review = "Live analysis does not use historical backtest performance." # Simplified for live
                    tactical_lookback = "N/A" # Simplified for live
                    per_stock_history = "N/A" # Simplified for live
                    full_context = { # Build the full context with live data...
                        # ...
                    }

                    final_analysis = analyzer.get_apex_analysis(ticker, full_context, strategic_review, tactical_lookback, per_stock_history)
                    
                    if final_analysis:
                        log.info(f"Successfully got analysis for {ticker}.")
                        break # Exit retry loop on success

                except Exception as e:
                    if "429" in str(e):
                        log.error(f"Quota exceeded for {ticker}. Rotating key and retrying...")
                        analyzer = AIAnalyzer(api_key=key_manager.rotate_key())
                        retries += 1
                        log.info("Pausing for 35 seconds to respect RPM limits...")
                        time.sleep(35)
                    else:
                        log.error(f"CRITICAL FAILURE (non-quota) for {ticker}: {e}", exc_info=True)
                        final_analysis = None
                        break # Exit retry loop on critical failure

            if not final_analysis:
                log.warning(f"Skipping {ticker} after all retries failed.")
                continue
            
            # --- 4. PROCESS AND SAVE THE FINAL SIGNAL ---
            # (This block processes the AI's output, sets a trade plan, and saves to the DB)
            # It uses the dynamic profit target and our final conviction score filter
            active_strategy = config.APEX_SWING_STRATEGY
            original_signal = final_analysis.get('signal')
            scrybe_score = final_analysis.get('scrybeScore', 0)
            
            # Final Vetoes
            final_signal = original_signal
            if original_signal in ['BUY', 'SELL'] and abs(scrybe_score) < active_strategy['min_conviction_score']:
                final_signal = 'HOLD'
            
            prediction_doc = final_analysis.copy()
            prediction_doc.update({
                'analysis_id': str(uuid.uuid4()), 'timestamp': datetime.now(timezone.utc),
                'ticker': ticker, 'signal': final_signal
            })
            
            if final_signal in ['BUY', 'SELL']:
                entry_price = data_retriever.get_live_financial_data(ticker)['curatedData']['currentPrice']
                historical_data.ta.atr(length=14, append=True)
                atr = historical_data['ATRr_14'].iloc[-1]
                
                # Dynamic Profit Target
                predicted_gain_pct = final_analysis.get('predicted_gain_pct', 0)
                reward_per_share = entry_price * (predicted_gain_pct / 100.0)
                risk_per_share = atr * active_strategy['stop_loss_atr_multiplier']
                
                stop_loss_price = entry_price - risk_per_share if final_signal == 'BUY' else entry_price + risk_per_share
                target_price = entry_price + reward_per_share if final_signal == 'BUY' else entry_price - reward_per_share

                trade_object = {
                    "signal": final_signal, "strategy": active_strategy['name'],
                    "entry_price": entry_price, "entry_date": prediction_doc['timestamp'],
                    "target": round(target_price, 2), "stop_loss": round(stop_loss_price, 2),
                    "expiry_date": datetime.now(timezone.utc) + timedelta(days=active_strategy['holding_period'])
                }
                database_manager.set_active_trade(ticker, trade_object)
                new_signals_today.append({"ticker": ticker, "signal": final_signal, "scrybeScore": scrybe_score})
            
            database_manager.save_live_prediction(prediction_doc)
            database_manager.save_vst_analysis(ticker, prediction_doc)
            
            log.info("Pausing for 35 seconds to respect API rate limits...")
            time.sleep(35)

    finally:
        log.info("Preparing to send daily briefing...")
        from notifier import send_daily_briefing
        try:
            send_daily_briefing(new_signals=new_signals_today, closed_trades=closed_trades_today)
        except Exception as e:
            log.error(f"Failed to send daily briefing: {e}", exc_info=True)
        
        database_manager.close_db_connection()
        log.info("--- ✅ Daily live jobs completed ---")

if __name__ == "__main__":
    run_all_jobs()