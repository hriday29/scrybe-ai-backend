# run_daily_jobs.py (APEX UNIFIED VERSION - FINAL)

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

def _get_live_per_stock_trade_history(ticker: str) -> str:
    """Gets the results of the last 3 closed trades for a specific stock from the live DB."""
    query = { "ticker": ticker }
    recent_trades = list(database_manager.live_performance_collection.find(query).sort("close_date", -1).limit(3))
    if not recent_trades:
        return "No recent live trade history for this stock."
    
    history_lines = []
    for i, trade in enumerate(recent_trades):
        line = (f"{i+1}. Signal: {trade.get('signal')}, "
                f"Outcome: {trade.get('return_pct'):.2f}% "
                f"({trade.get('closing_reason')})")
        history_lines.append(line)
    return "\n".join(history_lines)

def _get_live_30_day_performance_review() -> str:
    """Gets a 30-day performance review from the LIVE performance collection."""
    thirty_days_prior = datetime.now(timezone.utc) - timedelta(days=30)
    # Note: We query the live_performance_collection here
    query = {"close_date": {"$gte": thirty_days_prior}}
    recent_trades = list(database_manager.live_performance_collection.find(query))
    
    if not recent_trades:
        return "No live trading history in the last 30 days."

    df = pd.DataFrame(recent_trades)
    # The rest of the logic is identical to the historical version
    total_signals = len(df)
    win_rate = (df['return_pct'] > 0).mean() * 100
    gross_profit = df[df['return_pct'] > 0]['return_pct'].sum()
    gross_loss = abs(df[df['return_pct'] < 0]['return_pct'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    review = (f"Live 30-Day Performance Review:\n- Total Trades Closed: {total_signals}\n- Win Rate: {win_rate:.1f}%\n- Profit Factor: {profit_factor:.2f}")
    return review

def _get_live_1_day_tactical_lookback(ticker: str) -> str:
    """Gets the previous day's analysis for a ticker from the LIVE predictions collection."""
    # Look back 3 days to be safe and find the most recent entry
    previous_trading_day = datetime.now(timezone.utc) - timedelta(days=3)
    # Note: We query the live_predictions_collection here
    query = {"ticker": ticker, "prediction_date": {"$gte": previous_trading_day}}
    # Sort to get the absolute latest one
    last_analysis = database_manager.live_predictions_collection.find_one(query, sort=[("prediction_date", -1)])

    if not last_analysis:
        return "No live analysis for this stock in the last 2 trading days."
        
    lookback = (f"Previous Day's Live Note ({ticker}):\n- Signal: {last_analysis.get('signal', 'N/A')}\n- Scrybe Score: {last_analysis.get('scrybeScore', 0)}\n"
                f"- Key Insight: \"{last_analysis.get('keyInsight', 'N/A')}\"")
    return lookback

# --- END: LIVE HELPER FUNCTIONS ---


def manage_open_trades():
    """
    Finds all open trades, checks their status, and closes them if conditions are met.
    This version is corrected to use the single APEX strategy and the correct ATR column name.
    """
    log.info("--- ⚙️ Starting Open Trade Management ---")
    
    closed_trades_today = []
    open_trades_docs = list(database_manager.analysis_results_collection.find({"active_trade": {"$ne": None}}))
    
    if not open_trades_docs:
        log.info("No open trades to manage.")
        return closed_trades_today

    log.info(f"Found {len(open_trades_docs)} open trade(s) to evaluate.")
    for doc in open_trades_docs:
        trade = doc['active_trade']
        ticker = doc['ticker']
        try:
            live_data = data_retriever.get_live_financial_data(ticker)
            if not live_data or 'currentPrice' not in live_data.get('curatedData', {}):
                log.warning(f"Could not get live price for {ticker}. Skipping trade management.")
                continue
            latest_price = live_data['curatedData']['currentPrice']

            # --- START: TRAILING STOP LOGIC (Corrected) ---
            active_strategy = config.APEX_SWING_STRATEGY # Use the single, unified strategy
            
            # Check if trailing stop is enabled in the strategy config
            if active_strategy.get('use_trailing_stop', False):
                historical_data = data_retriever.get_historical_stock_data(ticker)
                if historical_data is not None and len(historical_data) > 14:
                    data_for_ta = historical_data.copy()
                    data_for_ta.ta.atr(length=14, append=True)
                    
                    # --- FIX: Use the correct 'ATRr_14' column name ---
                    if 'ATRr_14' in data_for_ta.columns:
                        latest_atr = data_for_ta['ATRr_14'].iloc[-1]
                        trailing_stop_atr_multiplier = active_strategy.get('trailing_stop_atr_multiplier', 1.5) # Assumes you add this key
                        
                        if trade['signal'] == 'BUY':
                            new_potential_stop = latest_price - (latest_atr * trailing_stop_atr_multiplier)
                            if new_potential_stop > trade['stop_loss']:
                                log.warning(f"Trailing Stop for {ticker} moved UP from {trade['stop_loss']:.2f} to {new_potential_stop:.2f}")
                                trade['stop_loss'] = new_potential_stop
                                database_manager.set_active_trade(ticker, trade)
                        elif trade['signal'] == 'SELL':
                            new_potential_stop = latest_price + (latest_atr * trailing_stop_atr_multiplier)
                            if new_potential_stop < trade['stop_loss']:
                                log.warning(f"Trailing Stop for {ticker} moved DOWN from {trade['stop_loss']:.2f} to {new_potential_stop:.2f}")
                                trade['stop_loss'] = new_potential_stop
                                database_manager.set_active_trade(ticker, trade)
            # --- END: TRAILING STOP LOGIC ---

            close_reason = None
            close_price = 0

            expiry_date = trade.get('expiry_date')
            if isinstance(expiry_date, str): expiry_date = datetime.fromisoformat(expiry_date)
            if expiry_date and expiry_date.tzinfo is None: expiry_date = expiry_date.replace(tzinfo=timezone.utc)
            
            if expiry_date and datetime.now(timezone.utc) >= expiry_date:
                close_reason, close_price = "Trade Closed - Expired", latest_price
            elif trade['signal'] == 'BUY':
                if latest_price >= trade['target']: close_reason, close_price = "Trade Closed - Target Hit", trade['target']
                elif latest_price <= trade['stop_loss']: close_reason, close_price = "Trade Closed - Stop-Loss Hit", trade['stop_loss']
            elif trade['signal'] == 'SELL':
                if latest_price <= trade['target']: close_reason, close_price = "Trade Closed - Target Hit", trade['target']
                elif latest_price >= trade['stop_loss']: close_reason, close_price = "Trade Closed - Stop-Loss Hit", trade['stop_loss']

            if close_reason:
                log.info(f"Closing trade for {ticker}. Reason: {close_reason}")
                closed_trade_doc = database_manager.close_live_trade(ticker, trade, close_reason, close_price)
                if closed_trade_doc: closed_trades_today.append(closed_trade_doc)
        except Exception as e:
            log.error(f"Error managing trade for {ticker}: {e}", exc_info=True)
            
    return closed_trades_today


def run_apex_analysis_pipeline(ticker: str, analyzer: AIAnalyzer, market_regime: str):
    """
    The "Apex-Aware" live analysis pipeline that mirrors historical_runner.py.
    """
    log.info(f"\n--- STARTING APEX LIVE ANALYSIS FOR: {ticker} ---")
    
    existing_trade_doc = database_manager.analysis_results_collection.find_one({"ticker": ticker, "active_trade": {"$ne": None}})
    if existing_trade_doc:
        log.warning(f"Skipping new analysis for {ticker} as it already has an active trade.")
        return None

    historical_data = data_retriever.get_historical_stock_data(ticker)
    if historical_data is None or len(historical_data) < 252:
        raise ValueError("Not enough historical data for Apex analysis.")
    
    data_for_ta = historical_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    data_for_ta.ta.bbands(length=20, append=True); data_for_ta.ta.rsi(length=14, append=True)
    data_for_ta.ta.macd(fast=12, slow=26, signal=9, append=True); data_for_ta.ta.adx(length=14, append=True)
    data_for_ta.ta.atr(length=14, append=True)
    
    indicator_cols = ['BBU_20_2.0', 'BBL_20_2.0', 'RSI_14', 'MACD_12_26_9', 'ADX_14', 'ATRr_14']
    for col in indicator_cols:
        if col in data_for_ta.columns: historical_data[col] = data_for_ta[col]

    latest_row = historical_data.iloc[-1]
    if latest_row[indicator_cols].isnull().any():
        raise ValueError("NaN indicators found in live data.")

    nifty_data = data_retriever.get_historical_stock_data("^NSEI")
    nifty_5d_change = (nifty_data['close'].iloc[-1] / nifty_data['close'].iloc[-6] - 1) * 100 if nifty_data is not None and len(nifty_data) > 5 else 0
    stock_5d_change = (latest_row['close'] / historical_data['close'].iloc[-6] - 1) * 100 if len(historical_data) > 5 else 0
    relative_strength = "Outperforming" if stock_5d_change > nifty_5d_change else "Underperforming"
    weekly_data = historical_data['close'].resample('W').last()
    weekly_trend = "Bullish" if len(weekly_data) > 2 and weekly_data.iloc[-1] > weekly_data.iloc[-2] else "Bearish"
    fundamental_proxies = data_retriever.get_fundamental_proxies(historical_data)
    options_data = data_retriever.get_options_data(ticker)
    news_data = data_retriever.get_news_articles_for_ticker(ticker)

    full_context_for_ai = {
        "layer_1_macro_context": {"nifty_50_regime": market_regime},
        "layer_2_relative_strength": {"stock_5d_change_pct": f"{stock_5d_change:.2f}%", "relative_strength_vs_nifty50": relative_strength},
        "layer_3_fundamental_moat": fundamental_proxies,
        "layer_4_technicals": {"daily_indicators": {"ADX": f"{latest_row['ADX_14']:.2f}", "RSI": f"{latest_row['RSI_14']:.2f}"}, "weekly_trend": weekly_trend},
        "layer_5_options_sentiment": options_data if options_data else {"sentiment": "N/A"},
        "layer_6_news_catalyst": news_data if news_data else {"summary": "N/A"}
    }

    strategic_review_30d = _get_live_30_day_performance_review()
    tactical_lookback_1d = _get_live_1_day_tactical_lookback(ticker)
    per_stock_history = _get_live_per_stock_trade_history(ticker)

    final_analysis = analyzer.get_apex_analysis(
        ticker=ticker,
        full_context=full_context_for_ai,
        strategic_review=strategic_review_30d,
        tactical_lookback=tactical_lookback_1d,
        per_stock_history=per_stock_history
    )

    if not final_analysis: raise ValueError("Apex AI failed to return an analysis.")

    final_analysis.update({
        'analysis_id': str(uuid.uuid4()), 'timestamp': datetime.now(timezone.utc),
        'ticker': ticker, 'price_at_prediction': latest_row['close'],
        'atr_at_prediction': latest_row['ATRr_14']
    })
    
    log.info(f"--- ✅ SUCCESS: APEX live analysis complete for {ticker} ---")
    return final_analysis

def run_all_jobs():
    """
    The master "Apex-Aware" job runner.
    """
    log.info("--- 🚀 Starting all daily jobs (APEX UNIFIED VERSION) ---")

    # --- Tracking lists for notifier ---
    new_signals_today = []
    closed_trades_today = []

    try:
        database_manager.init_db(purpose='analysis')
        # --- START: Live Volatility Filter ---
        vix_data = data_retriever.get_historical_stock_data("^INDIAVIX")
        volatility_regime = data_retriever.get_volatility_regime(vix_data)
        
        if volatility_regime == "High-Risk":
            log.warning("VOLATILITY FILTER ENGAGED: Live market is in 'High-Risk' state. Standing aside for today.")
            # We can optionally send a notification that no trades will be generated today
            # send_daily_briefing(...) 
            return # Exit the entire job runner for the day
        # --- END: Live Volatility Filter ---
        key_manager = APIKeyManager(api_keys=config.GEMINI_API_KEY_POOL)
        analyzer = AIAnalyzer(api_key=key_manager.get_key())

        # --- Manage existing trades ---
        closed_trades_today = manage_open_trades()
        market_regime = data_retriever.get_market_regime()
        
        tickers_to_run = config.LIVE_TRADING_UNIVERSE

        for ticker in tickers_to_run:
            final_analysis = None
            try:
                final_analysis = run_apex_analysis_pipeline(ticker, analyzer, market_regime)
                if not final_analysis:
                    continue
                
                active_strategy = config.APEX_SWING_STRATEGY
                log.info(f"Applying Profile ('{active_strategy['name']}') for {ticker}")
                
                original_signal = final_analysis.get('signal')
                scrybe_score = final_analysis.get('scrybeScore', 0)
                
                fundamental_data = final_analysis.get("layer_3_fundamental_moat", {})
                live_quality_score = fundamental_data.get("quality_score", 0)
                is_quality_ok = True if original_signal != 'BUY' else live_quality_score >= 40
                is_conviction_ok = abs(scrybe_score) >= active_strategy['min_conviction_score']
                is_regime_ok = (
                    (original_signal == 'BUY' and market_regime == 'Bullish') or
                    (original_signal == 'SELL' and market_regime == 'Bearish')
                )
                
                final_signal = original_signal
                filter_reason, reason_code = None, None
                if original_signal in ['BUY', 'SELL']:
                    if not is_regime_ok:
                        final_signal, filter_reason, reason_code = 'HOLD', f"Vetoed: Signal '{original_signal}' contradicts regime '{market_regime}'.", "REGIME_VETO"
                    elif not is_conviction_ok:
                        final_signal, filter_reason, reason_code = 'HOLD', f"Vetoed: Conviction score ({scrybe_score}) is below threshold.", "LOW_CONVICTION"
                    elif not is_quality_ok:
                        final_signal, filter_reason, reason_code = 'HOLD', f"Vetoed: Poor fundamental quality (Proxy Score: {live_quality_score}).", "QUALITY_VETO"

                prediction_doc = final_analysis.copy()
                prediction_doc['signal'] = final_signal
                if filter_reason:
                    log.info(filter_reason)
                    prediction_doc['analystVerdict'] = filter_reason
                    prediction_doc['reason_code'] = reason_code

                if final_signal in ['BUY', 'SELL']:
                    entry_price = prediction_doc['price_at_prediction']
                    predicted_gain = prediction_doc.get('predicted_gain_pct', 0)
                    stop_multiplier = active_strategy['stop_loss_atr_multiplier']
                    atr = prediction_doc['atr_at_prediction']
                    
                    stop_loss_price = (
                        entry_price - (stop_multiplier * atr)
                        if final_signal == 'BUY'
                        else entry_price + (stop_multiplier * atr)
                    )
                    target_price = (
                        entry_price * (1 + (predicted_gain / 100.0))
                        if final_signal == 'BUY'
                        else entry_price * (1 - (predicted_gain / 100.0))
                    )
                    
                    prediction_doc['tradePlan'] = {
                        "entryPrice": round(entry_price, 2),
                        "target": round(target_price, 2),
                        "stopLoss": round(stop_loss_price, 2)
                    }
                    
                    trade_object = {
                        "signal": final_signal,
                        "strategy": active_strategy['name'],
                        "entry_price": entry_price,
                        "entry_date": prediction_doc['timestamp'],
                        "target": round(target_price, 2),
                        "stop_loss": round(stop_loss_price, 2),
                        "expiry_date": datetime.now(timezone.utc) + timedelta(days=active_strategy['holding_period']),
                        "confidence": prediction_doc.get('confidence'),
                        "scrybeScore": scrybe_score   # ✅ added for DB consistency
                    }
                    database_manager.set_active_trade(ticker, trade_object)

                    # --- Capture for email ---
                    new_signals_today.append({
                        "ticker": ticker,
                        "signal": final_signal,
                        "scrybeScore": scrybe_score,
                        "confidence": prediction_doc.get('confidence')
                    })
                else:
                    database_manager.set_active_trade(ticker, None)

                database_manager.save_live_prediction(prediction_doc)
                database_manager.save_vst_analysis(ticker, prediction_doc)
                time.sleep(5)

            except Exception as e:
                log.error(f"CRITICAL FAILURE processing {ticker}: {e}", exc_info=True)
                if "429" in str(e):
                    log.warning("Quota reached. Rotating API key...")
                    analyzer = AIAnalyzer(api_key=key_manager.rotate_key())
                continue
        
        # --- Email notifier integration ---
        from notifier import send_daily_briefing
        try:
            send_daily_briefing(new_signals=new_signals_today, closed_trades=closed_trades_today)
        except Exception as e:
            log.error(f"Failed to send daily briefing: {e}", exc_info=True)

    finally:
        database_manager.close_db_connection()
    log.info("--- 🔒 Database connection closed ---")
    log.info("--- ✅ Daily jobs completed ---")

    # Optional: return results for debugging
    return {"new_signals": new_signals_today, "closed_trades": closed_trades_today}

if __name__ == "__main__":
    run_all_jobs()
