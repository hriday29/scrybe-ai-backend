# run_daily_jobs.py
import time
from datetime import datetime, timezone, timedelta
import database_manager
import config
from logger_config import log
from ai_analyzer import AIAnalyzer
import data_retriever
import performance_analyzer
import technical_analyzer
from live_reporter import generate_performance_report
import uuid
from notifier import send_daily_briefing
from index_manager import get_nifty50_tickers
from utils import APIKeyManager

def manage_open_trades():
    """
    Finds all open trades, checks their status, and closes them if conditions are met.
    Returns a list of any trades that were closed during the run.
    """
    log.info("--- ⚙️ Starting Open Trade Management ---")
    database_manager.init_db(purpose='analysis')
    
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
                log.warning(f"Could not get live price for {ticker}. Skipping trade management for now.")
                continue
            latest_price = live_data['curatedData']['currentPrice']

            close_reason = None
            close_price = 0

            # --- DEFINITIVE FIX FOR DATETIME ERROR ---
            expiry_date = trade.get('expiry_date')
            if isinstance(expiry_date, str):
                expiry_date = datetime.fromisoformat(expiry_date)

            # This is the critical fix: make sure the date from the DB is timezone-aware
            if expiry_date and expiry_date.tzinfo is None:
                expiry_date = expiry_date.replace(tzinfo=timezone.utc)

            if expiry_date and datetime.now(timezone.utc) >= expiry_date:
            # --- END FIX ---
                close_reason = "Trade Closed - Expired"
                close_price = latest_price
            elif trade['signal'] == 'BUY':
                if latest_price >= trade['target']:
                    close_reason, close_price = "Trade Closed - Target Hit", trade['target']
                elif latest_price <= trade['stop_loss']:
                    close_reason, close_price = "Trade Closed - Stop-Loss Hit", trade['stop_loss']
            elif trade['signal'] == 'SELL':
                if latest_price <= trade['target']:
                    close_reason, close_price = "Trade Closed - Target Hit", trade['target']
                elif latest_price >= trade['stop_loss']:
                    close_reason, close_price = "Trade Closed - Stop-Loss Hit", trade['stop_loss']

            if close_reason:
                log.info(f"Closing trade for {ticker}. Reason: {close_reason}")
                closed_trade_doc = database_manager.close_live_trade(ticker, trade, close_reason, close_price)
                if closed_trade_doc:
                    closed_trades_today.append(closed_trade_doc)
        except Exception as e:
            log.error(f"Error managing trade for {ticker}: {e}", exc_info=True)
            
    return closed_trades_today

def _validate_analysis_output(analysis: dict, ticker: str) -> bool:
    if not analysis or 'signal' not in analysis:
        log.error(f"VALIDATION FAILED: AI analysis for {ticker} is invalid or None.")
        return False
    return True

def _validate_trade_plan(analysis: dict) -> bool:
    try:
        if analysis['signal'] == 'HOLD': return True
        entry = analysis['price_at_prediction']
        target = float(analysis['tradePlan']['target']['price'])
        stop = float(analysis['tradePlan']['stopLoss']['price'])
        if analysis['signal'] == 'BUY' and (target < entry or stop > entry): return False
        if analysis['signal'] == 'SELL' and (target > entry or stop < entry): return False
    except (KeyError, ValueError, TypeError): return False
    return True

def run_vst_analysis_pipeline(ticker: str, analyzer: AIAnalyzer, market_data: dict, market_regime: str, macro_data: dict):
    """A streamlined pipeline that only runs the VST analysis."""
    try:
        log.info(f"\n--- STARTING VST ANALYSIS FOR: {ticker} (Market Regime: {market_regime}) ---")
        
        database_manager.init_db(purpose='analysis')
        existing_trade_doc = database_manager.analysis_results_collection.find_one({"ticker": ticker})
        if existing_trade_doc and existing_trade_doc.get('active_trade'):
            log.warning(f"Skipping new analysis for {ticker} as it already has an active trade.")
            return None

        historical_data = data_retriever.get_historical_stock_data(ticker)
        if historical_data is None or len(historical_data) < 50: raise ValueError("Not enough historical data.")

        live_financial_data = data_retriever.get_live_financial_data(ticker)
        if not live_financial_data: raise ValueError("Could not get live financial data.")
        
        options_data = data_retriever.get_options_data(ticker)
        correlations = performance_analyzer.calculate_correlations(historical_data, data_retriever.get_benchmarks_data())
        
        historical_data.ta.bbands(length=20, append=True); historical_data.ta.rsi(length=14, append=True)
        historical_data.ta.macd(fast=12, slow=26, signal=9, append=True); historical_data.ta.adx(length=14, append=True)
        historical_data.ta.atr(length=14, append=True); historical_data.dropna(inplace=True)
        
        charts = technical_analyzer.generate_focused_charts(historical_data, ticker)
        latest_row = historical_data.iloc[-1]
        volume_ma_20 = historical_data['volume'].rolling(window=20).mean().iloc[-1]
        is_volume_high = latest_row['volume'] > volume_ma_20 * config.VOLUME_SURGE_THRESHOLD
        
        latest_indicators = {
            "ADX": f"{latest_row['ADX_14']:.2f}", "RSI": f"{latest_row['RSI_14']:.2f}",
            "MACD": f"{latest_row['MACD_12_26_9']:.2f}",
            "Bollinger Band Width Percent": f"{(latest_row['BBU_20_2.0'] - latest_row['BBL_20_2.0']) / latest_row['BBM_20_2.0'] * 100:.2f}",
            "Volume Surge": "Yes" if is_volume_high else "No"
        }
        
        market_context = {
            "stock_sector": market_data.get("stock_performance", {}).get(ticker, {}).get('sector', 'Unknown'), 
            "sector_performance_today": market_data.get("sector_performance", {}),
            "CURRENT_MARKET_REGIME": market_regime
        }
        
        strategy_config = config.VST_STRATEGY
        analysis_result = analyzer.get_stock_analysis(
            live_financial_data=live_financial_data, latest_atr=latest_row['ATRr_14'], model_name=config.PRO_MODEL,
            charts=charts, trading_horizon_text=strategy_config['horizon_text'],
            technical_indicators=latest_indicators, min_rr_ratio=strategy_config.get('min_rr_ratio', 2.0),
            market_context=market_context, options_data=options_data,
            macro_data=macro_data
        )
        if not analysis_result: raise ValueError("The main AI stock analysis failed.")

        analysis_result.update({
            'analysis_id': str(uuid.uuid4()),
            'charts': charts, 'signalQualifier': analyzer.get_volatility_qualifier(latest_indicators),
            'dvmScores': analyzer.get_dvm_scores(live_financial_data, latest_indicators),
            'companyName': live_financial_data.get('rawDataSheet', {}).get('longName', ticker),
            'performanceSnapshot': performance_analyzer.calculate_historical_performance(historical_data),
            'timestamp': datetime.now(timezone.utc), 'ticker': ticker, 'correlations': correlations,
            'price_at_prediction': latest_row['close'], 'prediction_date': datetime.now(timezone.utc),
            'strategy': strategy_config['name'],
            'atr_at_prediction': latest_row['ATRr_14'],
            'market_regime_at_analysis': market_regime
        })
        
        log.info(f"--- ✅ SUCCESS: VST analysis complete for {ticker} ---")
        return analysis_result
    except Exception as e:
            # If the error is a quota error, we MUST re-raise it
            # so the outer rotation loop in run_all_jobs() can catch it and rotate the key.
            if "429" in str(e) and "quota" in str(e).lower():
                raise e
            
            # For all other types of errors, we log them and continue to the next stock.
            log.error(f"--- ❌ FAILURE: VST analysis for {ticker} failed. Error: {e} ---", exc_info=True)
            return None

def run_all_jobs():
    """
    Runs the VST-only analysis pipeline with Market Regime and Email Notifications.
    This version includes resilient API key rotation.
    """
    log.info("--- 🚀 Kicking off ALL DAILY JOBS (VST-Only Mode) ---")
    
    try:
        # --- MODIFICATION: Initialize DB and AI Manager once at the start ---
        database_manager.init_db(purpose='analysis')
        
        try:
            key_manager = APIKeyManager(api_keys=config.GEMINI_API_KEY_POOL)
            analyzer = AIAnalyzer(api_key=key_manager.get_key())
        except (ValueError, AttributeError) as e: 
            log.fatal(f"Failed to initialize AI Analyzer. Check GEMINI_API_KEY_POOL in config. Error: {e}. Aborting.")
            return

        closed_trades = manage_open_trades()

        log.info("--- 🔍 Starting New Analysis Generation ---")
        
        market_data = data_retriever.get_nifty50_performance()
        if not market_data: 
            log.error("Could not fetch market performance data. Aborting.")
            return

        market_regime = data_retriever.get_market_regime()
        log.info("Fetching live benchmark data for macro context...")
        benchmarks_data = data_retriever.get_benchmarks_data(period="10d") # Fetch recent data
        macro_data = {}
        if benchmarks_data is not None and not benchmarks_data.empty:
            if len(benchmarks_data) > 5:
                # Calculate 5-day percentage change
                changes = ((benchmarks_data.iloc[-1] - benchmarks_data.iloc[-6]) / benchmarks_data.iloc[-6]) * 100
                macro_data = {
                    "Nifty50_5D_Change": f"{changes.get('Nifty50', 0):.2f}%",
                    "CrudeOil_5D_Change": f"{changes.get('Crude Oil', 0):.2f}%",
                    "Gold_5D_Change": f"{changes.get('Gold', 0):.2f}%",
                    "USDINR_5D_Change": f"{changes.get('USD-INR', 0):.2f}%",
                }
        log.info(f"Macro Context for today: {macro_data}")
        new_signals = [] 
        nifty50_tickers = config.NIFTY_50_TICKERS

        log.info(f"--- Starting analysis for {len(nifty50_tickers)} tickers ---")
        for ticker in nifty50_tickers:
            # --- MODIFICATION: Resilient analysis loop with key rotation ---
            max_retries = len(key_manager.api_keys)
            vst_analysis = None
            for attempt in range(max_retries):
                try:
                    # The analysis pipeline is called within the try block
                    vst_analysis = run_vst_analysis_pipeline(ticker, analyzer, market_data, market_regime, macro_data=macro_data)
                    # If the call succeeds, we break out of the retry loop
                    break 
                
                except Exception as e:
                    log.warning(f"Analysis pipeline failed for {ticker} on attempt {attempt + 1}. Error: {e}")
                    # If it's a quota error and we haven't run out of keys, we rotate and retry
                    if "429" in str(e) and "quota" in str(e).lower() and attempt < max_retries - 1:
                        new_key = key_manager.rotate_key()
                        analyzer = AIAnalyzer(api_key=new_key) # Re-initialize analyzer with new key
                        log.info("Retrying analysis with the new key...")
                        time.sleep(2)
                    # If it's another type of error, or we've run out of keys, we stop trying for this stock
                    else:
                        log.error(f"Could not recover from error for {ticker}. Analysis for this stock failed.")
                        break # Exit the retry loop
            # --- End of resilient loop ---
            
            if not vst_analysis or not _validate_analysis_output(vst_analysis, ticker) or not _validate_trade_plan(vst_analysis):
                log.warning(f"No valid VST analysis was generated for {ticker}.")
                if vst_analysis:
                    database_manager.save_vst_analysis(ticker, vst_analysis)
                continue

            database_manager.save_vst_analysis(ticker, vst_analysis)
            database_manager.save_live_prediction(vst_analysis)
            
            original_signal = vst_analysis.get('signal')
            scrybe_score = vst_analysis.get('scrybeScore', 0)
            dvm_scores = vst_analysis.get('dvmScores', {})

            final_signal = original_signal
            filter_reason = None

            # Rule 1: The Regime Filter
            if not ((original_signal == 'BUY' and market_regime == 'Bullish') or \
                (original_signal == 'SELL' and market_regime == 'Bearish') or \
                (original_signal == 'HOLD')):
                final_signal = 'HOLD'
                filter_reason = f"Signal '{original_signal}' was vetoed by the Risk Manager due to an unfavorable market regime ('{market_regime}')."

            # Rule 2: The Conviction Filter
            elif abs(scrybe_score) < 60 and original_signal != 'HOLD':
                final_signal = 'HOLD'
                filter_reason = f"Signal '{original_signal}' (Score: {scrybe_score}) was vetoed by the Risk Manager because it did not meet the high-conviction threshold (>60)."

            # Rule 3: The Quality Filter
            elif dvm_scores and dvm_scores.get('durability', {}).get('score', 100) < 40:
                final_signal = 'HOLD'
                filter_reason = f"Signal '{original_signal}' was vetoed by the Risk Manager due to a poor fundamental Quality (Durability) score."

            vst_analysis['signal'] = final_signal
            if filter_reason:
                log.warning(filter_reason)
                vst_analysis['analystVerdict'] = filter_reason
                vst_analysis['tradePlan'] = {"status": "Filtered by Risk Manager", "reason": filter_reason}
                database_manager.save_vst_analysis(ticker, vst_analysis)

            # The existing trade creation logic now runs on the final, filtered signal
            if vst_analysis.get('signal') in ['BUY', 'SELL']:
                log.info(f"PASSED FILTERS: Creating live trade for '{vst_analysis['signal']}' signal on {ticker}.")
                new_signals.append({
                    "ticker": ticker,
                    "signal": vst_analysis.get('signal'),
                    "confidence": vst_analysis.get('confidence'),
                    "scrybeScore": vst_analysis.get('scrybeScore')
                })
                try:
                    # Deterministic trade plan calculation
                    signal = vst_analysis['signal']
                    entry_price = vst_analysis['price_at_prediction']
                    atr = vst_analysis.get('atr_at_prediction')

                    if not atr:
                        log.error(f"Could not find ATR in analysis for {ticker}. Cannot create trade object.")
                        database_manager.set_active_trade(ticker, None)
                        continue # Skips to the next ticker in the loop

                    rr_ratio = config.VST_STRATEGY['min_rr_ratio']
                    stop_loss_price = entry_price - (2 * atr) if signal == 'BUY' else entry_price + (2 * atr)
                    target_price = entry_price + ((2 * atr) * rr_ratio) if signal == 'BUY' else entry_price - ((2 * atr) * rr_ratio)

                    trade_object = {
                        "signal": signal,
                        "strategy": vst_analysis.get('strategy'),
                        "entry_price": entry_price,
                        "entry_date": vst_analysis.get('prediction_date'),
                        "target": round(target_price, 2),
                        "stop_loss": round(stop_loss_price, 2),
                        "risk_reward_ratio": rr_ratio,
                        "expiry_date": datetime.now(timezone.utc) + timedelta(days=config.TRADE_EXPIRY_DAYS),
                        "confidence": vst_analysis.get('confidence')
                    }
                    database_manager.set_active_trade(ticker, trade_object)

                except Exception as e:
                    log.error(f"Could not create trade object for {ticker}: {e}", exc_info=True)
                    database_manager.set_active_trade(ticker, None)
            else:
                log.info(f"Signal for {ticker} is 'HOLD' after filtering. No active trade will be set.")
                database_manager.set_active_trade(ticker, None)
            
            time.sleep(10)

        log.info("--- ✅ All daily jobs finished ---")
        generate_performance_report()
        send_daily_briefing(new_signals, closed_trades)
    
    finally:
        database_manager.close_db_connection()
        
if __name__ == "__main__":
    run_all_jobs()