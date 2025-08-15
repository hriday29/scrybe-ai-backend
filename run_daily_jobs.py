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

def run_vst_analysis_pipeline(ticker: str, analyzer: AIAnalyzer, market_data: dict, market_regime: str, macro_data: dict, active_strategy: dict):
    """
    A streamlined pipeline that runs the multi-strategy AI analysis.
    NOTE: This pipeline now returns a TUPLE of (final_analysis, dvm_scores)
    """
    try:
        log.info(f"\n--- STARTING VST ANALYSIS FOR: {ticker} (Market Regime: {market_regime}) ---")
        
        database_manager.init_db(purpose='analysis')
        existing_trade_doc = database_manager.analysis_results_collection.find_one({"ticker": ticker})
        if existing_trade_doc and existing_trade_doc.get('active_trade'):
            log.warning(f"Skipping new analysis for {ticker} as it already has an active trade.")
            return None, None

        historical_data = data_retriever.get_historical_stock_data(ticker)
        if historical_data is None or len(historical_data) < 50: raise ValueError("Not enough historical data.")

        live_financial_data = data_retriever.get_live_financial_data(ticker)
        if not live_financial_data: raise ValueError("Could not get live financial data.")
        
        options_data = data_retriever.get_options_data(ticker)
        
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
        candle_high = latest_row['high']
        candle_low = latest_row['low']
        candle_close = latest_row['close']
        candle_range = candle_high - candle_low if candle_high > candle_low else 0.01
        
        # Position of close within the candle's full range (0=low, 1=high)
        position_in_range = (candle_close - candle_low) / candle_range if candle_range > 0 else 0.5
        atr_percent = (latest_row['ATRr_14'] / latest_row['close']) * 100
        latest_indicators['ATR_Percent'] = f"{atr_percent:.2f}%"
        
        latest_indicators['Confirmation_Candle'] = {
            "position_in_range": round(position_in_range, 2)
        }
        
        market_context = {
            "stock_sector": market_data.get("stock_performance", {}).get(ticker, {}).get('sector', 'Unknown'), 
            "sector_performance_today": market_data.get("sector_performance", {}),
            "CURRENT_MARKET_REGIME": market_regime
        }
        
        # --- "CIO" LOGIC FOR LIVE RUNNER ---
        log.info(f"Querying Momentum Specialist for {ticker}...")
        momentum_analysis = analyzer.get_momentum_analysis(
            live_financial_data=live_financial_data, latest_atr=latest_row['ATRr_14'], model_name=config.PRO_MODEL,
            charts=charts, trading_horizon_text=active_strategy['horizon_text'],
            technical_indicators=latest_indicators, min_rr_ratio=active_strategy['min_rr_ratio'],
            market_context=market_context, options_data=options_data, macro_data=macro_data
        )

        log.info(f"Querying Mean-Reversion Specialist for {ticker}...")
        mean_reversion_analysis = analyzer.get_mean_reversion_analysis(
            live_financial_data=live_financial_data, model_name=config.FLASH_MODEL,
            technical_indicators=latest_indicators, market_context=market_context
        )

        # --- START: 3-Specialist CIO Decision Logic ---
        log.info(f"Querying Breakout Specialist for {ticker}...")
        breakout_analysis = analyzer.get_breakout_analysis(
            live_financial_data=live_financial_data, model_name=config.FLASH_MODEL,
            technical_indicators=latest_indicators, market_context=market_context
        )

        # Create a list of all successful analyses
        analyses = []
        if momentum_analysis:
            analyses.append({'name': 'Momentum', 'score': momentum_analysis.get('scrybeScore', 0), 'analysis': momentum_analysis})
        if mean_reversion_analysis:
            analyses.append({'name': 'Mean-Reversion', 'score': mean_reversion_analysis.get('scrybeScore', 0), 'analysis': mean_reversion_analysis})
        if breakout_analysis:
            analyses.append({'name': 'Breakout', 'score': breakout_analysis.get('scrybeScore', 0), 'analysis': breakout_analysis})

        # Determine the winning analysis based on the highest absolute score
        if analyses:
            best_analysis_info = max(analyses, key=lambda x: abs(x['score']))
            final_analysis = best_analysis_info['analysis']
            log.info(f"CIO Decision for {ticker}: {best_analysis_info['name']} strategy selected (Score: {best_analysis_info['score']}).")
        else:
            final_analysis = None
            log.error(f"All three specialists failed to provide an analysis for {ticker}.")
        # --- END: 3-Specialist CIO Decision Logic ---
        
        # Generate DVM scores and add them to the final analysis object
        dvm_scores = analyzer.get_dvm_scores(live_financial_data, latest_indicators)
        final_analysis['dvmScores'] = dvm_scores
        
        # Add all other necessary metadata
        final_analysis.update({
            'analysis_id': str(uuid.uuid4()), 'charts': charts,
            'companyName': live_financial_data.get('rawDataSheet', {}).get('longName', ticker),
            'timestamp': datetime.now(timezone.utc), 'ticker': ticker,
            'price_at_prediction': latest_row['close'], 'prediction_date': datetime.now(timezone.utc),
            'strategy': active_strategy['name'], 'atr_at_prediction': latest_row['ATRr_14'],
            'market_regime_at_analysis': market_regime
        })
        
        log.info(f"--- ✅ SUCCESS: VST analysis complete for {ticker} ---")
        return final_analysis

    except Exception as e:
        if "429" in str(e) and "quota" in str(e).lower():
            raise e
        log.error(f"--- ❌ FAILURE: VST analysis for {ticker} failed. Error: {e} ---", exc_info=True)
        return None

def run_all_jobs():
    """
    Runs the final, multi-strategy "AI Hedge Fund" analysis pipeline for all tickers.
    """
    log.info("--- 🚀 Kicking off ALL DAILY JOBS (Hedge Fund-Grade Architecture) ---")
    
    try:
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
        
        # Prepare Live Macro Context Data for the AI
        log.info("Fetching live benchmark data for macro context...")
        benchmarks_data = data_retriever.get_benchmarks_data(period="10d")
        macro_data = {}
        if benchmarks_data is not None and not benchmarks_data.empty:
            if len(benchmarks_data) > 5:
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
            vst_analysis = None
            if ticker in config.BLUE_CHIP_TICKERS:
                active_strategy = config.BLUE_CHIP_STRATEGY
                log.info(f"Applying Blue-Chip Strategy Profile for {ticker}")
            else:
                active_strategy = config.DEFAULT_SWING_STRATEGY
            try:
                vst_analysis = run_vst_analysis_pipeline(ticker, analyzer, market_data, market_regime, macro_data=macro_data, active_strategy=active_strategy)
            except Exception as e:
                log.warning(f"Analysis pipeline failed for {ticker}. Error: {e}")
                if "429" in str(e) and "quota" in str(e).lower():
                    log.warning("Attempting API key rotation...")
                    try:
                        new_key = key_manager.rotate_key()
                        analyzer = AIAnalyzer(api_key=new_key)
                        log.info("Retrying analysis with the new key...")
                        vst_analysis = run_vst_analysis_pipeline(ticker, analyzer, market_data, market_regime, macro_data=macro_data)
                    except Exception as retry_e:
                         log.error(f"Retry failed for {ticker} after key rotation. Error: {retry_e}")
                else:
                    log.error(f"Could not recover from a non-quota error for {ticker}.")

            if not vst_analysis or not _validate_analysis_output(vst_analysis, ticker):
                log.warning(f"No valid VST analysis was generated for {ticker}.")
                if vst_analysis:
                    database_manager.save_vst_analysis(ticker, vst_analysis)
                continue

            database_manager.save_vst_analysis(ticker, vst_analysis)
            database_manager.save_live_prediction(vst_analysis)
            
            # --- START: FINAL RISK MANAGEMENT OVERLAY ---
            original_signal = vst_analysis.get('signal')
            scrybe_score = vst_analysis.get('scrybeScore', 0)
            dvm_scores = vst_analysis.get('dvmScores', {})

            final_signal = original_signal
            filter_reason = None

            # --- START: NEW "MARKET WEATHER" MASTER FILTER ---
            if market_regime == 'Bearish' and original_signal == 'BUY':
                final_signal = 'HOLD'
                filter_reason = f"MASTER FILTER VETO: BUY signal for {ticker} blocked by Bearish market regime."
            # In a strong Bullish market, we will be more skeptical of SELL signals.
            elif market_regime == 'Bullish' and original_signal == 'SELL' and abs(scrybe_score) < 80:
                    final_signal = 'HOLD'
                    filter_reason = f"MASTER FILTER VETO: SELL signal for {ticker} blocked by Bullish market regime (conviction < 80)."
            # --- END: NEW "MARKET WEATHER" MASTER FILTER ---

            # --- Existing Filters (Conviction and Quality) ---
            elif abs(scrybe_score) < 60 and original_signal != 'HOLD':
                final_signal = 'HOLD'
                filter_reason = f"Signal '{original_signal}' (Score: {scrybe_score}) was vetoed because it did not meet the conviction threshold (>60)."
            elif original_signal == 'BUY':
                durability_score = dvm_scores.get('durability', {}).get('score', 100) if dvm_scores else 100
                if durability_score < 40:
                    final_signal = 'HOLD'
                    filter_reason = f"Signal '{original_signal}' was vetoed due to a poor fundamental Quality (Durability) score."
            
            vst_analysis['signal'] = final_signal
            if filter_reason:
                log.warning(filter_reason)
                vst_analysis['analystVerdict'] = filter_reason
                vst_analysis['tradePlan'] = {"status": "Filtered by Risk Manager", "reason": filter_reason}
                database_manager.save_vst_analysis(ticker, vst_analysis)
            
            if vst_analysis.get('signal') in ['BUY', 'SELL']:
                log.info(f"PASSED FILTERS: Creating live trade for '{vst_analysis['signal']}' signal on {ticker}.")
                new_signals.append({
                    "ticker": ticker, "signal": vst_analysis.get('signal'),
                    "confidence": vst_analysis.get('confidence'), "scrybeScore": vst_analysis.get('scrybeScore')
                })
                try:
                    signal = vst_analysis['signal']
                    entry_price = vst_analysis['price_at_prediction']
                    atr = vst_analysis.get('atr_at_prediction')

                    if not atr:
                        raise ValueError("ATR not found in analysis document.")

                    # Use parameters from the selected active_strategy
                    rr_ratio = active_strategy['min_rr_ratio']
                    stop_multiplier = active_strategy['stop_loss_atr_multiplier']
                    
                    stop_loss_price = entry_price - (stop_multiplier * atr) if signal == 'BUY' else entry_price + (stop_multiplier * atr)
                    target_price = entry_price + ((stop_multiplier * atr) * rr_ratio) if signal == 'BUY' else entry_price - ((stop_multiplier * atr) * rr_ratio)

                    trade_object = {
                        "signal": signal, "strategy": active_strategy['name'],
                        "entry_price": entry_price, "entry_date": vst_analysis.get('prediction_date'),
                        "target": round(target_price, 2), "stop_loss": round(stop_loss_price, 2),
                        "risk_reward_ratio": rr_ratio,
                        "expiry_date": datetime.now(timezone.utc) + timedelta(days=active_strategy['holding_period']),
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