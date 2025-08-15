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
import pandas_ta as ta

def manage_open_trades():
    """
    Finds all open trades, checks their status, and closes them if conditions are met.
    Implements live trailing stop-loss logic to match the backtester.
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
                log.warning(f"Could not get live price for {ticker}. Skipping trade management.")
                continue
            latest_price = live_data['curatedData']['currentPrice']

            # --- START: TRAILING STOP LOGIC ---
            if trade['strategy'] == 'BlueChip':
                active_strategy = config.BLUE_CHIP_STRATEGY
            else:
                active_strategy = config.DEFAULT_SWING_STRATEGY

            if active_strategy.get('use_trailing_stop', False):
                historical_data = data_retriever.get_historical_stock_data(ticker)
                if historical_data is not None and len(historical_data) > 14:
                    historical_data.ta.atr(length=14, append=True)
                    latest_atr = historical_data['ATRr_14'].iloc[-1]
                    trailing_stop_atr_multiplier = active_strategy.get('trailing_stop_pct', 1.5)
                    
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
            if isinstance(expiry_date, str):
                expiry_date = datetime.fromisoformat(expiry_date)
            if expiry_date and expiry_date.tzinfo is None:
                expiry_date = expiry_date.replace(tzinfo=timezone.utc)
            
            if expiry_date and datetime.now(timezone.utc) >= expiry_date:
                close_reason = "Trade Closed - Expired"
                close_price = latest_price
            elif trade['signal'] == 'BUY':
                if latest_price >= trade['target']:
                    close_reason, close_price = "Trade Closed - Target Hit", trade['target']
                elif latest_price <= trade['stop_loss']: # This now checks against the potentially updated stop_loss
                    close_reason, close_price = "Trade Closed - Stop-Loss Hit", trade['stop_loss']
            elif trade['signal'] == 'SELL':
                if latest_price <= trade['target']:
                    close_reason, close_price = "Trade Closed - Target Hit", trade['target']
                elif latest_price >= trade['stop_loss']: # This now checks against the potentially updated stop_loss
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
    The SPRINT2_V2_FIX version of the analysis pipeline.
    """
    try:
        log.info(f"\n--- STARTING VST ANALYSIS FOR: {ticker} (Market Regime: {market_regime}) ---")
        
        # This function should NOT initialize the DB, it's handled by the main job.
        existing_trade_doc = database_manager.analysis_results_collection.find_one({"ticker": ticker, "active_trade": {"$ne": None}})
        if existing_trade_doc:
            log.warning(f"Skipping new analysis for {ticker} as it already has an active trade.")
            return None, None

        historical_data = data_retriever.get_historical_stock_data(ticker)
        if historical_data is None or len(historical_data) < 50:
            raise ValueError("Not enough historical data.")

        live_financial_data = data_retriever.get_live_financial_data(ticker)
        if not live_financial_data:
            raise ValueError("Could not get live financial data.")
        
        options_data = data_retriever.get_options_data(ticker)
        
        historical_data.ta.bbands(length=20, append=True)
        historical_data.ta.rsi(length=14, append=True)
        historical_data.ta.macd(fast=12, slow=26, signal=9, append=True)
        historical_data.ta.adx(length=14, append=True)
        historical_data.ta.atr(length=14, append=True)
        historical_data.dropna(inplace=True)
        
        charts = technical_analyzer.generate_focused_charts(historical_data, ticker)
        latest_row = historical_data.iloc[-1]
        volume_ma_20 = historical_data['volume'].rolling(window=20).mean().iloc[-1]
        is_volume_high = latest_row['volume'] > volume_ma_20 * config.VOLUME_SURGE_THRESHOLD
        
        latest_indicators = {
            "ADX": f"{latest_row['ADX_14']:.2f}",
            "RSI": f"{latest_row['RSI_14']:.2f}",
            "MACD": f"{latest_row['MACD_12_26_9']:.2f}",
            "Bollinger Band Width Percent": f"{(latest_row['BBU_20_2.0'] - latest_row['BBL_20_2.0']) / latest_row['BBM_20_2.0'] * 100:.2f}",
            "Volume Surge": "Yes" if is_volume_high else "No"
        }

        stock_5d_change = ((latest_row['close'] - historical_data['close'].iloc[-6]) / historical_data['close'].iloc[-6]) * 100 if len(historical_data) > 5 else 0
        
        market_context = {
            "stock_sector": market_data.get("stock_performance", {}).get(ticker, {}).get('sector', 'Unknown'),
            "sector_performance_today": market_data.get("sector_performance", {}),
            "CURRENT_MARKET_REGIME": market_regime,
            "Stock_5D_Change": f"{stock_5d_change:.2f}%"
        }
        
        momentum_analysis = analyzer.get_momentum_analysis(
            live_financial_data=live_financial_data,
            latest_atr=latest_row['ATRr_14'],
            model_name=config.PRO_MODEL,
            charts=charts,
            trading_horizon_text=active_strategy['horizon_text'],
            technical_indicators=latest_indicators,
            min_rr_ratio=active_strategy['min_rr_ratio'],
            market_context=market_context,
            options_data=options_data,
            macro_data=macro_data
        )

        mean_reversion_analysis = analyzer.get_mean_reversion_analysis(
            live_financial_data=live_financial_data,
            model_name=config.FLASH_MODEL,
            technical_indicators=latest_indicators,
            market_context=market_context
        )
        
        final_analysis = None
        if momentum_analysis:
            momentum_score = momentum_analysis.get('scrybeScore', 0)
            reversion_score = mean_reversion_analysis.get('scrybeScore', 0) if mean_reversion_analysis else 0
            if abs(momentum_score) >= abs(reversion_score):
                final_analysis = momentum_analysis
            else:
                final_analysis = mean_reversion_analysis
        elif mean_reversion_analysis:
            final_analysis = mean_reversion_analysis

        if not final_analysis:
            raise ValueError("Both AI specialists failed to return an analysis.")
        
        dvm_scores = analyzer.get_dvm_scores(live_financial_data, latest_indicators)
        final_analysis['dvmScores'] = dvm_scores
        
        final_analysis.update({
            'analysis_id': str(uuid.uuid4()),
            'charts': charts,
            'companyName': live_financial_data.get('rawDataSheet', {}).get('longName', ticker),
            'timestamp': datetime.now(timezone.utc),
            'ticker': ticker,
            'price_at_prediction': latest_row['close'],
            'prediction_date': datetime.now(timezone.utc),
            'strategy': active_strategy['name'],
            'atr_at_prediction': latest_row['ATRr_14'],
            'market_regime_at_analysis': market_regime
        })
        
        log.info(f"--- ✅ SUCCESS: VST analysis complete for {ticker} ---")
        return final_analysis, None  # Return None for best_analysis_info to match signature

    except Exception as e:
        if "429" in str(e) and "quota" in str(e).lower():
            raise e
        log.error(f"--- ❌ FAILURE: VST analysis for {ticker} failed. Error: {e} ---", exc_info=True)
        return None, None


def run_all_jobs():
    """
    The SPRINT2_V2_FIX version of the main job runner.
    """
    log.info("--- 🚀 Starting all daily jobs ---")
    try:
        database_manager.init_db(purpose='analysis')

        key_manager = APIKeyManager(api_keys=config.GEMINI_API_KEY_POOL)
        analyzer = AIAnalyzer(api_key=key_manager.get_key())

        closed_trades = manage_open_trades()

        market_data = data_retriever.get_nifty50_performance()
        if not market_data:
            log.error("Could not fetch market performance data. Aborting.")
            return

        market_regime = data_retriever.get_market_regime()
        macro_data = {}  # Fill if needed with benchmark changes, similar to your macro context logic.

        nifty50_tickers = config.NIFTY_50_TICKERS

        for ticker in nifty50_tickers:
            if ticker in config.BLUE_CHIP_TICKERS:
                active_strategy = config.BLUE_CHIP_STRATEGY
            else:
                active_strategy = config.DEFAULT_SWING_STRATEGY

            log.info(f"Applying Profile ('{active_strategy['name']}') for {ticker}")

            vst_analysis = None
            try:
                vst_analysis, _ = run_vst_analysis_pipeline(
                    ticker, analyzer, market_data, market_regime, macro_data=macro_data, active_strategy=active_strategy
                )
            except Exception as e:
                if "429" in str(e) and "quota" in str(e).lower():
                    log.warning("Quota reached. Rotating API key...")
                    new_key = key_manager.rotate_key()
                    analyzer = AIAnalyzer(api_key=new_key)
                    try:
                        vst_analysis, _ = run_vst_analysis_pipeline(
                            ticker, analyzer, market_data, market_regime, macro_data=macro_data, active_strategy=active_strategy
                        )
                    except Exception as retry_e:
                        log.error(f"Retry after key rotation failed for {ticker}: {retry_e}", exc_info=True)
                        continue
                else:
                    log.error(f"Analysis failed for {ticker}: {e}", exc_info=True)
                    continue

            if not vst_analysis or not _validate_analysis_output(vst_analysis, ticker):
                log.warning(f"No valid analysis for {ticker}, skipping.")
                continue

            vst_analysis['strategy'] = active_strategy['name']
            database_manager.save_vst_analysis(ticker, vst_analysis)
            database_manager.save_live_prediction(vst_analysis)

            original_signal = vst_analysis.get('signal')
            scrybe_score = vst_analysis.get('scrybeScore', 0)
            dvm_scores = vst_analysis.get('dvmScores', {})

            is_regime_ok = (
                (original_signal == 'BUY' and market_regime == 'Bullish') or
                (original_signal == 'SELL' and market_regime == 'Bearish') or
                (original_signal == 'HOLD')
            )
            is_conviction_ok = abs(scrybe_score) >= 60 or original_signal == 'HOLD'
            is_quality_ok = True
            if original_signal == 'BUY' and dvm_scores:
                if dvm_scores.get('durability', {}).get('score', 100) < 40:
                    is_quality_ok = False

            final_signal = original_signal
            filter_reason = None
            if not is_regime_ok:
                final_signal = 'HOLD'
                filter_reason = "Vetoed by Market Regime"
            elif not is_conviction_ok:
                final_signal = 'HOLD'
                filter_reason = "Vetoed by Low Conviction"
            elif not is_quality_ok:
                final_signal = 'HOLD'
                filter_reason = "Vetoed by Poor Quality"

            vst_analysis['signal'] = final_signal
            if filter_reason:
                vst_analysis['analystVerdict'] = filter_reason
                vst_analysis['tradePlan'] = {"status": "Filtered", "reason": filter_reason}
                database_manager.save_vst_analysis(ticker, vst_analysis)

            if vst_analysis.get('signal') in ['BUY', 'SELL']:
                try:
                    signal = vst_analysis['signal']
                    entry_price = vst_analysis['price_at_prediction']
                    atr = vst_analysis.get('atr_at_prediction')
                    rr_ratio = active_strategy['min_rr_ratio']
                    stop_mult = active_strategy['stop_loss_atr_multiplier']

                    if signal == 'BUY':
                        stop_loss_price = entry_price - (stop_mult * atr)
                        target_price = entry_price + ((stop_mult * atr) * rr_ratio)
                    else:
                        stop_loss_price = entry_price + (stop_mult * atr)
                        target_price = entry_price - ((stop_mult * atr) * rr_ratio)

                    trade_object = {
                        "signal": signal,
                        "strategy": active_strategy['name'],
                        "entry_price": entry_price,
                        "entry_date": vst_analysis.get('prediction_date'),
                        "target": round(target_price, 2),
                        "stop_loss": round(stop_loss_price, 2),
                        "risk_reward_ratio": rr_ratio,
                        "expiry_date": datetime.now(timezone.utc) + timedelta(days=active_strategy['holding_period']),
                        "confidence": vst_analysis.get('confidence')
                    }
                    database_manager.set_active_trade(ticker, trade_object)
                except Exception as e:
                    log.error(f"Failed to create trade object for {ticker}: {e}", exc_info=True)
                    database_manager.set_active_trade(ticker, None)
            else:
                database_manager.set_active_trade(ticker, None)

            time.sleep(10)

        log.info("--- ✅ All daily jobs completed ---")
        generate_performance_report()
        send_daily_briefing([], closed_trades)

    finally:
        database_manager.close_db_connection()
    log.info("--- 🔒 Database connection closed ---")

if __name__ == "__main__":
    run_all_jobs()