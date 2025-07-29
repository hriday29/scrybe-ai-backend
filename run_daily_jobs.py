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

            # Ensure expiry_date is a timezone-aware datetime object for comparison
            if isinstance(trade.get('expiry_date'), str):
                 trade['expiry_date'] = datetime.fromisoformat(trade['expiry_date'])

            if datetime.now(timezone.utc) >= trade['expiry_date']:
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

def run_vst_analysis_pipeline(ticker: str, analyzer: AIAnalyzer, market_data: dict, market_regime: str):
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
            market_context=market_context, options_data=options_data
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
            'market_regime_at_analysis': market_regime
        })
        
        log.info(f"--- ✅ SUCCESS: VST analysis complete for {ticker} ---")
        return analysis_result
    except Exception as e:
        log.error(f"--- ❌ FAILURE: VST analysis for {ticker} failed. Error: {e} ---", exc_info=True)
        return None

def run_all_jobs():
    """Runs the VST-only analysis pipeline with Market Regime and Email Notifications."""
    log.info("--- 🚀 Kicking off ALL DAILY JOBS (VST-Only Mode) ---")
    
    closed_trades = manage_open_trades()

    log.info("--- 🔍 Starting New Analysis Generation ---")
    try:
        analyzer = AIAnalyzer(api_key=config.GEMINI_API_KEY)
    except ValueError as e: return log.fatal(f"Failed to initialize AI Analyzer: {e}. Aborting.")

    market_data = data_retriever.get_nifty50_performance()
    if not market_data: return log.error("Could not fetch market performance data. Aborting.")

    market_regime = data_retriever.get_market_regime()
    
    new_signals = [] 

    for ticker in config.FOCUS_STOCKS:
        vst_analysis = run_vst_analysis_pipeline(ticker, analyzer, market_data, market_regime)
        
        if not vst_analysis or not _validate_analysis_output(vst_analysis, ticker) or not _validate_trade_plan(vst_analysis):
            log.warning(f"No valid VST analysis was generated for {ticker}.")
            if vst_analysis:
                database_manager.save_vst_analysis(ticker, vst_analysis)
            continue

        database_manager.init_db(purpose='analysis')
        database_manager.save_vst_analysis(ticker, vst_analysis)
        database_manager.save_live_prediction(vst_analysis)

        if vst_analysis.get('signal') in ['BUY', 'SELL']:
            new_signals.append({
                "ticker": ticker,
                "signal": vst_analysis.get('signal'),
                "confidence": vst_analysis.get('confidence')
            })
            try:
                trade_object = {
                    "signal": vst_analysis.get('signal'), "strategy": vst_analysis.get('strategy'),
                    "entry_price": vst_analysis.get('price_at_prediction'),
                    "entry_date": vst_analysis.get('prediction_date'),
                    "target": float(vst_analysis['tradePlan']['target']['price']),
                    "stop_loss": float(vst_analysis['tradePlan']['stopLoss']['price']),
                    "expiry_date": datetime.now(timezone.utc) + timedelta(days=config.TRADE_EXPIRY_DAYS),
                    "confidence": vst_analysis.get('confidence')
                }
                database_manager.set_active_trade(ticker, trade_object)
            except (KeyError, ValueError, TypeError) as e:
                log.error(f"Could not create trade object for {ticker}: {e}")
                database_manager.set_active_trade(ticker, None)
        else:
            log.info(f"Signal for {ticker} is 'HOLD'. No active trade will be set.")
            database_manager.set_active_trade(ticker, None)
        
        time.sleep(5)

    log.info("--- ✅ All daily jobs finished ---")
    generate_performance_report()

    send_daily_briefing(new_signals, closed_trades)


if __name__ == "__main__":
    run_all_jobs()