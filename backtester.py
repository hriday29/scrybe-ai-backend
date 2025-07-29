# backtester.py

import database_manager
import pandas as pd
from datetime import datetime, timezone
from logger_config import log
import config

def run_backtest(full_historical_data_cache: dict):
    """
    Finds all open BUY/SELL predictions and evaluates them against a provided cache of full historical data.
    """
    log.info("\n--- 📈 Starting Consolidated Backtesting Job ---")
    database_manager.init_db(purpose='scheduler')
    
    query = {"status": "open", "signal": {"$in": ["BUY", "SELL"]}}
    open_trades = list(database_manager.predictions_collection.find(query))
    log.info(f"Found {len(open_trades)} open trades to evaluate.")

    for trade in open_trades:
        try:
            ticker = trade['ticker']
            if ticker not in full_historical_data_cache:
                log.warning(f"No historical data found in cache for {ticker}, skipping trade.")
                continue
            
            process_single_trade(trade, full_historical_data_cache[ticker])

        except Exception as e:
            log.error(f"--- ❌ FAILED to process {trade.get('ticker')}. Error: {e} ---", exc_info=True)

    log.info(f"--- ✅ Consolidated Backtesting Job Finished ---")

def process_single_trade(trade: dict, historical_data: pd.DataFrame):
    """Evaluates a single BUY or SELL trade against its plan."""
    log.info(f"Processing trade for {trade['ticker']} predicted on {trade['prediction_date'].strftime('%Y-%m-%d')}...")
    signal = trade.get('signal')

    try:
        trade_plan = trade['tradePlan']
        target_price = float(trade_plan.get('target', {}).get('price', 0))
        stop_loss_price = float(trade_plan.get('stopLoss', {}).get('price', 0))
        # Use VST_STRATEGY directly since it's the only one
        strategy_name = config.VST_STRATEGY['name']
        holding_period = config.VST_STRATEGY['holding_period']
    except (ValueError, KeyError, TypeError) as e:
        log.error(f"Invalid trade plan for {trade['ticker']}: {e}. Closing trade.")
        invalid_plan_close_date = trade.get('prediction_date', datetime.now(timezone.utc))
        log_and_close_trade(trade, 0, "Trade Closed - Invalid Plan", trade['price_at_prediction'], invalid_plan_close_date)
        return

    prediction_date = trade['prediction_date'].replace(tzinfo=None)
    trading_days_since = historical_data[historical_data.index > prediction_date]

    if trading_days_since.empty:
        log.info(f"--> Trade for {trade['ticker']} still open (no new market data available yet).")
        return

    for i, day_data in enumerate(trading_days_since.itertuples()):
        day_number = i + 1
        current_day_date = day_data.Index.to_pydatetime().replace(tzinfo=timezone.utc)
        
        if signal == 'BUY':
            if day_data.low <= stop_loss_price:
                log_and_close_trade(trade, day_number, "Trade Closed - Stop-Loss Hit", stop_loss_price, current_day_date)
                return
            if day_data.high >= target_price:
                log_and_close_trade(trade, day_number, "Trade Closed - Target Hit", target_price, current_day_date)
                return
        
        elif signal == 'SELL':
            if day_data.high >= stop_loss_price:
                log_and_close_trade(trade, day_number, "Trade Closed - Stop-Loss Hit", stop_loss_price, current_day_date)
                return
            if day_data.low <= target_price:
                log_and_close_trade(trade, day_number, "Trade Closed - Target Hit", target_price, current_day_date)
                return

        if day_number >= holding_period:
            log_and_close_trade(trade, day_number, f"Trade Closed - {holding_period}-Day Time Exit", day_data.close, current_day_date)
            return
            
    log.info(f"--> Trade for {trade['ticker']} still open after {len(trading_days_since)} days.")

def log_and_close_trade(trade: dict, evaluation_day: int, event: str, event_price: float, close_date: datetime):
    """Calculates gross and net returns and saves the final result."""
    price_at_prediction = trade['price_at_prediction']
    signal = trade.get('signal')
    strategy_name = trade.get('strategy', 'unknown')
    
    # --- REALISTIC COST CALCULATION ---
    
    # 1. Calculate Gross Return
    if signal == 'SELL':
        gross_return_pct = ((price_at_prediction - event_price) / price_at_prediction) * 100
    else: # BUY
        gross_return_pct = ((event_price - price_at_prediction) / price_at_prediction) * 100

    # 2. Calculate Total Transaction Costs
    costs = config.BACKTEST_CONFIG
    # Brokerage for entry and exit
    total_brokerage_pct = costs['brokerage_pct'] * 2
    # Slippage for entry and exit
    total_slippage_pct = costs['slippage_pct'] * 2
    # STT is only on the sell-side for delivery
    stt_cost_pct = costs['stt_pct']
    
    total_costs_pct = total_brokerage_pct + total_slippage_pct + stt_cost_pct
    
    # 3. Calculate Net Return
    net_return_pct = gross_return_pct - total_costs_pct

    performance_doc = {
        "prediction_id": trade['_id'], 
        "ticker": trade['ticker'], 
        "strategy": strategy_name,
        "signal": signal, 
        "status": "Closed", 
        "open_date": trade['prediction_date'],
        "close_date": close_date,
        "evaluation_days": evaluation_day,
        "closing_reason": event, 
        "gross_return_pct": round(gross_return_pct, 2),
        "net_return_pct": round(net_return_pct, 2) # Save the new net return
    }
    
    database_manager.performance_collection.insert_one(performance_doc)
    database_manager.predictions_collection.update_one({"_id": trade['_id']}, {"$set": {"status": "Closed"}})
    log.info(f"--> ✅ PERFORMANCE LOGGED: {trade['ticker']} - Gross: {gross_return_pct:.2f}%, Net: {net_return_pct:.2f}%")