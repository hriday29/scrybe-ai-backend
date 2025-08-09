# backtester.py

import database_manager
import pandas as pd
from datetime import datetime, timezone
from logger_config import log
import config
import argparse
import data_retriever # --- ADDITION: We need this for fetching data ---

def preload_historical_data_for_batch(batch_id: str) -> dict:
    """
    Finds all unique tickers for a given batch_id, pre-loads their full
    historical data, and returns it in a cache dictionary.
    """
    log.info(f"Pre-loading all historical data for tickers in batch: {batch_id}...")
    
    # Find all unique tickers that have a prediction in this batch
    tickers_in_batch = database_manager.predictions_collection.distinct("ticker", {"batch_id": batch_id})
    
    if not tickers_in_batch:
        log.warning(f"No tickers found for batch '{batch_id}'.")
        return {}

    log.info(f"Found {len(tickers_in_batch)} unique tickers to load data for.")
    
    data_cache = {}
    for ticker in tickers_in_batch:
        # We fetch the full history, as the backtester will slice it as needed.
        # We don't need an end_date here as we want all data up to the present.
        data = data_retriever.get_historical_stock_data(ticker)
        if data is not None and not data.empty:
            data_cache[ticker] = data
        else:
            log.warning(f"Could not load historical data for {ticker}.")
            
    log.info("✅ Data pre-loading complete.")
    return data_cache

def run_backtest(batch_id: str, full_historical_data_cache: dict):
    """
    Finds all open BUY/SELL predictions and evaluates them against a provided cache of full historical data.
    """
    log.info(f"\n--- 📈 Starting Consolidated Backtesting Job for Batch: {batch_id} ---")
    database_manager.init_db(purpose='scheduler')
    
    query = {"status": "open", "signal": {"$in": ["BUY", "SELL"]}, "batch_id": batch_id}
    open_trades = list(database_manager.predictions_collection.find(query))
    log.info(f"Found {len(open_trades)} open trades to evaluate in this batch.")

    for trade in open_trades:
        try:
            ticker = trade['ticker']
            if ticker not in full_historical_data_cache:
                log.warning(f"No historical data found in cache for {ticker}, skipping trade.")
                continue
            
            process_single_trade(trade, full_historical_data_cache[ticker])

        except Exception as e:
            log.error(f"--- ❌ FAILED to process trade {trade.get('_id')} for {trade.get('ticker')}. Error: {e} ---", exc_info=True)

    log.info(f"--- ✅ Consolidated Backtesting Job Finished for Batch: {batch_id} ---")

def process_single_trade(trade: dict, historical_data: pd.DataFrame):
    """Evaluates a single BUY or SELL trade against its plan with robust validation."""
    log.info(f"Processing trade for {trade['ticker']} predicted on {trade['prediction_date'].strftime('%Y-%m-%d')}...")
    signal = trade.get('signal')
    price_at_prediction = trade['price_at_prediction']

    try:
        trade_plan = trade['tradePlan']
        target_price = float(trade_plan.get('target', {}).get('price', 0))
        stop_loss_price = float(trade_plan.get('stopLoss', {}).get('price', 0))
        holding_period = config.VST_STRATEGY['holding_period']

        # --- NEW VALIDATION LOGIC ---
        # Define a realistic threshold (e.g., a 50% move in 5 days is highly unlikely)
        REALISTIC_THRESHOLD_PCT = 50.0 
        
        target_pct_change = abs(target_price - price_at_prediction) / price_at_prediction * 100
        stop_loss_pct_change = abs(stop_loss_price - price_at_prediction) / price_at_prediction * 100

        if target_pct_change > REALISTIC_THRESHOLD_PCT or stop_loss_pct_change > REALISTIC_THRESHOLD_PCT:
            log.error(f"Invalid trade plan for {trade['ticker']}: Target or Stop-Loss is unrealistic (> {REALISTIC_THRESHOLD_PCT}%).")
            invalid_plan_close_date = trade.get('prediction_date', datetime.now(timezone.utc))
            log_and_close_trade(trade, 0, "Trade Closed - Unrealistic Plan", price_at_prediction, invalid_plan_close_date)
            return
        # --- END VALIDATION LOGIC ---

    except (ValueError, KeyError, TypeError) as e:
        log.error(f"Invalid trade plan for {trade['ticker']}: {e}. Closing trade.")
        invalid_plan_close_date = trade.get('prediction_date', datetime.now(timezone.utc))
        log_and_close_trade(trade, 0, "Trade Closed - Invalid Plan", price_at_prediction, invalid_plan_close_date)
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
    batch_id = trade.get('batch_id')
    
    # If the plan was invalid or unrealistic, the return is neutralized to zero.
    if "Invalid Plan" in event or "Unrealistic Plan" in event:
        gross_return_pct = 0.0
    else:
        # Otherwise, calculate the return based on the entry and exit price.
        if signal == 'SELL':
            gross_return_pct = ((price_at_prediction - event_price) / price_at_prediction) * 100
        else: # BUY
            gross_return_pct = ((event_price - price_at_prediction) / price_at_prediction) * 100

    # Calculate net return by subtracting transaction costs
    costs = config.BACKTEST_CONFIG
    total_costs_pct = (costs['brokerage_pct'] * 2) + (costs['slippage_pct'] * 2) + costs['stt_pct']
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
        "net_return_pct": round(net_return_pct, 2), 
        "batch_id": batch_id
    }
    
    database_manager.performance_collection.insert_one(performance_doc)
    database_manager.predictions_collection.update_one({"_id": trade['_id']}, {"$set": {"status": "Closed"}})
    log.info(f"--> ✅ PERFORMANCE LOGGED: {trade['ticker']} - Net: {net_return_pct:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the trade simulator for a specific backtest batch.")
    parser.add_argument('--batch_id', required=True, help='The unique ID of the backtest batch to process.')
    args = parser.parse_args()

    log.info(f"--- Starting Trade Simulation for Batch: {args.batch_id} ---")
    
    # --- MODIFICATION: Pre-load the necessary historical data ---
    database_manager.init_db(purpose='scheduler')
    full_data_cache = preload_historical_data_for_batch(args.batch_id)
    
    run_backtest(batch_id=args.batch_id, full_historical_data_cache=full_data_cache)

    database_manager.close_db_connection()
    log.info(f"--- ✅ Simulation Finished for Batch: {args.batch_id} ---")