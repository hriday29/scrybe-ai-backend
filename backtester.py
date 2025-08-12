# backtester.py

import database_manager
import pandas as pd
from datetime import datetime, timezone
from logger_config import log
import config
import argparse
import data_retriever # --- ADDITION: We need this for fetching data ---

def preload_historical_data_for_batch(batch_id: str, end_date: str) -> dict:
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
        data = data_retriever.get_historical_stock_data(ticker, end_date=end_date)
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
        trade_plan = trade.get('tradePlan', {})
        
        # Directly attempt to convert price strings to floats.
        # The except block will catch any non-numeric values like "N/A".
        target_price = float(trade_plan.get('target', {}).get('price'))
        stop_loss_price = float(trade_plan.get('stopLoss', {}).get('price'))
        
        atr_at_prediction = float(trade.get('atr_at_prediction', 0))
        holding_period = config.VST_STRATEGY['holding_period']

        if not all([target_price, stop_loss_price, atr_at_prediction]):
             raise ValueError("Essential trade plan values (target, stop-loss, ATR) are zero after conversion.")

        # Using the same dynamic validation
        ATR_REALISM_MULTIPLIER = 6.0
        target_distance = abs(target_price - price_at_prediction)
        max_realistic_move = ATR_REALISM_MULTIPLIER * atr_at_prediction
        
        if target_distance > max_realistic_move:
            log.error(f"Invalid trade plan for {trade['ticker']}: Target move ({target_distance:.2f}) is unrealistic (> {ATR_REALISM_MULTIPLIER} * ATR ({max_realistic_move:.2f})).")
            log_and_close_trade(trade, 0, "Trade Closed - Unrealistic Plan", price_at_prediction, trade['prediction_date'])
            return

    except (ValueError, KeyError, TypeError) as e:
        log.error(f"Invalid trade plan for {trade['ticker']}: {e}. This may be a 'HOLD' signal or have non-numeric prices. Closing trade.")
        log_and_close_trade(trade, 0, "Trade Closed - Invalid Plan", price_at_prediction, trade['prediction_date'])
        return
    
    use_trailing_stop = config.VST_STRATEGY.get('use_trailing_stop', False)
    trailing_stop_atr_multiplier = config.VST_STRATEGY.get('trailing_stop_pct', 1.5)

    # Initialize the current stop-loss price
    current_stop_loss = stop_loss_price
    
    prediction_date = trade['prediction_date'].replace(tzinfo=None)
    trading_days_since = historical_data[historical_data.index > prediction_date]

    if trading_days_since.empty:
        log.info(f"--> Trade for {trade['ticker']} still open (no new market data available yet).")
        return

    for i, day_data in enumerate(trading_days_since.itertuples()):
        day_number = i + 1
        current_day_date = day_data.Index.to_pydatetime().replace(tzinfo=timezone.utc)

        if signal == 'BUY':
            # Check if the price hit our current stop-loss
            if day_data.low <= current_stop_loss:
                reason = "Trade Closed - Trailing Stop Hit" if use_trailing_stop and current_stop_loss > stop_loss_price else "Trade Closed - Stop-Loss Hit"
                log_and_close_trade(trade, day_number, reason, current_stop_loss, current_day_date)
                return
            # Check for target hit
            if day_data.high >= target_price:
                log_and_close_trade(trade, day_number, "Trade Closed - Target Hit", target_price, current_day_date)
                return

            # Trailing Stop Logic for BUY trades
            if use_trailing_stop:
                # Calculate a potential new stop-loss based on today's high
                new_potential_stop = day_data.high - (atr_at_prediction * trailing_stop_atr_multiplier)
                # If this new stop is higher than our current one, "trail" it up
                if new_potential_stop > current_stop_loss:
                    current_stop_loss = new_potential_stop
                    log.info(f"Trailing stop for {trade['ticker']} moved up to {current_stop_loss:.2f}")

        elif signal == 'SELL':
            # Check if the price hit our current stop-loss
            if day_data.high >= current_stop_loss:
                reason = "Trade Closed - Trailing Stop Hit" if use_trailing_stop and current_stop_loss < stop_loss_price else "Trade Closed - Stop-Loss Hit"
                log_and_close_trade(trade, day_number, reason, current_stop_loss, current_day_date)
                return
            # Check for target hit
            if day_data.low <= target_price:
                log_and_close_trade(trade, day_number, "Trade Closed - Target Hit", target_price, current_day_date)
                return

            # Trailing Stop Logic for SELL trades
            if use_trailing_stop:
                # Calculate a potential new stop-loss based on today's low
                new_potential_stop = day_data.low + (atr_at_prediction * trailing_stop_atr_multiplier)
                # If this new stop is lower than our current one, "trail" it down
                if new_potential_stop < current_stop_loss:
                    current_stop_loss = new_potential_stop
                    log.info(f"Trailing stop for {trade['ticker']} moved down to {current_stop_loss:.2f}")

        # Time-based exit remains the final check
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
    parser.add_argument('--end_date', required=True, help='The end date (YYYY-MM-DD) of the simulation period to prevent lookahead bias.')
    args = parser.parse_args()

    log.info(f"--- Starting Trade Simulation for Batch: {args.batch_id} up to {args.end_date} ---")
    
    database_manager.init_db(purpose='scheduler')
    # Pass the end_date to the data loading function
    full_data_cache = preload_historical_data_for_batch(batch_id=args.batch_id, end_date=args.end_date)
    
    run_backtest(batch_id=args.batch_id, full_historical_data_cache=full_data_cache)