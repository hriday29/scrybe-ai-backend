import pandas as pd
import time
import config
from logger_config import log
import database_manager
import json
import os
import data_retriever
import performance_analyzer
import technical_analyzer
import pandas_ta as ta
from ai_analyzer import AIAnalyzer
from datetime import datetime
import uuid
from utils import APIKeyManager
import argparse

# --- Configuration ---
STATE_FILE = 'simulation_state.json'
# These are now defined in the main block for manual runs, not as global constants.

def save_state(next_day_to_run):
    """Saves the next day to be processed to a state file."""
    with open(STATE_FILE, 'w') as f:
        json.dump({'next_start_date': next_day_to_run.strftime('%Y-%m-%d')}, f)

def load_state():
    """Loads the last saved state, returns a start date string."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            log.warning(f"Resuming from saved state. Next day to run: {state['next_start_date']}")
            return state['next_start_date']
    return None

def run_historical_test(batch_id: str, start_date: str, end_date: str, stocks_to_test: list, is_fresh_run: bool = False, generate_charts: bool = False):
    """
    Generates historical AI predictions for a given date range and stock list.
    """
    log.info(f"### STARTING HISTORICAL PREDICTION GENERATION FOR BATCH: {batch_id} ###")
    log.info(f"Period: {start_date} to {end_date}")

    database_manager.init_db(purpose='scheduler')
    if is_fresh_run:
        log.warning("FRESH RUN ENABLED: Deleting all previous backtest predictions and performance data.")
        database_manager.clear_scheduler_data()
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)

    try:
        key_manager = APIKeyManager(api_keys=config.GEMINI_API_KEY_POOL)
        analyzer = AIAnalyzer(api_key=key_manager.get_key())
        
        log.info(f"Pre-loading historical data for {len(stocks_to_test)} stocks...")
        full_historical_data_cache = {}
        for ticker in stocks_to_test:
            data = data_retriever.get_historical_stock_data(ticker, end_date=end_date)
            if data is not None and len(data) > 100:
                full_historical_data_cache[ticker] = data
            else:
                log.warning(f"Skipping {ticker} due to insufficient historical data.")
        
        log.info("Pre-loading historical data for NIFTY 50 index...")
        nifty_data_cache = data_retriever.get_historical_stock_data("^NSEI", end_date=end_date)
        
        log.info("Pre-loading historical data for benchmarks...")
        benchmarks_data_cache = data_retriever.get_benchmarks_data(end_date=end_date)
        log.info("✅ All data pre-loading complete.")

    except Exception as e:
        log.fatal(f"Failed during pre-run initialization. Error: {e}")
        return

    resumed_start_date = load_state()
    if resumed_start_date:
        start_date = resumed_start_date
    
    simulation_days = pd.bdate_range(start=start_date, end=end_date)
    
    if not simulation_days.empty:
        log.info(f"\n--- Starting simulation for {len(simulation_days)} trading days ---")
        for i, current_day in enumerate(simulation_days):
            day_str = current_day.strftime('%Y-%m-%d')
            log.info(f"\n--- Simulating Day {i+1}/{len(simulation_days)}: {day_str} ---")

            try:
                # Calculate the true market regime for the current simulated day
                nifty_slice = nifty_data_cache.loc[:day_str] if nifty_data_cache is not None else None
                current_regime = data_retriever.calculate_regime_from_data(nifty_slice)
                log.info(f"Regime for {day_str} calculated as: {current_regime}")
                
                market_context_for_day = {"CURRENT_MARKET_REGIME": current_regime, "sector_performance_today": {}}

                for ticker in stocks_to_test:
                    if ticker not in full_historical_data_cache: continue
                    data_slice = full_historical_data_cache[ticker].loc[:day_str]
                    
                    if len(data_slice) < 100: continue

                    log.info(f"--- Analyzing Ticker: {ticker} for {day_str} ---")
                    
                    max_retries = len(key_manager.api_keys)
                    for attempt in range(max_retries):
                        try:
                            live_financial_data = {"curatedData": {}, "rawDataSheet": {"symbol": ticker, "longName": ticker}}
                            
                            data_slice_copy = data_slice.copy()
                            data_slice_copy.ta.bbands(length=20, append=True); data_slice_copy.ta.rsi(length=14, append=True)
                            data_slice_copy.ta.macd(fast=12, slow=26, signal=9, append=True); data_slice_copy.ta.adx(length=14, append=True)
                            data_slice_copy.ta.atr(length=14, append=True); data_slice_copy.dropna(inplace=True)
                            
                            if data_slice_copy.empty: continue

                            latest_row = data_slice_copy.iloc[-1]
                            volume_ma_20 = data_slice_copy['volume'].rolling(window=20).mean().iloc[-1]
                            is_volume_high = latest_row['volume'] > volume_ma_20 * config.VOLUME_SURGE_THRESHOLD
                            
                            latest_indicators = { "ADX": f"{latest_row['ADX_14']:.2f}", "RSI": f"{latest_row['RSI_14']:.2f}", "MACD": f"{latest_row['MACD_12_26_9']:.2f}", "Bollinger Band Width Percent": f"{(latest_row['BBU_20_2.0'] - latest_row['BBL_20_2.0']) / latest_row['BBM_20_2.0'] * 100:.2f}", "Volume Surge": "Yes" if is_volume_high else "No" }

                            charts_for_ai = {}
                            if generate_charts:
                                log.info(f"Generating charts for {ticker} (High-Fidelity Run)...")
                                charts_for_ai = technical_analyzer.generate_focused_charts(data_slice, ticker)

                            analysis_result = analyzer.get_stock_analysis(
                                live_financial_data=live_financial_data, latest_atr=latest_row['ATRr_14'], model_name=config.PRO_MODEL,
                                charts=charts_for_ai, trading_horizon_text=config.VST_STRATEGY['horizon_text'],
                                technical_indicators=latest_indicators, min_rr_ratio=config.VST_STRATEGY['min_rr_ratio'],
                                market_context=market_context_for_day, options_data={}
                            )

                            if analysis_result:
                                log.info(f"Generating DVM scores for {ticker}...")
                                dvm_scores = analyzer.get_dvm_scores(live_financial_data, latest_indicators)

                                benchmarks_slice = benchmarks_data_cache.loc[:day_str] if benchmarks_data_cache is not None else None
                                correlations = performance_analyzer.calculate_correlations(data_slice, benchmarks_slice)

                                prediction_doc = analysis_result.copy()
                                prediction_doc.update({
                                    'analysis_id': str(uuid.uuid4()),
                                    'ticker': ticker,
                                    'prediction_date': current_day.to_pydatetime(),
                                    'price_at_prediction': latest_row['close'],
                                    'status': 'open',
                                    'strategy': config.VST_STRATEGY['name'],
                                    'dvmScores': dvm_scores,
                                    'correlations': correlations,
                                    'atr_at_prediction': latest_row['ATRr_14']
                                })
                                database_manager.save_prediction_for_backtesting(prediction_doc, batch_id)
                            
                            break

                        except Exception as e:
                            log.warning(f"AI analysis failed for {ticker} on attempt {attempt + 1}. Error: {e}")
                            if attempt < max_retries - 1:
                                new_key = key_manager.rotate_key()
                                analyzer = AIAnalyzer(api_key=new_key)
                                log.info("Retrying AI call with the new key...")
                                time.sleep(5)
                            else:
                                log.error(f"All API keys failed for {ticker}. Skipping for this day.")
                    
                    time.sleep(5)

                if i + 1 < len(simulation_days):
                    save_state(simulation_days[i+1])

            except Exception as e:
                log.error(f"A critical error occurred on {day_str}. Error: {e}", exc_info=True)
                return

    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
    
    log.info("\n--- ✅ Full Historical Prediction Generation Finished! ---")
    log.info("--- You can now run the backtester.py script. ---")
    database_manager.close_db_connection()

if __name__ == "__main__":
    # This block now reads arguments from the command line (e.g., from the GitHub Action)
    parser = argparse.ArgumentParser(description="Run a historical backtest batch.")
    parser.add_argument('--batch_id', required=True, help='Unique ID for the backtest batch.')
    parser.add_argument('--start_date', required=True, help='Start date in YYYY-MM-DD format.')
    parser.add_argument('--end_date', required=True, help='End date in YYYY-MM-DD format.')
    parser.add_argument('--stocks', required=True, help='Comma-separated string of stock tickers.')
    parser.add_argument('--fresh_run', type=lambda x: (str(x).lower() == 'true'), help='Boolean flag to clear previous data.')
    parser.add_argument('--generate_charts', type=lambda x: (str(x).lower() == 'true'), help='Boolean flag to generate charts.')
    
    args = parser.parse_args()
    
    # Convert the comma-separated string of stocks into a list
    stocks_list = [stock.strip() for stock in args.stocks.split(',')]

    log.info(f"--- Starting Backtest Run from Command Line for Batch: {args.batch_id} ---")
    
    run_historical_test(
        batch_id=args.batch_id,
        start_date=args.start_date,
        end_date=args.end_date,
        stocks_to_test=stocks_list,
        is_fresh_run=args.fresh_run,
        generate_charts=args.generate_charts
    )