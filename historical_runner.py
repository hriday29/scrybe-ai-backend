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
    Generates historical AI predictions for a given date range and stock list using the final multi-strategy architecture.
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
                nifty_slice = nifty_data_cache.loc[:day_str] if nifty_data_cache is not None else None
                current_regime = data_retriever.calculate_regime_from_data(nifty_slice)
                log.info(f"Regime for {day_str} calculated as: {current_regime}")
                
                market_context_for_day = {"CURRENT_MARKET_REGIME": current_regime, "sector_performance_today": {}}
                
                macro_data = {}
                if benchmarks_data_cache is not None and not benchmarks_data_cache.empty:
                    benchmarks_slice = benchmarks_data_cache.loc[:day_str].tail(6)
                    if len(benchmarks_slice) > 1:
                        changes = ((benchmarks_slice.iloc[-1] - benchmarks_slice.iloc[0]) / benchmarks_slice.iloc[0]) * 100
                        macro_data = {
                            "Nifty50_5D_Change": f"{changes.get('Nifty50', 0):.2f}%",
                            "CrudeOil_5D_Change": f"{changes.get('Crude Oil', 0):.2f}%",
                            "Gold_5D_Change": f"{changes.get('Gold', 0):.2f}%",
                            "USDINR_5D_Change": f"{changes.get('USD-INR', 0):.2f}%",
                        }

                for ticker in stocks_to_test:
                    if ticker not in full_historical_data_cache: continue
                    data_slice = full_historical_data_cache[ticker].loc[:day_str].copy()
                    if len(data_slice) < 100: continue

                    log.info(f"--- Analyzing Ticker: {ticker} for {day_str} ---")
                    
                    # --- Data Preparation for AI ---
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
                    
                    candle_high = latest_row['high']
                    candle_low = latest_row['low']
                    candle_close = latest_row['close']
                    candle_range = candle_high - candle_low if candle_high > candle_low else 0.01
                    position_in_range = (candle_close - candle_low) / candle_range if candle_range > 0 else 0.5
                    latest_indicators['Confirmation_Candle'] = {"position_in_range": round(position_in_range, 2)}
                    
                    atr_percent = (latest_row['ATRr_14'] / latest_row['close']) * 100
                    latest_indicators['ATR_Percent'] = f"{atr_percent:.2f}%"
                    
                    stock_5d_change = ((latest_row['close'] - data_slice_copy['close'].iloc[-6]) / data_slice_copy['close'].iloc[-6]) * 100 if len(data_slice_copy) > 5 else 0
                    market_context_for_day['Stock_5D_Change'] = f"{stock_5d_change:.2f}%"
                    
                    charts_for_ai = {}
                    if generate_charts:
                        charts_for_ai = technical_analyzer.generate_focused_charts(data_slice, ticker)

                    # --- Step 1: Determine the default strategy profile based on Stock Personality ---
                    if ticker in config.HIGH_BETA_CYCLICAL_TICKERS:
                        default_strategy_profile = config.DEFAULT_SWING_STRATEGY
                    elif ticker in config.LOW_VOLATILITY_COMPOUNDER_TICKERS:
                        default_strategy_profile = config.LOW_VOLATILITY_STRATEGY
                    else: # Default to Stable Blue-Chip for all others
                        default_strategy_profile = config.BLUE_CHIP_STRATEGY
                    
                    log.info(f"Applying default profile ('{default_strategy_profile['name']}') for {ticker}")

                    # --- START: Robust CIO Logic with Key Rotation ---
                    final_analysis = None
                    best_analysis_info = None
                    max_retries = len(key_manager.api_keys)
                    for attempt in range(max_retries):
                        try:
                            momentum_analysis = analyzer.get_momentum_analysis(
                                live_financial_data=live_financial_data, latest_atr=latest_row['ATRr_14'], model_name=config.PRO_MODEL,
                                charts=charts_for_ai, trading_horizon_text=default_strategy_profile['horizon_text'],
                                technical_indicators=latest_indicators, min_rr_ratio=default_strategy_profile['min_rr_ratio'],
                                market_context=market_context_for_day, options_data={}, macro_data=macro_data
                            )
                            mean_reversion_analysis = analyzer.get_mean_reversion_analysis(
                                live_financial_data=live_financial_data, model_name=config.FLASH_MODEL,
                                technical_indicators=latest_indicators, market_context=market_context_for_day
                            )
                            breakout_analysis = analyzer.get_breakout_analysis(
                                live_financial_data=live_financial_data, model_name=config.FLASH_MODEL,
                                technical_indicators=latest_indicators, market_context=market_context_for_day
                            )

                            analyses = []
                            if momentum_analysis: analyses.append({'name': 'Momentum', 'score': momentum_analysis.get('scrybeScore', 0), 'analysis': momentum_analysis})
                            if mean_reversion_analysis: analyses.append({'name': 'Mean-Reversion', 'score': mean_reversion_analysis.get('scrybeScore', 0), 'analysis': mean_reversion_analysis})
                            if breakout_analysis: analyses.append({'name': 'Breakout', 'score': breakout_analysis.get('scrybeScore', 0), 'analysis': breakout_analysis})

                            if analyses:
                                best_analysis_info = max(analyses, key=lambda x: abs(x['score']))
                                final_analysis = best_analysis_info['analysis']
                                log.info(f"CIO Decision for {ticker}: {best_analysis_info['name']} strategy selected (Score: {best_analysis_info['score']}).")
                            else:
                                final_analysis = None
                                log.error(f"All three specialists failed to provide an analysis for {ticker}.")
                            
                            break 

                        except Exception as e:
                            if "429" in str(e) and "quota" in str(e).lower():
                                log.warning(f"Quota error on attempt {attempt + 1} for {ticker}.")
                                if attempt < max_retries - 1:
                                    new_key = key_manager.rotate_key()
                                    analyzer = AIAnalyzer(api_key=new_key)
                                    log.info("Retrying analysis with the new key...")
                                else:
                                    log.error(f"All {max_retries} API keys have been exhausted. Skipping ticker for this day.")
                                    final_analysis = None
                            else:
                                log.error(f"A non-quota error occurred during AI analysis for {ticker}. Error: {e}")
                                final_analysis = None
                                break
                    
                    if final_analysis:
                        # --- Step 2: Select the FINAL strategy parameters based on the WINNING specialist ---
                        winning_specialist_name = best_analysis_info['name']
                        if winning_specialist_name == 'Breakout':
                            final_active_strategy = config.BREAKOUT_STRATEGY
                        else: # For Momentum or Mean-Reversion, we use the stock's personality profile
                            final_active_strategy = default_strategy_profile
                        
                        # --- Risk Manager Filter ---
                        original_signal = final_analysis.get('signal')
                        scrybe_score = final_analysis.get('scrybeScore', 0)
                        dvm_scores = analyzer.get_dvm_scores(live_financial_data, latest_indicators)
                        final_analysis['dvmScores'] = dvm_scores
                        
                        final_signal = original_signal
                        filter_reason = None

                        if current_regime == 'Bearish' and original_signal == 'BUY':
                            final_signal = 'HOLD'
                            filter_reason = f"MASTER FILTER VETO: BUY signal for {ticker} blocked by Bearish market regime."
                        elif current_regime == 'Bullish' and original_signal == 'SELL' and abs(scrybe_score) < 80:
                             final_signal = 'HOLD'
                             filter_reason = f"MASTER FILTER VETO: SELL signal for {ticker} blocked by Bullish market regime (conviction < 80)."
                        elif abs(scrybe_score) < 60 and original_signal != 'HOLD':
                            final_signal = 'HOLD'
                            filter_reason = f"Signal '{original_signal}' (Score: {scrybe_score}) was vetoed because it did not meet the conviction threshold (>60)."
                        elif original_signal == 'BUY':
                            durability_score = dvm_scores.get('durability', {}).get('score', 100) if dvm_scores else 100
                            if durability_score < 40:
                                final_signal = 'HOLD'
                                filter_reason = f"Signal '{original_signal}' was vetoed due to a poor fundamental Quality (Durability) score."
                        
                        final_analysis['signal'] = final_signal
                        if filter_reason:
                            log.info(filter_reason)
                            final_analysis['analystVerdict'] = filter_reason
                            final_analysis['tradePlan'] = {"status": "Filtered by Risk Manager", "reason": filter_reason}

                        if final_analysis.get('signal') in ['BUY', 'SELL']:
                            signal = final_analysis['signal']
                            entry_price = latest_row['close']
                            atr = latest_row['ATRr_14']
                            rr_ratio = final_active_strategy['min_rr_ratio']
                            stop_multiplier = final_active_strategy['stop_loss_atr_multiplier']
                            stop_loss_price = entry_price - (stop_multiplier * atr) if signal == 'BUY' else entry_price + (stop_multiplier * atr)
                            target_price = entry_price + ((stop_multiplier * atr) * rr_ratio) if signal == 'BUY' else entry_price - ((stop_multiplier * atr) * rr_ratio)
                            final_analysis['tradePlan'] = {
                                "timeframe": final_active_strategy['horizon_text'],
                                "strategy": f"{final_active_strategy['name']} (ATR-based R/R)",
                                "entryPrice": {"price": round(entry_price, 2), "rationale": "Closing price on prediction day."},
                                "target": {"price": round(target_price, 2), "rationale": f"Calculated using {rr_ratio} R/R based on {atr:.2f} ATR."},
                                "stopLoss": {"price": round(stop_loss_price, 2), "rationale": f"Calculated using {stop_multiplier}*ATR ({atr:.2f} ATR)."},
                                "riskRewardRatio": rr_ratio
                            }
                        else:
                            if 'tradePlan' not in final_analysis:
                                final_analysis['tradePlan'] = {}

                        correlations = performance_analyzer.calculate_correlations(data_slice, benchmarks_data_cache.loc[:day_str] if benchmarks_data_cache is not None else None)
                        prediction_doc = final_analysis.copy()
                        prediction_doc.update({
                            'analysis_id': str(uuid.uuid4()), 'ticker': ticker,
                            'prediction_date': current_day.to_pydatetime(), 'price_at_prediction': latest_row['close'],
                            'status': 'open', 'strategy': final_active_strategy['name'],
                            'dvmScores': dvm_scores, 'correlations': correlations,
                            'atr_at_prediction': latest_row['ATRr_14']
                        })
                        database_manager.save_prediction_for_backtesting(prediction_doc, batch_id)
                    
                    time.sleep(5)

            except Exception as e:
                log.error(f"A critical error occurred on the simulation for day {day_str}. Error: {e}", exc_info=True)
                # Continue to the next day instead of stopping the whole run
                continue
            
            if i + 1 < len(simulation_days):
                save_state(simulation_days[i+1])

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