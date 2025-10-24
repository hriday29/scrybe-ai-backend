# main_orchestrator.py
import pandas as pd
import time
import config
from logger_config import log
import database_manager
import json
import os
import argparse
import yfinance as yf
from config import PORTFOLIO_CONSTRAINTS
from collections import deque
from sector_analyzer import CORE_SECTOR_INDICES, BENCHMARK_INDEX
import index_manager
from analysis_pipeline import AnalysisPipeline, BENCHMARK_TICKERS
from pandas.tseries.offsets import BDay
from performance_analyzer import generate_backtest_report

api_call_timestamps = deque()

def run_simulation(batch_id: str, start_date: str, end_date: str, is_fresh_run: bool = False, tickers: list = None, batch_num: int = 1, total_batches: int = 1):
    """
    Runs a memory-efficient, day-by-day backtest simulation using on-demand data loading
    with file caching for the Angel One data source.
    Phase 1: generate and save 'open' predictions to the DB.
    """
    overall_start = time.time()
    log.info("=" * 80)
    log.info(f"### STARTING UNIFIED SIMULATION FOR BATCH: {batch_id} ###")
    log.info(f"PERIOD: {start_date} → {end_date}")
    log.info("=" * 80)

    pipeline = AnalysisPipeline()
    pipeline._setup(mode='scheduler')

    # --- Fetch the Nifty Smallcap 250 universe ONCE for the entire backtest run ---
    log.info("Fetching the Nifty Smallcap 250 list for this backtest run...")
    backtest_universe = index_manager.get_nifty_smallcap_250_tickers()
    if not backtest_universe:
        log.fatal("[SETUP ERROR] Could not fetch the Smallcap ticker list. Aborting backtest.")
        pipeline.close()
        return
    log.info(f"Using a static universe of {len(backtest_universe)} Smallcap tickers for this run.")
    # --- End fetching universe ---

    # --- NEW: SLICE UNIVERSE FOR BATCH PROCESSING ---
    if total_batches > 1:
        if not (1 <= batch_num <= total_batches):
            log.error(f"Invalid batch_num {batch_num}. Must be between 1 and {total_batches}.")
            pipeline.close()
            return

        universe_size = len(backtest_universe)
        batch_size = (universe_size + total_batches - 1) // total_batches  # Ceiling division
        start_idx = (batch_num - 1) * batch_size
        end_idx = min(batch_num * batch_size, universe_size)  # Ensure end_idx doesn't exceed list size

        backtest_universe_batch = backtest_universe[start_idx:end_idx]

        log.warning(f"RUNNING BATCH {batch_num}/{total_batches}: Processing tickers {start_idx+1} to {end_idx} "
                    f"(Size: {len(backtest_universe_batch)} tickers).")
    else:
        # If total_batches is 1, run the full universe
        backtest_universe_batch = backtest_universe
        log.info("Running on the full fetched universe (total_batches=1).")
    # --- END UNIVERSE SLICING ---

    # --- Use the sliced batch 'backtest_universe_batch' from now on ---
    log.info(f"Using a universe of {len(backtest_universe_batch)} tickers for this batch.")

    if is_fresh_run:
        log.warning("FRESH RUN: Deleting previous predictions and performance data for this batch.")
        if getattr(database_manager, "predictions_collection", None) is not None:
            database_manager.predictions_collection.delete_many({"batch_id": batch_id})
        if getattr(database_manager, "performance_collection", None) is not None:
            database_manager.performance_collection.delete_many({"batch_id": batch_id})

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'nifty50_historical_constituents.csv')
        constituents_df = pd.read_csv(csv_path)
        constituents_df['date'] = pd.to_datetime(constituents_df['date'])
        log.info("✅ Historical constituents list loaded successfully.")
    except Exception as e:
        log.fatal(f"[SETUP ERROR] Could not load constituents file: {e}")
        return

    simulation_days = pd.bdate_range(start=start_date, end=end_date)
    total_days = len(simulation_days)

    # --- Main Simulation Loop (Phase 1: analysis generation) ---
    for i, current_day in enumerate(simulation_days):
        # --- NEW: Daily Session Health Check ---
        log.info(f"Performing daily session health check for Angel One...")
        import angelone_retriever
        angelone_retriever.initialize_angelone_session()
        # --- END OF NEW CODE ---

        day_start = time.time()
        day_str = current_day.strftime('%Y-%m-%d')
        log.info("")
        log.info("-" * 80)
        log.info(f"--- Simulating Day {i+1}/{total_days}: {day_str} ---")

        try:
            # ======================================================================
            # --- CORRECTED: Define Required Assets using Static Universe ---
            # ======================================================================
            essential_indices = list(set(list(CORE_SECTOR_INDICES.values()) + [BENCHMARK_INDEX] + list(BENCHMARK_TICKERS.values())))
            log.info(f"Attempting separate download for {len(essential_indices)} essential indices...")

            # --- NEW: Rate Limiting Parameters for yfinance ---
            YFINANCE_RETRY_ATTEMPTS = 3
            YFINANCE_RETRY_DELAY_SECONDS = 30
            YFINANCE_INTER_CALL_DELAY_SECONDS = 5
            # --- End Rate Limiting Parameters ---

            # Prepare container dicts for index data
            index_data_analysis = {}
            index_data_simulation = {}

            # Prepare date strings used in downloads
            decision_date = (current_day - pd.tseries.offsets.BDay(1)).normalize()
            decision_date_str = decision_date.strftime('%Y-%m-%d')
            simulation_date_str = current_day.strftime('%Y-%m-%d')

            # --- NEW: Separate Index Downloads (Analysis: 5y up to decision_date, Simulation: 6mo up to simulation_date) ---
            try:
                index_analysis_raw = None
                index_sim_raw = None

                # Attempt analysis-length download for indices
                for attempt in range(YFINANCE_RETRY_ATTEMPTS):
                    try:
                        index_analysis_raw = yf.download(
                            tickers=essential_indices,
                            end=simulation_date_str,
                            period="5y",
                            progress=False,
                            ignore_tz=True,
                            auto_adjust=True
                        )
                        if index_analysis_raw is not None and not index_analysis_raw.empty:
                            log.info("Index analysis download successful.")
                            break
                        else:
                            log.warning(f"Index analysis attempt {attempt+1} returned empty.")
                            if attempt == YFINANCE_RETRY_ATTEMPTS - 1:
                                raise ValueError("Index analysis download empty after retries.")
                    except Exception as idx_an_e:
                        log.warning(f"Index analysis download attempt {attempt+1} failed: {idx_an_e}")
                        if attempt < YFINANCE_RETRY_ATTEMPTS - 1:
                            time.sleep(YFINANCE_RETRY_DELAY_SECONDS / 2)
                        else:
                            raise

                time.sleep(YFINANCE_INTER_CALL_DELAY_SECONDS / 2)

                # Attempt simulation-length download for indices
                for attempt in range(YFINANCE_RETRY_ATTEMPTS):
                    try:
                        index_sim_raw = yf.download(
                            tickers=essential_indices,
                            end=simulation_date_str,
                            period="6mo",
                            progress=False,
                            ignore_tz=True
                        )
                        if index_sim_raw is not None and not index_sim_raw.empty:
                            log.info("Index simulation download successful.")
                            break
                        else:
                            log.warning(f"Index simulation attempt {attempt+1} returned empty.")
                            if attempt == YFINANCE_RETRY_ATTEMPTS - 1:
                                raise ValueError("Index simulation download empty after retries.")
                    except Exception as idx_sim_e:
                        log.warning(f"Index simulation download attempt {attempt+1} failed: {idx_sim_e}")
                        if attempt < YFINANCE_RETRY_ATTEMPTS - 1:
                            time.sleep(YFINANCE_RETRY_DELAY_SECONDS / 2)
                        else:
                            raise

                # Process downloaded index data into dict format
                if index_analysis_raw is not None and not index_analysis_raw.empty:
                    index_analysis_data = index_analysis_raw.loc[:decision_date_str]
                    for ticker in essential_indices:
                        try:
                            df = index_analysis_data.xs(ticker, level=1, axis=1).dropna(how='all')
                            df.rename(columns={
                                'Open': 'open', 'High': 'high', 'Low': 'low',
                                'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'
                            }, inplace=True)
                            if not df.empty:
                                index_data_analysis[ticker] = df[['open', 'high', 'low', 'close', 'volume']]
                        except KeyError:
                            # Index may not have been returned for this ticker
                            pass
                        except Exception as ex_proc:
                            log.warning(f"Error processing index analysis data for {ticker}: {ex_proc}")

                if index_sim_raw is not None and not index_sim_raw.empty:
                    for ticker in essential_indices:
                        try:
                            df = index_sim_raw.xs(ticker, level=1, axis=1).dropna(how='all')
                            df.rename(columns={
                                'Open': 'open', 'High': 'high', 'Low': 'low',
                                'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'
                            }, inplace=True)
                            if not df.empty:
                                index_data_simulation[ticker] = df[['open', 'high', 'low', 'close', 'volume']]
                        except KeyError:
                            pass
                        except Exception as ex_proc:
                            log.warning(f"Error processing index simulation data for {ticker}: {ex_proc}")

                log.info(f"Indices downloaded. Analysis: {len(index_data_analysis)}, Simulation: {len(index_data_simulation)}")

            except Exception as idx_e:
                log.error(f"Failed to download essential indices separately after retries: {idx_e}. Market context may be incomplete.")
                # Continue without index data; downstream code should handle missing keys gracefully

            # --- Now define assets for the main STOCK bulk download ---
            stock_assets_for_today = [t for t in backtest_universe_batch if t not in essential_indices]
            log.info(f"[DATA] Preparing to load data for {len(stock_assets_for_today)} stock assets for {day_str}...")

            # ======================================================================
            # --- MODIFIED: Bulk Data Loading (Now for STOCKS ONLY) ---
            # ======================================================================
            data_load_start = time.time()
            bulk_stock_data_for_analysis = None
            bulk_stock_data_for_simulation = None

            try:
                # --- Download for Analysis (Stocks Only) with Retry ---
                for attempt in range(YFINANCE_RETRY_ATTEMPTS):
                    try:
                        log.info(f"Attempting STOCKS bulk download for ANALYSIS, Attempt {attempt+1}/{YFINANCE_RETRY_ATTEMPTS}...")
                        bulk_stock_data_analysis_raw = yf.download(
                            tickers=stock_assets_for_today,
                            end=simulation_date_str,
                            period="5y",
                            progress=False,
                            ignore_tz=True
                        )
                        if bulk_stock_data_analysis_raw is not None and not bulk_stock_data_analysis_raw.empty:
                            bulk_stock_data_for_analysis = bulk_stock_data_analysis_raw.loc[:decision_date_str]
                            log.info("Stock analysis data download successful.")
                            break
                        else:
                            log.warning(f"Attempt {attempt+1}: Stock analysis download returned empty.")
                            if attempt == YFINANCE_RETRY_ATTEMPTS - 1:
                                raise ValueError("Stock analysis download empty.")
                    except Exception as yf_e:
                        log.warning(f"Attempt {attempt+1} failed for stock analysis download: {yf_e}")
                        if attempt < YFINANCE_RETRY_ATTEMPTS - 1:
                            time.sleep(YFINANCE_RETRY_DELAY_SECONDS)
                        else:
                            log.error("Stock analysis download failed.")
                            raise

                log.info(f"Pausing for {YFINANCE_INTER_CALL_DELAY_SECONDS}s...")
                time.sleep(YFINANCE_INTER_CALL_DELAY_SECONDS)

                # --- Download for Simulation (Stocks Only) with Retry ---
                for attempt in range(YFINANCE_RETRY_ATTEMPTS):
                    try:
                        log.info(f"Attempting STOCKS bulk download for SIMULATION, Attempt {attempt+1}/{YFINANCE_RETRY_ATTEMPTS}...")
                        bulk_stock_data_simulation_raw = yf.download(
                            tickers=stock_assets_for_today,
                            end=simulation_date_str,
                            period="6mo",
                            progress=False,
                            ignore_tz=True
                        )
                        if bulk_stock_data_simulation_raw is not None and not bulk_stock_data_simulation_raw.empty:
                            bulk_stock_data_for_simulation = bulk_stock_data_simulation_raw
                            log.info("Stock simulation data download successful.")
                            break
                        else:
                            log.warning(f"Attempt {attempt+1}: Stock simulation download returned empty.")
                            if attempt == YFINANCE_RETRY_ATTEMPTS - 1:
                                raise ValueError("Stock simulation download empty.")
                    except Exception as yf_e:
                        log.warning(f"Attempt {attempt+1} failed for stock simulation download: {yf_e}")
                        if attempt < YFINANCE_RETRY_ATTEMPTS - 1:
                            time.sleep(YFINANCE_RETRY_DELAY_SECONDS)
                        else:
                            log.error("Stock simulation download failed.")
                            raise

                # Check final status
                if bulk_stock_data_for_analysis is None or bulk_stock_data_for_simulation is None:
                    raise ValueError("One or both stock bulk data downloads failed after retries.")

                log.info(f"[DATA] Stock bulk data loading completed in {time.time()-data_load_start:.2f}s")

            except Exception as dl_e:
                log.error(f"[DATA ERROR] Stock bulk data download ultimately failed for {day_str}: {dl_e}. Skipping day.")
                continue  # Skip to the next day

            # ======================================================================
            # --- Convert Multi-Index Data & COMBINE with Index Data ---
            # ======================================================================
            # Start with the index data we fetched separately
            data_for_analysis = index_data_analysis.copy()
            data_for_simulation = index_data_simulation.copy()

            # Process and add stock data for analysis
            if bulk_stock_data_for_analysis is not None:
                for ticker in stock_assets_for_today:  # Iterate stocks only
                    try:
                        df = bulk_stock_data_for_analysis.xs(ticker, level=1, axis=1).dropna(how='all')
                        df.rename(columns={
                            'Open': 'open', 'High': 'high', 'Low': 'low',
                            'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'
                        }, inplace=True)
                        if not df.empty:
                            data_for_analysis[ticker] = df[['open', 'high', 'low', 'close', 'volume']]
                    except KeyError:
                        log.debug(f"No analysis data found for {ticker} in stock bulk download for {decision_date_str}.")
                    except Exception as ex:
                        log.warning(f"Error processing stock analysis data for {ticker}: {ex}")

            # Process and add stock data for simulation
            if bulk_stock_data_for_simulation is not None:
                for ticker in stock_assets_for_today:  # Iterate stocks only
                    try:
                        df = bulk_stock_data_for_simulation.xs(ticker, level=1, axis=1).dropna(how='all')
                        df.rename(columns={
                            'Open': 'open', 'High': 'high', 'Low': 'low',
                            'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'
                        }, inplace=True)
                        if not df.empty:
                            data_for_simulation[ticker] = df[['open', 'high', 'low', 'close', 'volume']]
                    except KeyError:
                        log.debug(f"No simulation data found for {ticker} in stock bulk download for {simulation_date_str}.")
                    except Exception as ex:
                        log.warning(f"Error processing stock simulation data for {ticker}: {ex}")

            # Now data_for_analysis and data_for_simulation contain BOTH index and stock data
            log.info(f"[DATA] Combined data caches created. "
                     f"Analysis: {len(data_for_analysis)} assets, Simulation: {len(data_for_simulation)} assets.")
            # ======================================================================

            # --- Run Analysis Pipeline (Phase 1) ---
            # This generates and saves 'open' predictions to the database.
            pipeline_start = time.time()
            pipeline.run(
                point_in_time=decision_date,
                full_data_cache=data_for_analysis, # Use the 5y data cache
                is_backtest=True,
                batch_id=batch_id
            )
            log.info(f"[PIPELINE] Completed analysis for {day_str} in {time.time()-pipeline_start:.2f}s")
            # --- End Analysis Pipeline ---

            log.info(f"--- Day {i+1}/{total_days} analysis generated in {time.time()-day_start:.2f}s ---")

        except Exception as e:
            log.error(f"[ERROR] Exception during analysis generation for {day_str}: {e}", exc_info=True)
            # Optionally add logic here to mark the day as failed if needed
            continue # Continue to the next day even if one day fails

    # --- End Main Simulation Loop (Phase 1) ---
    log.info("=" * 80)
    log.info("--- ✅ Analysis Generation (Phase 1) Finished! ---")
    log.info(f"Total Analysis Generation Time: {time.time()-overall_start:.2f}s")
    log.info("=" * 80)

    # --- Run Backtest Simulation (Phase 2) ---
    run_backtest_simulation(batch_id, start_date, end_date)

    # --- Generate Final Report (Phase 3) ---
    log.info("=" * 80)
    log.info("--- Generating Final Backtest Report (Phase 3)... ---")
    report_start_time = time.time()
    # Call the report generator - assumes it reads from DB using batch_id
    # We will need to modify performance_analyzer.py for this
    generate_backtest_report(batch_id) # Pass batch_id instead of portfolio
    log.info(f"--- ✅ Report Generation Finished in {time.time()-report_start_time:.2f}s ---")
    log.info("=" * 80)


def run_backtest_simulation(batch_id: str, start_date: str, end_date: str):
    """
    Phase 2: Simulates the lifecycle of all 'open' trades saved during Phase 1.
    Reads 'open' trades, simulates them day-by-day using cached historical data,
    saves closed trades to the 'performance' collection, and updates the
    original prediction status.
    """
    log.info("")
    log.info("=" * 80)
    log.info(f"### STARTING BACKTEST SIMULATION (Phase 2) for Batch: {batch_id} ###")
    log.info("=" * 80)
    sim_start_time = time.time()

    open_trades = database_manager.get_open_backtest_trades(batch_id)
    if not open_trades:
        log.warning("No open trades found in the database for this batch_id. Cannot run simulation.")
        return

    # Determine the full range needed for simulation data
    sim_start_date = pd.to_datetime(start_date) - BDay(1) # Need data from the day *before* the first prediction
    sim_end_date = pd.to_datetime(end_date)
    simulation_days_range = pd.date_range(start=sim_start_date, end=sim_end_date, freq='B') # Use BDays

    # --- Efficient Data Loading for Simulation ---
    # Load data ONCE for all tickers needed across all open trades for the simulation period
    log.info("Loading required historical data for simulation phase...")
    tickers_needed = list(set(trade['ticker'] for trade in open_trades))
    full_data_cache_sim = {}
    total_tickers = len(tickers_needed)
    from data_retriever import get_historical_stock_data # Local import

    for i, ticker in enumerate(tickers_needed):
        # Fetch data up to the end_date of the simulation.
        # get_historical_stock_data uses file caching, so this is fast on subsequent runs.
        df = get_historical_stock_data(ticker, end_date=sim_end_date.strftime('%Y-%m-%d'))
        if df is not None and not df.empty:
            # Only keep data within the simulation range + a buffer for indicators if needed
            full_data_cache_sim[ticker] = df.loc[sim_start_date - pd.Timedelta(days=60):sim_end_date] # Keep ~2 months buffer
        else:
            log.warning(f"Could not load simulation data for {ticker}. Trades for this ticker will be skipped.")
        if (i + 1) % 50 == 0:
            log.info(f"Loaded simulation data for {i+1}/{total_tickers} tickers...")

    log.info(f"✅ Simulation data loaded for {len(full_data_cache_sim)} tickers.")
    # --- End Data Loading ---

    closed_trade_count = 0
    processed_prediction_count = 0
    strategy_config = config.APEX_SWING_STRATEGY # Load strategy params once

    # --- Simulate Each Open Trade ---
    for trade in open_trades:
        ticker = trade['ticker']
        prediction_id = trade['_id'] # Use MongoDB's _id
        entry_signal = trade['signal']
        entry_date_dt = pd.to_datetime(trade['prediction_date']).tz_localize(None) + BDay(1) # Trade entry is T+1
        entry_price_signal = trade['tradePlan']['entryPrice'] # This is usually the close of prediction day
        stop_loss_initial = trade['tradePlan']['stopLoss']
        target_initial = trade['tradePlan']['target']
        holding_period = strategy_config['holding_period']

        # Get the historical data for this specific ticker
        hist_data = full_data_cache_sim.get(ticker)
        if hist_data is None or hist_data.empty:
            log.warning(f"Skipping simulation for {ticker} on {entry_date_dt.date()}: Missing historical data.")
            database_manager.mark_backtest_prediction_processed(prediction_id, batch_id, new_status="error_missing_data")
            continue

        # Find the actual entry day's data (T+1)
        try:
            # Ensure entry_date_dt exists in the index
            entry_day_data = hist_data.loc[entry_date_dt:entry_date_dt] # Select the row for entry day
            if entry_day_data.empty:
                 # Try finding the next available business day if T+1 was a holiday
                 next_bday = entry_date_dt
                 while next_bday <= sim_end_date:
                     entry_day_data = hist_data.loc[next_bday:next_bday]
                     if not entry_day_data.empty:
                         entry_date_dt = next_bday # Update actual entry date
                         log.info(f"Adjusted entry date for {ticker} to next BDay: {entry_date_dt.date()}")
                         break
                     next_bday += BDay(1)
                 if entry_day_data.empty:
                     raise IndexError("Could not find valid entry day data within simulation range.")

            actual_entry_price = entry_day_data['open'].iloc[0] # Enter at T+1 open

            # Check for immediate stop-out on gap
            if entry_signal == 'BUY' and actual_entry_price <= stop_loss_initial:
                closing_price = actual_entry_price
                close_reason = "Stop-Loss Hit (Gap Down at Entry)"
                close_date_dt = entry_date_dt
            elif entry_signal == 'SHORT' and actual_entry_price >= stop_loss_initial:
                closing_price = actual_entry_price
                close_reason = "Stop-Loss Hit (Gap Up at Entry)"
                close_date_dt = entry_date_dt
            else:
                 # --- Day-by-day simulation loop ---
                 close_reason = None
                 closing_price = None
                 close_date_dt = None
                 trailing_stop = stop_loss_initial # Initialize trailing stop

                 # Iterate from the day *after* entry up to holding period or end_date
                 sim_days_for_trade = simulation_days_range[simulation_days_range > entry_date_dt]

                 for current_sim_day in sim_days_for_trade:
                     if (current_sim_day - entry_date_dt).days >= holding_period:
                         close_reason = "Time Exit (Holding Period)"
                         closing_price = hist_data.loc[current_sim_day]['close'] # Exit at close on expiry day
                         close_date_dt = current_sim_day
                         break

                     try:
                         day_data = hist_data.loc[current_sim_day]
                         open_p, high_p, low_p, close_p = day_data['open'], day_data['high'], day_data['low'], day_data['close']

                         # Trailing Stop Logic (simplified example - needs ATR calculation if using config)
                         # Add ATR calculation here if use_trailing_stop is True in config
                         # For now, using fixed initial stop as trailing stop for simplicity

                         # Check Exit Conditions (Priority: Gap SL > Intraday SL > Target)
                         if entry_signal == 'BUY':
                             if open_p <= stop_loss_initial: # Gap Check
                                 close_reason, closing_price = "Stop-Loss Hit (Gap Down)", open_p
                             elif low_p <= trailing_stop: # Intraday Stop Check (using trailing stop)
                                 close_reason, closing_price = "Trailing Stop Hit", trailing_stop
                             elif low_p <= stop_loss_initial: # Intraday Original Stop Check
                                 close_reason, closing_price = "Stop-Loss Hit", stop_loss_initial
                             elif target_initial and high_p >= target_initial: # Target Check
                                 close_reason, closing_price = "Target Hit", target_initial
                         elif entry_signal == 'SHORT':
                             if open_p >= stop_loss_initial: # Gap Check
                                 close_reason, closing_price = "Stop-Loss Hit (Gap Up)", open_p
                             elif high_p >= trailing_stop: # Intraday Stop Check (using trailing stop)
                                 close_reason, closing_price = "Trailing Stop Hit", trailing_stop
                             elif high_p >= stop_loss_initial: # Intraday Original Stop Check
                                 close_reason, closing_price = "Stop-Loss Hit", stop_loss_initial
                             elif target_initial and low_p <= target_initial: # Target Check
                                 close_reason, closing_price = "Target Hit", target_initial

                         if close_reason:
                             close_date_dt = current_sim_day
                             break # Exit found

                     except KeyError:
                         # Skip if data for current_sim_day is missing for this ticker
                         continue
                     except Exception as day_e:
                         log.error(f"Error simulating day {current_sim_day.date()} for {ticker}: {day_e}")
                         close_reason = "Error During Simulation"
                         closing_price = hist_data.loc[current_sim_day]['close'] if current_sim_day in hist_data.index else actual_entry_price # Fallback exit price
                         close_date_dt = current_sim_day
                         break

                 # If loop finished without exit, it's a time exit on the last possible day
                 if not close_reason:
                     last_valid_day = hist_data.index[-1]
                     if last_valid_day > entry_date_dt:
                          close_reason = "Time Exit (End of Data)"
                          closing_price = hist_data.loc[last_valid_day]['close']
                          close_date_dt = last_valid_day
                     else: # Handle case where there's no data after entry
                          close_reason = "Error - No Data Post Entry"
                          closing_price = actual_entry_price
                          close_date_dt = entry_date_dt

            # --- Calculate P&L and Save Performance ---
            # Re-calculate PnL based on actual entry and simulated exit
            if entry_signal == 'BUY':
                gross_pnl_share = closing_price - actual_entry_price
            elif entry_signal == 'SHORT':
                gross_pnl_share = actual_entry_price - closing_price
            else:
                gross_pnl_share = 0

            # NOTE: We don't have num_shares here as position sizing wasn't done yet.
            # We save % return instead of absolute PnL.
            net_return_pct = (gross_pnl_share / actual_entry_price) * 100 if actual_entry_price != 0 else 0

            # Apply Costs (can refine this later)
            costs = config.BACKTEST_CONFIG
            total_costs_pct = (costs['brokerage_pct'] * 2) + (costs['slippage_pct'] * 2) + costs.get('stt_pct', 0.1) # Added STT default
            final_net_return_pct = net_return_pct - total_costs_pct

            performance_doc = {
                "prediction_id": prediction_id, # Link back to the original prediction
                "ticker": ticker,
                "strategy": trade['strategy'],
                "signal": entry_signal,
                "status": "Closed",
                "open_date": entry_date_dt.to_pydatetime(), # Actual entry date/time
                "close_date": close_date_dt.to_pydatetime(),
                "entry_price": round(actual_entry_price, 2),
                "close_price": round(closing_price, 2),
                "closing_reason": close_reason,
                "net_pnl": None, # Cannot calculate without num_shares
                "net_return_pct": round(final_net_return_pct, 2),
                "batch_id": batch_id
            }
            database_manager.save_backtest_performance_trade(performance_doc)
            database_manager.mark_backtest_prediction_processed(prediction_id, batch_id) # Mark original prediction
            closed_trade_count += 1
            processed_prediction_count += 1

        except IndexError:
             log.warning(f"Skipping trade simulation for {ticker}: Could not find valid entry day data around {entry_date_dt.date()}. Marking as error.")
             database_manager.mark_backtest_prediction_processed(prediction_id, batch_id, new_status="error_no_entry_data")
        except Exception as sim_e:
            log.error(f"CRITICAL ERROR simulating trade for {ticker} (PredID: {prediction_id}): {sim_e}", exc_info=True)
            database_manager.mark_backtest_prediction_processed(prediction_id, batch_id, new_status="error_simulation_failed")

    log.info(f"--- Backtest Simulation (Phase 2) Finished ---")
    log.info(f"Processed {processed_prediction_count}/{len(open_trades)} open predictions.")
    log.info(f"Saved {closed_trade_count} closed trades to performance collection.")
    log.info(f"Phase 2 completed in {time.time() - sim_start_time:.2f} seconds.")
    log.info("=" * 80)


# --- CLI Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a full backtest using the main orchestrator.")
    parser.add_argument("--batch_id", required=True, help="Unique ID for the backtest batch.")
    parser.add_argument("--start_date", required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end_date", required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument('--fresh_run', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Delete previous data for this batch_id? (true/false)')
    parser.add_argument("--tickers", nargs='+', required=False,
                        help="Optional: Space-separated list of tickers to backtest instead of index.")

    # --- BATCH ARGUMENTS ---
    parser.add_argument("--batch_num", type=int, default=1,
                        help="Current batch number (1-based) if splitting the universe.")
    parser.add_argument("--total_batches", type=int, default=1,
                        help="Total number of batches the universe is split into.")
    # --- END ADDED ARGUMENTS ---

    args = parser.parse_args()

    run_simulation(
        batch_id=args.batch_id,
        start_date=args.start_date,
        end_date=args.end_date,
        is_fresh_run=args.fresh_run,
        tickers=args.tickers,
        # --- PASS BATCH ARGUMENTS ---
        batch_num=args.batch_num,
        total_batches=args.total_batches
        # --- END PASSING ARGUMENTS ---
    )
