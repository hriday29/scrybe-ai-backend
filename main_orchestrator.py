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
from run_backtest_simulation import run_backtest_simulation

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
    run_backtest_simulation(batch_id)

    # --- Generate Final Report (Phase 3) ---
    log.info("=" * 80)
    log.info("--- Generating Final Backtest Report (Phase 3)... ---")
    report_start_time = time.time()
    # Call the report generator - assumes it reads from DB using batch_id
    # We will need to modify performance_analyzer.py for this
    generate_backtest_report(batch_id) # Pass batch_id instead of portfolio
    log.info(f"--- ✅ Report Generation Finished in {time.time()-report_start_time:.2f}s ---")
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
