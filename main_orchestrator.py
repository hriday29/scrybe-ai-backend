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

STATE_FILE = 'simulation_state.json'

def save_state(next_day_to_run):
    with open(STATE_FILE, 'w') as f:
        json.dump({'next_start_date': next_day_to_run.strftime('%Y-%m-%d')}, f)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            log.warning(f"Resuming from saved state. Next day to run: {state['next_start_date']}")
            return state['next_start_date']
    return None

api_call_timestamps = deque()

def run_simulation(batch_id: str, start_date: str, end_date: str, is_fresh_run: bool = False, tickers: list = None, batch_num: int = 1, total_batches: int = 1):
    """
    Runs a memory-efficient, day-by-day backtest simulation using on-demand data loading
    with file caching for the Angel One data source.
    """
    overall_start = time.time()
    log.info("=" * 80)
    log.info(f"### STARTING UNIFIED SIMULATION FOR BATCH: {batch_id} ###")
    log.info(f"PERIOD: {start_date} → {end_date}")
    log.info("=" * 80)

    pipeline = AnalysisPipeline()
    pipeline._setup(mode='scheduler')

    # --- Fetch the Nifty Smallcap 250 universe ONCE for the entire backtest run --- # <-- MODIFIED COMMENT
    log.info("Fetching the Nifty Smallcap 250 list for this backtest run...") # <-- MODIFIED LOG
    backtest_universe = index_manager.get_nifty_smallcap_250_tickers() # <-- USE NEW FUNCTION
    if not backtest_universe:
        log.fatal("[SETUP ERROR] Could not fetch the Smallcap ticker list. Aborting backtest.") # <-- MODIFIED LOG
        pipeline.close()
        return
    log.info(f"Using a static universe of {len(backtest_universe)} Smallcap tickers for this run.") # <-- MODIFIED LOG
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

    # --- Portfolio Setup ---
    portfolio = {
        'equity': config.BACKTEST_PORTFOLIO_CONFIG['initial_capital'],
        'open_positions': [],
        'closed_trades': [],
        'daily_equity_log': []
    }

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

    # --- Main Simulation Loop ---
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
            # The 'backtest_universe' variable was fetched *before* the loop started.
            # We'll download essential indices separately, then download stocks excluding those indices.
            essential_indices = list(set(list(CORE_SECTOR_INDICES.values()) + [BENCHMARK_INDEX] + list(BENCHMARK_TICKERS.values())))  # [UPDATED]
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
                            ignore_tz=True
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
            # Exclude indices we already downloaded to avoid downloading them again
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

            # --- Manage Exits ---
            exits_start = time.time()
            before_exits = len(portfolio['open_positions'])
            _manage_exits_for_day(portfolio, current_day, data_for_simulation, batch_id)
            closed_today = before_exits - len(portfolio['open_positions'])
            log.info(f"[EXIT] {closed_today} positions closed in {time.time()-exits_start:.2f}s")

            # --- Run Analysis Pipeline ---
            pipeline_start = time.time()
            pipeline.run(
                point_in_time=decision_date,
                full_data_cache=data_for_analysis,
                is_backtest=True,
                batch_id=batch_id
            )
            log.info(f"[PIPELINE] Completed in {time.time()-pipeline_start:.2f}s")

            # --- Manage Entries ---
            entries_start = time.time()
            signals_for_today = list(database_manager.predictions_collection.find({
                "batch_id": batch_id,
                "prediction_date": decision_date.to_pydatetime(),
                "signal": {"$in": ["BUY", "SHORT"]}
            }))
            before_entries = len(portfolio['open_positions'])
            _manage_entries_for_day(portfolio, signals_for_today, current_day, data_for_simulation)
            opened_today = len(portfolio['open_positions']) - before_entries
            log.info(f"[ENTRY] {len(signals_for_today)} signals processed, "
                    f"{opened_today} trades opened in {time.time()-entries_start:.2f}s")

            # --- Equity Logging ---
            valuation_start = time.time()
            open_value = 0
            for pos in portfolio['open_positions']:
                try:
                    current_price = data_for_simulation[pos['ticker']].loc[current_day]['close']
                    open_value += current_price * pos['num_shares']
                except (KeyError, IndexError):
                    open_value += pos['entry_price'] * pos['num_shares']

            total_equity = portfolio['equity'] + open_value
            portfolio['daily_equity_log'].append({'date': current_day, 'equity': total_equity})
            log.info(f"[PORTFOLIO] End-of-Day Equity: ₹{total_equity:,.2f} "
                    f"(valuation time {time.time()-valuation_start:.2f}s)")

            log.info(f"--- Day {i+1}/{total_days} completed in {time.time()-day_start:.2f}s ---")

        except Exception as e:
            log.error(f"[ERROR] Exception during simulation of {day_str}: {e}", exc_info=True)
            continue

    # --- Final Report ---
    log.info("")
    log.info("=" * 80)
    log.info("--- ✅ UNIFIED SIMULATION FINISHED! Generating Final Report... ---")

    # --- NEW: Final Liquidation Step ---
    log.info("--- Liquidating all open positions at the end of the backtest period... ---")
    final_day = simulation_days[-1] if not simulation_days.empty else None
    if final_day and portfolio['open_positions']:
        # Important: Iterate over a copy of the list because we are modifying it
        for position in portfolio['open_positions'][:]:
            ticker = position['ticker']
            close_reason = "End of Backtest Liquidation"
            closing_price = None

            try:
                # Use the last available data cache to get the final closing price
                if ticker in data_for_simulation and not data_for_simulation[ticker].empty:
                    if final_day in data_for_simulation[ticker].index:
                        closing_price = data_for_simulation[ticker].loc[final_day]['close']
                    else:
                        # Fallback if the stock didn't trade on the very last day
                        closing_price = data_for_simulation[ticker]['close'].iloc[-1]
                        log.warning(f"Ticker {ticker} had no data for final day {final_day.date()}. Using last available price: {closing_price}")

                if closing_price is not None:
                    closed_trade_doc, net_pnl = _calculate_closed_trade(position, closing_price, close_reason, final_day)
                    log.info(f"[LIQUIDATE] Closing {ticker} ({position['signal']}) at {closing_price:.2f}. Net P&L: ₹{net_pnl:.2f}")

                    portfolio['equity'] += net_pnl
                    portfolio['closed_trades'].append(closed_trade_doc)
                    portfolio['open_positions'].remove(position)
                    database_manager.performance_collection.insert_one(closed_trade_doc)
                else:
                    log.error(f"[LIQUIDATE] Could not find final closing price for {ticker}.")
            except Exception as e:
                log.error(f"[LIQUIDATE] Error liquidating {ticker}: {e}", exc_info=True)

    if portfolio['daily_equity_log']:
        portfolio['daily_equity_log'][-1]['equity'] = portfolio['equity']

    # All reporting logic is now handled by the dedicated analyzer.
    generate_backtest_report(portfolio, batch_id)

    log.info(f"Total Simulation Time: {time.time()-overall_start:.2f}s")
    log.info("=" * 80)

# --- Helper Functions (Exits, Entries, P&L) ---

def _manage_exits_for_day(portfolio: dict, point_in_time: pd.Timestamp, data_cache: dict, batch_id: str):
    """
    MODIFIED: Manages exits with a new, sophisticated trailing stop-loss logic.
    """
    for position in portfolio['open_positions'][:]:
        ticker = position['ticker']
        try:
            day_data = data_cache[ticker].loc[point_in_time]
        except (KeyError, IndexError):
            continue

        close_reason, closing_price = None, None
        open_price, high_price, low_price, close_price = day_data['open'], day_data['high'], day_data['low'], day_data['close']
        
        # Initialize trailing_stop for this iteration
        trailing_stop = position.get('trailing_stop', position['stop_loss'])
        stop_loss = position['stop_loss']
        target = position['target']
        
        # --- Trailing Stop Logic ---
        use_trailing_stop = config.APEX_SWING_STRATEGY.get('use_trailing_stop', False)
        
        if use_trailing_stop:
            initial_risk_per_share = abs(position['entry_price'] - position['stop_loss'])
            activation_threshold = config.APEX_SWING_STRATEGY.get('trailing_stop_activation_r', 0) * initial_risk_per_share
            
            should_trail = False
            if position['signal'] == 'BUY' and close_price > position['entry_price'] + activation_threshold:
                should_trail = True
            elif position['signal'] == 'SHORT' and close_price < position['entry_price'] - activation_threshold:
                should_trail = True

            if should_trail:
                try:
                    atr_df = data_cache[ticker].loc[:point_in_time].copy()
                    atr_df.ta.atr(length=14, append=True)
                    current_atr = atr_df['ATRr_14'].iloc[-1]
                    trail_amount = current_atr * config.APEX_SWING_STRATEGY['trailing_stop_atr_multiplier']

                    if position['signal'] == 'BUY':
                        new_trail = close_price - trail_amount
                        if new_trail > trailing_stop:
                            position['trailing_stop'] = new_trail
                            trailing_stop = new_trail
                    elif position['signal'] == 'SHORT':
                        new_trail = close_price + trail_amount
                        if new_trail < trailing_stop:
                            position['trailing_stop'] = new_trail
                            trailing_stop = new_trail
                except Exception as e:
                    log.warning(f"Could not calculate trailing stop for {ticker}: {e}")

        # --- Exit Priority ---
        if position['signal'] == 'BUY':
            if open_price <= stop_loss:
                close_reason, closing_price = "Stop-Loss Hit (Gap Down)", open_price
            elif low_price <= trailing_stop:
                close_reason, closing_price = "Trailing Stop Hit", trailing_stop
            elif low_price <= stop_loss:
                close_reason, closing_price = "Stop-Loss Hit (Intraday)", stop_loss
            elif target and high_price >= target:
                close_reason, closing_price = "Target Hit", target
        elif position['signal'] == 'SHORT':
            if open_price >= stop_loss:
                close_reason, closing_price = "Stop-Loss Hit (Gap Up)", open_price
            elif high_price >= trailing_stop:
                close_reason, closing_price = "Trailing Stop Hit", trailing_stop
            elif high_price >= stop_loss:
                close_reason, closing_price = "Stop-Loss Hit (Intraday)", stop_loss
            elif target and low_price <= target:
                close_reason, closing_price = "Target Hit", target

        if not close_reason and (point_in_time - position['open_date']).days >= position['holding_period']:
            close_reason, closing_price = "Time Exit", close_price

        if close_reason:
            closed_trade_doc, net_pnl = _calculate_closed_trade(position, closing_price, close_reason, point_in_time)
            log.info(f"[EXIT] Closing {ticker} ({position['signal']}) → {close_reason}. Net P&L: ₹{net_pnl:.2f}")
            
            portfolio['equity'] += net_pnl
            portfolio['closed_trades'].append(closed_trade_doc)
            portfolio['open_positions'].remove(position)
            database_manager.performance_collection.insert_one(closed_trade_doc)

def _manage_entries_for_day(portfolio: dict, signals_for_today: list, point_in_time: pd.Timestamp, data_cache: dict):
    """
    MODIFIED: Manages entries with a new, conviction-based dynamic position sizing model.
    """
    # Sort signals by the absolute value of their score to prioritize highest conviction
    sorted_signals = sorted(signals_for_today, key=lambda x: abs(x.get('scrybeScore', 0)), reverse=True)
    
    for signal in sorted_signals:
        if len(portfolio['open_positions']) >= PORTFOLIO_CONSTRAINTS['max_concurrent_trades']:
            log.warning("[ENTRY] Skipping further signals: Max concurrent trade limit reached.")
            break
        if any(p['ticker'] == signal['ticker'] for p in portfolio['open_positions']):
            log.debug(f"[ENTRY] Skipping {signal['ticker']}: Position already open.")
            continue

        ticker = signal['ticker']
        trade_plan = signal.get('tradePlan', {})
        stop_loss_price = trade_plan.get('stopLoss')

        if not stop_loss_price:
            log.warning(f"[ENTRY] Skipping {ticker}: Signal has no stop-loss price.")
            continue

        try:
            day_data = data_cache[ticker].loc[point_in_time]
            actual_entry_price = day_data['open']
        except (KeyError, IndexError):
            log.warning(f"[ENTRY] Skipping {ticker}: No simulation data available for today.")
            continue

        # Pre-trade gap risk check (no change here)
        if signal['signal'] == 'BUY' and actual_entry_price <= stop_loss_price:
            log.warning(f"[ENTRY] VETO (BUY): {ticker} gapped down at open below stop-loss.")
            continue
        elif signal['signal'] == 'SHORT' and actual_entry_price >= stop_loss_price:
            log.warning(f"[ENTRY] VETO (SHORT): {ticker} gapped up at open above stop-loss.")
            continue
        
        # --- DYNAMIC POSITION SIZING LOGIC ---
        scrybe_score = signal.get('scrybeScore', 0)
        base_risk_pct = config.BACKTEST_PORTFOLIO_CONFIG['risk_per_trade_pct']
        
        # Define conviction tiers and corresponding risk multipliers
        if abs(scrybe_score) >= 70:      # Very High Conviction
            risk_multiplier = 1.0       # Risk the full 1%
            conviction_level = "VERY HIGH"
        elif abs(scrybe_score) >= 40:    # High Conviction
            risk_multiplier = 0.75      # Risk 0.75%
            conviction_level = "HIGH"
        else:                            # Medium Conviction (score is between 25-39)
            risk_multiplier = 0.5       # Risk only 0.5%
            conviction_level = "MEDIUM"
            
        final_risk_pct = base_risk_pct * risk_multiplier
        risk_amount = portfolio['equity'] * (final_risk_pct / 100.0)
        log.info(f"Conviction for {ticker} is {conviction_level} (Score: {scrybe_score}). Adjusting risk to {final_risk_pct:.2f}%.")
        # --- *** END OF NEW LOGIC *** ---

        risk_per_share = abs(actual_entry_price - stop_loss_price)
        if risk_per_share <= 0.01:
            log.warning(f"[ENTRY] Skipping {ticker}: Risk per share is zero.")
            continue
            
        num_shares = int(risk_amount / risk_per_share)
        
        if num_shares == 0:
            log.warning(f"[ENTRY] Skipping {ticker}: Position size is zero shares.")
            continue

        new_position = {
            'prediction_id': signal['_id'], 'ticker': ticker, 'signal': signal['signal'],
            'entry_price': actual_entry_price, 'num_shares': num_shares, 'stop_loss': stop_loss_price,
            'target': trade_plan.get('target'), 'open_date': point_in_time,
            'holding_period': config.APEX_SWING_STRATEGY['holding_period'],
            'strategy': signal['strategy'], 'batch_id': signal['batch_id']
        }
        portfolio['open_positions'].append(new_position)
        log.info(f"[ENTRY] {signal['signal']} {num_shares} shares of {ticker} @ OPEN {actual_entry_price:.2f}")

def _calculate_closed_trade(position: dict, closing_price: float, closing_reason: str, close_date: pd.Timestamp):
    entry_price = position['entry_price']
    num_shares = position['num_shares']
    signal = position['signal']

    if signal == 'BUY':
        gross_pnl = (closing_price - entry_price) * num_shares
    elif signal == 'SHORT':
        gross_pnl = (entry_price - closing_price) * num_shares
    else:
        gross_pnl = 0 # Or handle other signals if they exist

    costs = config.BACKTEST_CONFIG
    turnover = (entry_price * num_shares) + (closing_price * num_shares)
    brokerage = turnover * (costs['brokerage_pct'] / 100.0)
    stt = turnover * (costs['stt_pct'] / 100.0)
    other_charges = turnover * (costs['slippage_pct'] / 100.0)
    total_transaction_costs = brokerage + stt + other_charges
    net_pnl = gross_pnl - total_transaction_costs

    initial_investment = entry_price * num_shares
    net_return_pct = (net_pnl / initial_investment) * 100 if initial_investment != 0 else 0

    performance_doc = {
        "prediction_id": position['prediction_id'], "ticker": position['ticker'],
        "strategy": position['strategy'], "signal": signal, "status": "Closed",
        "open_date": position['open_date'].to_pydatetime(), "close_date": close_date.to_pydatetime(),
        "closing_reason": closing_reason,
        "net_pnl": round(net_pnl, 2),
        "net_return_pct": round(net_return_pct, 2),
        "batch_id": position['batch_id']
    }
    return performance_doc, net_pnl

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
