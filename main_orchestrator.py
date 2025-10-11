# main_orchestrator.py
import pandas as pd
import time
import config
from logger_config import log
import database_manager
import json
import os
import data_retriever
import pandas_ta as ta
import uuid
import argparse
from config import PORTFOLIO_CONSTRAINTS
from collections import deque
from sector_analyzer import CORE_SECTOR_INDICES, BENCHMARK_INDEX
import random
import index_manager
from analysis_pipeline import AnalysisPipeline, BENCHMARK_TICKERS
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

def run_simulation(batch_id: str, start_date: str, end_date: str, is_fresh_run: bool = False, tickers: list = None):
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
        day_start = time.time()
        day_str = current_day.strftime('%Y-%m-%d')
        log.info("")
        log.info("-" * 80)
        log.info(f"--- Simulating Day {i+1}/{total_days}: {day_str} ---")

        try:
            # --- Determine Stock Universe ---
            if tickers:
                stock_universe_for_today = tickers
                log.info(f"[UNIVERSE] Using predefined stock universe of {len(tickers)} tickers.")
            else:
                stock_universe_for_today = index_manager.get_point_in_time_nifty50_tickers(current_day)
                if not stock_universe_for_today:
                    log.warning(f"[UNIVERSE] No tickers found for {day_str}. Skipping day.")
                    continue

            # --- Data Loading ---
            required_indices = list(CORE_SECTOR_INDICES.values()) + [BENCHMARK_INDEX] + list(BENCHMARK_TICKERS.values())
            tickers_for_today = list(set(stock_universe_for_today + required_indices))
            log.info(f"[DATA] Loading {len(tickers_for_today)} assets for {day_str} ...")
            
            # The date used to fetch data for making the trading decision.
            decision_date = current_day - pd.Timedelta(days=1)
            decision_date_str = decision_date.strftime('%Y-%m-%d')
            
            data_load_start = time.time()
            
            # This cache is for the AI/Pipeline. It cannot see today's data.
            data_for_analysis = {
                ticker: data_retriever.get_historical_stock_data(ticker, end_date=decision_date_str)
                for ticker in tickers_for_today
            }
            data_for_analysis = {k: v for k, v in data_for_analysis.items() if v is not None and not v.empty}
            log.info(f"[DATA] Loaded {len(data_for_analysis)} tickers for ANALYSIS (up to {decision_date_str})")

            # This cache is for the backtester to simulate today's trading activity.
            data_for_simulation = {
                ticker: data_retriever.get_historical_stock_data(ticker, end_date=day_str)
                for ticker in tickers_for_today
            }
            data_for_simulation = {k: v for k, v in data_for_simulation.items() if v is not None and not v.empty}
            log.info(f"[DATA] Loaded {len(data_for_simulation)} tickers for SIMULATION (up to {day_str})")
            
            log.info(f"[DATA] Full data loading completed in {time.time()-data_load_start:.2f}s")

            # --- Manage Exits ---
            exits_start = time.time()
            before_exits = len(portfolio['open_positions'])
            # Pass the SIMULATION data cache which contains the current day's prices
            _manage_exits_for_day(portfolio, current_day, data_for_simulation, batch_id)
            closed_today = before_exits - len(portfolio['open_positions'])
            log.info(f"[EXIT] {closed_today} positions closed in {time.time()-exits_start:.2f}s")

            # --- Run Analysis Pipeline ---
            pipeline_start = time.time()
            # Pass the ANALYSIS data cache which ONLY has data up to the previous day
            pipeline.run(
                point_in_time=decision_date, # The decision is made based on the previous day's close
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
                "signal": {"$in": ["BUY", "SHORT"]} # FIX: Replaced "SELL" with "SHORT"
            }))
            before_entries = len(portfolio['open_positions'])
            _manage_entries_for_day(portfolio, signals_for_today, current_day, data_for_simulation)
            opened_today = len(portfolio['open_positions']) - before_entries
            log.info(f"[ENTRY] {len(signals_for_today)} signals processed, {opened_today} trades opened in {time.time()-entries_start:.2f}s")

            # --- Equity Logging ---
            valuation_start = time.time()
            open_value = 0
            for pos in portfolio['open_positions']:
                try:
                    # Use the SIMULATION data cache for valuation
                    current_price = data_for_simulation[pos['ticker']].loc[current_day]['close']
                    open_value += current_price * pos['num_shares']
                except (KeyError, IndexError):
                    open_value += pos['entry_price'] * pos['num_shares']

            total_equity = portfolio['equity'] + open_value
            portfolio['daily_equity_log'].append({'date': current_day, 'equity': total_equity})
            log.info(f"[PORTFOLIO] End-of-Day Equity: ₹{total_equity:,.2f} (valuation time {time.time()-valuation_start:.2f}s)")

            # --- Day Complete ---
            log.info(f"--- Day {i+1}/{total_days} completed in {time.time()-day_start:.2f}s ---")
        except Exception as e:
            log.error(f"[ERROR] Exception during simulation of {day_str}: {e}", exc_info=True)
            continue

    # --- Final Report ---
    log.info("")
    log.info("=" * 80)
    log.info("--- ✅ UNIFIED SIMULATION FINISHED! Generating Final Report... ---")
    
    # All reporting logic is now handled by the dedicated analyzer.
    generate_backtest_report(portfolio, batch_id)

    log.info(f"Total Simulation Time: {time.time()-overall_start:.2f}s")
    log.info("=" * 80)

# --- Helper Functions (Exits, Entries, P&L) ---

def _manage_exits_for_day(portfolio: dict, point_in_time: pd.Timestamp, data_cache: dict, batch_id: str):
    """
    Manages exits for the current day using a realistic, sequential logic.
    1. Checks for a gap at the open against the stop-loss.
    2. Assumes Stop-Loss is hit before Target intraday (conservative approach).
    3. Checks for the target being hit.
    4. Checks for time-based exit at the close.
    """
    for position in portfolio['open_positions'][:]:
        ticker = position['ticker']
        try:
            day_data = data_cache[ticker].loc[point_in_time]
        except (KeyError, IndexError):
            continue # No data for this stock today, hold position and check tomorrow.

        close_reason, closing_price = None, None
        open_price = day_data['open']
        high_price = day_data['high']
        low_price = day_data['low']
        close_price = day_data['close']
        stop_loss = position['stop_loss']
        target = position['target']

        # --- SEQUENTIAL EXIT LOGIC ---
        if position['signal'] == 'BUY':
            # Priority 1: Check for gap down at open hitting the stop-loss.
            if open_price <= stop_loss:
                close_reason, closing_price = "Stop-Loss Hit (Gap Down)", open_price
            # Priority 2: Check for intraday stop-loss hit (conservative: stop is always prioritized).
            elif low_price <= stop_loss:
                close_reason, closing_price = "Stop-Loss Hit (Intraday)", stop_loss
            # Priority 3: If no stop was hit, check for target.
            elif target and high_price >= target:
                close_reason, closing_price = "Target Hit", target
        
        elif position['signal'] == 'SHORT':
            # Priority 1: Check for gap up at open hitting the stop-loss.
            if open_price >= stop_loss:
                close_reason, closing_price = "Stop-Loss Hit (Gap Up)", open_price
            # Priority 2: Check for intraday stop-loss hit.
            elif high_price >= stop_loss:
                close_reason, closing_price = "Stop-Loss Hit (Intraday)", stop_loss
            # Priority 3: If no stop was hit, check for target.
            elif target and low_price <= target:
                close_reason, closing_price = "Target Hit", target

        # Priority 4: If no price-based exit was triggered, check for time-based exit.
        if not close_reason and (point_in_time - position['open_date']).days >= position['holding_period']:
            close_reason, closing_price = "Time Exit", close_price

        # If any exit condition was met, process the trade closure.
        if close_reason:
            closed_trade_doc, net_pnl = _calculate_closed_trade(position, closing_price, close_reason, point_in_time)
            log.info(f"[EXIT] Closing {ticker} ({position['signal']}) → {close_reason}. Net P&L: ₹{net_pnl:.2f}")
            
            # Using your original method of updating total equity with the PnL of the closed trade.
            portfolio['equity'] += net_pnl
            portfolio['closed_trades'].append(closed_trade_doc)
            portfolio['open_positions'].remove(position)
            database_manager.performance_collection.insert_one(closed_trade_doc)

def _manage_entries_for_day(portfolio: dict, signals_for_today: list, point_in_time: pd.Timestamp, data_cache: dict):
    """
    Manages entries for the current simulation day using the OPEN price.
    - Uses the actual open price for trade entry.
    - Performs a pre-trade gap-risk check.
    - Recalculates position size based on the actual entry price.
    """
    sorted_signals = sorted(signals_for_today, key=lambda x: x.get('scrybeScore', 0), reverse=True)
    
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

        # --- REALISTIC ENTRY LOGIC ---
        try:
            day_data = data_cache[ticker].loc[point_in_time]
            actual_entry_price = day_data['open']
        except (KeyError, IndexError):
            log.warning(f"[ENTRY] Skipping {ticker}: No simulation data available for today.")
            continue

        # --- PRE-TRADE RISK CHECK ---
        # If the market gaps against us at the open, abort the trade.
        if signal['signal'] == 'BUY' and actual_entry_price <= stop_loss_price:
            log.warning(f"[ENTRY] VETO (BUY): {ticker} gapped down at open ({actual_entry_price:.2f}) below stop-loss ({stop_loss_price:.2f}).")
            continue
        elif signal['signal'] == 'SHORT' and actual_entry_price >= stop_loss_price:
            log.warning(f"[ENTRY] VETO (SHORT): {ticker} gapped up at open ({actual_entry_price:.2f}) above stop-loss ({stop_loss_price:.2f}).")
            continue
        
        # --- DYNAMIC POSITION SIZING ---
        risk_per_share = abs(actual_entry_price - stop_loss_price)
        if risk_per_share <= 0.01: # Avoid division by zero
            log.warning(f"[ENTRY] Skipping {ticker}: Risk per share is zero or negative.")
            continue
            
        risk_amount = portfolio['equity'] * (config.BACKTEST_PORTFOLIO_CONFIG['risk_per_trade_pct'] / 100.0)
        num_shares = int(risk_amount / risk_per_share)
        
        if num_shares == 0:
            log.warning(f"[ENTRY] Skipping {ticker}: Position size is zero shares.")
            continue

        # --- CREATE POSITION ---
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
                        help="Optional: Space-separated list of tickers to backtest.")
    args = parser.parse_args()

    run_simulation(
        batch_id=args.batch_id,
        start_date=args.start_date,
        end_date=args.end_date,
        is_fresh_run=args.fresh_run,
        tickers=args.tickers
    )
