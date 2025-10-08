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
import market_regime_analyzer
import sector_analyzer
import quantitative_screener
from config import PORTFOLIO_CONSTRAINTS
from collections import deque
from sector_analyzer import CORE_SECTOR_INDICES, BENCHMARK_INDEX
import random
import technical_analyzer
import index_manager
from analysis_pipeline import AnalysisPipeline, BENCHMARK_TICKERS

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

            data_load_start = time.time()
            data_cache_for_today = {
                ticker: data_retriever.get_historical_stock_data(ticker, end_date=day_str)
                for ticker in tickers_for_today
            }
            data_cache_for_today = {k: v for k, v in data_cache_for_today.items() if v is not None and not v.empty}
            log.info(f"[DATA] Loaded {len(data_cache_for_today)} tickers in {time.time()-data_load_start:.2f}s")

            # --- Manage Exits ---
            exits_start = time.time()
            before_exits = len(portfolio['open_positions'])
            _manage_exits_for_day(portfolio, current_day, data_cache_for_today, batch_id)
            closed_today = before_exits - len(portfolio['open_positions'])
            log.info(f"[EXIT] {closed_today} positions closed in {time.time()-exits_start:.2f}s")

            # --- Run Analysis Pipeline ---
            pipeline_start = time.time()
            pipeline.run(
                point_in_time=current_day,
                full_data_cache=data_cache_for_today,
                is_backtest=True,
                batch_id=batch_id
            )
            log.info(f"[PIPELINE] AnalysisPipeline.run() completed in {time.time()-pipeline_start:.2f}s")

            # --- Manage Entries ---
            entries_start = time.time()
            signals_for_today = list(database_manager.predictions_collection.find({
                "batch_id": batch_id,
                "prediction_date": current_day.to_pydatetime(),
                "signal": {"$in": ["BUY", "SELL"]}
            }))
            before_entries = len(portfolio['open_positions'])
            _manage_entries_for_day(portfolio, signals_for_today, current_day)
            opened_today = len(portfolio['open_positions']) - before_entries
            log.info(f"[ENTRY] {len(signals_for_today)} signals processed, {opened_today} trades opened in {time.time()-entries_start:.2f}s")

            # --- Equity Logging ---
            valuation_start = time.time()
            open_value = 0
            for pos in portfolio['open_positions']:
                try:
                    current_price = data_cache_for_today[pos['ticker']].loc[current_day]['close']
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

    try:
        if not portfolio['daily_equity_log']:
            log.warning("[REPORT] No equity data logged. Skipping final report.")
        else:
            equity_df = pd.DataFrame(portfolio['daily_equity_log']).set_index('date')
            closed_trades_df = pd.DataFrame(portfolio['closed_trades'])
            initial_capital = config.BACKTEST_PORTFOLIO_CONFIG['initial_capital']
            final_equity = equity_df['equity'].iloc[-1]
            total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100

            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown_pct'] = ((equity_df['peak'] - equity_df['equity']) / equity_df['peak']) * 100
            max_drawdown_pct = equity_df['drawdown_pct'].max() if not equity_df['drawdown_pct'].empty else 0
            total_trades = len(closed_trades_df)

            if total_trades > 0:
                win_rate = (closed_trades_df['net_pnl'] > 0).mean() * 100
                gross_profit = closed_trades_df[closed_trades_df['net_pnl'] > 0]['net_pnl'].sum()
                gross_loss = abs(closed_trades_df[closed_trades_df['net_pnl'] < 0]['net_pnl'].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                equity_df['daily_return'] = equity_df['equity'].pct_change().fillna(0)
                sharpe_ratio = (equity_df['daily_return'].mean() / equity_df['daily_return'].std()) * (252 ** 0.5) if equity_df['daily_return'].std() > 0 else 0
            else:
                win_rate = profit_factor = sharpe_ratio = 0

            print("\n" + "=" * 80)
            print(f"### Backtest Performance Summary: Batch '{batch_id}' ###")
            print("=" * 80)
            print(f"{'Total Return:':<25} {total_return_pct:.2f}%")
            print(f"{'Final Portfolio Value:':<25} ₹{final_equity:,.2f}")
            print(f"{'Max Drawdown:':<25} {max_drawdown_pct:.2f}%")
            print(f"{'Sharpe Ratio (Annualized):':<25} {sharpe_ratio:.2f}")
            print("-" * 80)
            print(f"{'Total Trades Closed:':<25} {total_trades}")
            print(f"{'Win Rate:':<25} {win_rate:.2f}%")
            print(f"{'Profit Factor:':<25} {profit_factor:.2f}")
            print("=" * 80)

        log.info(f"Total Simulation Time: {time.time()-overall_start:.2f}s")
        log.info("=" * 80)
    except Exception as e:
        log.error(f"[FINAL REPORT ERROR] {e}", exc_info=True)

    pipeline.close()

# --- Helper Functions (Exits, Entries, P&L) ---

def _manage_exits_for_day(portfolio: dict, point_in_time: pd.Timestamp, data_cache: dict, batch_id: str):
    for position in portfolio['open_positions'][:]:
        ticker = position['ticker']
        try:
            day_data = data_cache[ticker].loc[point_in_time]
        except KeyError:
            continue

        close_reason, closing_price = None, None

        if position['signal'] == 'BUY':
            if day_data.open <= position['stop_loss']:
                close_reason, closing_price = "Stop-Loss Hit (Gap Down)", day_data.open
            elif day_data.low <= position['stop_loss']:
                close_reason, closing_price = "Stop-Loss Hit (Intraday)", position['stop_loss']
            elif position['target'] and day_data.high >= position['target']:
                close_reason, closing_price = "Target Hit", position['target']
        elif position['signal'] == 'SELL':
            if day_data.open >= position['stop_loss']:
                close_reason, closing_price = "Stop-Loss Hit (Gap Up)", day_data.open
            elif day_data.high >= position['stop_loss']:
                close_reason, closing_price = "Stop-Loss Hit (Intraday)", position['stop_loss']
            elif position['target'] and day_data.low <= position['target']:
                close_reason, closing_price = "Target Hit", position['target']

        if not close_reason and (point_in_time - position['open_date']).days >= position['holding_period']:
            close_reason, closing_price = "Time Exit", day_data.close

        if close_reason:
            closed_trade_doc, net_pnl = _calculate_closed_trade(position, closing_price, close_reason, point_in_time)
            log.info(f"[EXIT] Closing {ticker} ({position['signal']}) → {close_reason}. Net P&L: ₹{net_pnl:.2f}")
            portfolio['equity'] += net_pnl
            portfolio['closed_trades'].append(closed_trade_doc)
            portfolio['open_positions'].remove(position)
            database_manager.performance_collection.insert_one(closed_trade_doc)

def _manage_entries_for_day(portfolio: dict, signals_for_today: list, point_in_time: pd.Timestamp):
    sorted_signals = sorted(signals_for_today, key=lambda x: x.get('scrybeScore', 0), reverse=True)
    for signal in sorted_signals:
        if len(portfolio['open_positions']) >= PORTFOLIO_CONSTRAINTS['max_concurrent_trades']:
            log.warning("[ENTRY] Skipping: Max concurrent trade limit reached.")
            break
        if any(p['ticker'] == signal['ticker'] for p in portfolio['open_positions']):
            continue

        trade_plan = signal.get('tradePlan', {})
        entry_price, stop_loss_price = trade_plan.get('entryPrice'), trade_plan.get('stopLoss')
        if not all([entry_price, stop_loss_price]):
            continue

        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share <= 0:
            continue

        risk_amount = portfolio['equity'] * (config.BACKTEST_PORTFOLIO_CONFIG['risk_per_trade_pct'] / 100.0)
        num_shares = int(risk_amount / risk_per_share)
        if num_shares == 0:
            continue

        new_position = {
            'prediction_id': signal['_id'], 'ticker': signal['ticker'], 'signal': signal['signal'],
            'entry_price': entry_price, 'num_shares': num_shares, 'stop_loss': stop_loss_price,
            'target': trade_plan.get('target'), 'open_date': point_in_time,
            'holding_period': config.APEX_SWING_STRATEGY['holding_period'],
            'strategy': signal['strategy'], 'batch_id': signal['batch_id']
        }
        portfolio['open_positions'].append(new_position)
        log.info(f"[ENTRY] {signal['signal']} {num_shares} shares of {signal['ticker']} @ {entry_price:.2f}")

def _calculate_closed_trade(position: dict, closing_price: float, closing_reason: str, close_date: pd.Timestamp):
    entry_price = position['entry_price']
    num_shares = position['num_shares']
    signal = position['signal']

    if signal == 'BUY':
        gross_pnl = (closing_price - entry_price) * num_shares
    else:
        gross_pnl = (entry_price - closing_price) * num_shares

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
