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
from ai_analyzer import AIAnalyzer
import uuid
from utils import APIKeyManager
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
    with open(STATE_FILE, 'w') as f: json.dump({'next_start_date': next_day_to_run.strftime('%Y-%m-%d')}, f)

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

    Added: optional 'tickers' argument — if provided, backtest will use this static list
    for every day (useful for targeted backtests).
    """
    log.info(f"### STARTING UNIFIED SIMULATION FOR BATCH: {batch_id} ###")
    log.info(f"Period: {start_date} to {end_date}")

    pipeline = AnalysisPipeline()
    pipeline._setup(mode='scheduler')

    if is_fresh_run:
        log.warning("FRESH RUN: Deleting previous predictions and performance data for this batch.")
        # guard against missing collection handles
        if getattr(database_manager, "predictions_collection", None) is not None:
            database_manager.predictions_collection.delete_many({"batch_id": batch_id})
        if getattr(database_manager, "performance_collection", None) is not None:
            database_manager.performance_collection.delete_many({"batch_id": batch_id})

    # --- 1. PORTFOLIO & SETUP (NO DATA PRE-LOADING) ---
    portfolio = {
        'equity': config.BACKTEST_PORTFOLIO_CONFIG['initial_capital'],
        'open_positions': [],
        'closed_trades': [],
        'daily_equity_log': []
    }
    
    # We only load the historical constituents list. All market data is loaded on-demand.
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'nifty50_historical_constituents.csv')
        constituents_df = pd.read_csv(csv_path)
        constituents_df['date'] = pd.to_datetime(constituents_df['date'])
        log.info("✅ Historical constituents list loaded.")
    except Exception as e:
        log.fatal(f"Failed during setup: Could not load constituents file. Error: {e}")
        return

    # --- 2. DAILY SIMULATION LOOP ---
    simulation_days = pd.bdate_range(start=start_date, end=end_date)

    for i, current_day in enumerate(simulation_days):
        day_str = current_day.strftime('%Y-%m-%d')
        log.info(f"\n--- Simulating Day {i+1}/{len(simulation_days)}: {day_str} ---")

        # --- USE 'tickers' ARGUMENT IF PROVIDED ---
        if tickers:
            stock_universe_for_today = tickers
            log.info(f"Using predefined stock universe of {len(tickers)} tickers.")
        else:
            # Determine the correct stock universe for THIS specific day using index_manager
            stock_universe_for_today = index_manager.get_point_in_time_nifty50_tickers(current_day)

        if not stock_universe_for_today:
            log.warning(f"Could not determine stock universe for {day_str}. Skipping day.")
            continue
        
        # Build the list of all tickers needed JUST FOR TODAY
        required_indices = list(sector_analyzer.CORE_SECTOR_INDICES.values()) + [sector_analyzer.BENCHMARK_INDEX] + list(BENCHMARK_TICKERS.values())
        tickers_for_today = list(set(stock_universe_for_today + required_indices))
        
        # Load ONLY the data needed for today.
        # IMPORTANT BUG FIX: pass the point-in-time day_str as end_date so data_retriever returns correct point-in-time slices.
        log.info(f"Loading data for {len(tickers_for_today)} assets for {day_str}...")
        data_cache_for_today = {
            ticker: data_retriever.get_historical_stock_data(ticker, end_date=day_str)
            for ticker in tickers_for_today
        }
        data_cache_for_today = {k: v for k, v in data_cache_for_today.items() if v is not None and not v.empty}
        
        # A. MANAGE EXITS: Check existing positions against today's market data
        _manage_exits_for_day(portfolio, current_day, data_cache_for_today, batch_id)

        # B. GENERATE NEW SIGNALS: Run the full pipeline for today
        pipeline.run(
            point_in_time=current_day,
            full_data_cache=data_cache_for_today,
            is_backtest=True,
            batch_id=batch_id
        )

        # C. MANAGE ENTRIES: Fetch signals just generated by the pipeline and enter new positions
        signals_for_today_from_db = list(database_manager.predictions_collection.find({
            "batch_id": batch_id, 
            "prediction_date": current_day.to_pydatetime(),
            "signal": {"$in": ["BUY", "SELL"]}
        }))
        if signals_for_today_from_db:
            _manage_entries_for_day(portfolio, signals_for_today_from_db, current_day)
        
        # D. LOG DAILY EQUITY
        open_positions_value = 0
        for pos in portfolio['open_positions']:
            try:
                # Use today's close price to value open positions
                current_price = data_cache_for_today[pos['ticker']].loc[current_day]['close']
                open_positions_value += current_price * pos['num_shares']
            except (KeyError, IndexError):
                # If market is closed or data is missing, use the last known entry price
                open_positions_value += pos['entry_price'] * pos['num_shares']
        
        # Equity available for new trades + value of open positions
        total_equity = portfolio['equity'] + open_positions_value
        portfolio['daily_equity_log'].append({'date': current_day, 'equity': total_equity})
        log.info(f"End of Day Equity: ₹{total_equity:,.2f}")

    # --- 3. FINAL REPORTING ---
    log.info("\n--- ✅ UNIFIED SIMULATION FINISHED! Generating Final Report... ---")

    if not portfolio['daily_equity_log']:
        log.warning("No equity data logged. Skipping final report.")
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
            sharpe_ratio = (equity_df['daily_return'].mean() / equity_df['daily_return'].std()) * (252**0.5) if equity_df['daily_return'].std() > 0 else 0
        else:
            win_rate, profit_factor, sharpe_ratio = 0, 0, 0
            
        print("\n" + "="*80)
        print(f"### Backtest Performance Summary: Batch '{batch_id}' ###")
        print("="*80)
        print(f"{'Total Return:':<25} {total_return_pct:.2f}%")
        print(f"{'Final Portfolio Value:':<25} ₹{final_equity:,.2f}")
        print(f"{'Max Drawdown:':<25} {max_drawdown_pct:.2f}%")
        print(f"{'Sharpe Ratio (Annualized):':<25} {sharpe_ratio:.2f}")
        print("-" * 80)
        print(f"{'Total Trades Closed:':<25} {total_trades}")
        print(f"{'Win Rate:':<25} {win_rate:.2f}%")
        print(f"{'Profit Factor:':<25} {profit_factor:.2f}")
        print("="*80)
    
    pipeline.close()

def _manage_exits_for_day(portfolio: dict, point_in_time: pd.Timestamp, data_cache: dict, batch_id: str):
    """Checks all open positions for stop-loss, target, or time-based exits."""
    for position in portfolio['open_positions'][:]: # Iterate on a copy
        ticker = position['ticker']
        try:
            day_data = data_cache[ticker].loc[point_in_time]
        except KeyError:
            continue # Market closed for this stock today

        close_reason, closing_price = None, None

        if position['signal'] == 'BUY':
            if day_data.open <= position['stop_loss']:
                close_reason, closing_price = "Stop-Loss Hit (Gap Down)", day_data.open
            elif day_data.low <= position['stop_loss']:
                close_reason, closing_price = "Stop-Loss Hit (Intraday)", position['stop_loss']
            elif position['target'] and day_data.high >= position['target']:
                close_reason, closing_price = "Target Hit", position['target']
        
        elif position['signal'] == 'SELL': # <-- NEW LOGIC FOR SELLS
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
            log.info(f"CLOSING TRADE: {ticker} ({position['signal']}) for {close_reason}. Net P&L: ₹{net_pnl:.2f}")
            
            portfolio['equity'] += net_pnl
            portfolio['closed_trades'].append(closed_trade_doc)
            portfolio['open_positions'].remove(position)
            
            database_manager.performance_collection.insert_one(closed_trade_doc)

def _manage_entries_for_day(portfolio: dict, signals_for_today: list, point_in_time: pd.Timestamp):
    """Checks for new signals and enters positions if portfolio capacity allows."""
    # Sort signals by Scrybe Score to prioritize the highest conviction trades
    sorted_signals = sorted(signals_for_today, key=lambda x: x.get('scrybeScore', 0), reverse=True)

    for signal in sorted_signals:
        if len(portfolio['open_positions']) >= config.PORTFOLIO_CONSTRAINTS['max_concurrent_trades']:
            log.warning("SKIPPING SIGNAL: Max concurrent trade limit reached.")
            break # Stop trying to enter new trades
        
        if any(p['ticker'] == signal['ticker'] for p in portfolio['open_positions']):
            continue # Already have a position in this ticker

        trade_plan = signal.get('tradePlan', {})
        entry_price, stop_loss_price = trade_plan.get('entryPrice'), trade_plan.get('stopLoss')
        
        if not all([entry_price, stop_loss_price]): continue

        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share <= 0: continue

        risk_amount = portfolio['equity'] * (config.BACKTEST_PORTFOLIO_CONFIG['risk_per_trade_pct'] / 100.0)
        num_shares = int(risk_amount / risk_per_share)
        if num_shares == 0: continue

        new_position = {
            'prediction_id': signal['_id'], 'ticker': signal['ticker'], 'signal': signal['signal'],
            'entry_price': entry_price, 'num_shares': num_shares, 'stop_loss': stop_loss_price,
            'target': trade_plan.get('target'), 'open_date': point_in_time, 
            'holding_period': config.APEX_SWING_STRATEGY['holding_period'],
            'strategy': signal['strategy'], 'batch_id': signal['batch_id']
        }
        portfolio['open_positions'].append(new_position)
        log.info(f"ENTERING TRADE: {signal['signal']} {num_shares} shares of {signal['ticker']} @ {entry_price:.2f}")

def _calculate_closed_trade(position: dict, closing_price: float, closing_reason: str, close_date: pd.Timestamp):
    """Calculates the P&L of a closed trade and returns a performance document."""
    entry_price = position['entry_price']
    num_shares = position['num_shares']
    signal = position['signal']

    if signal == 'BUY':
        gross_pnl = (closing_price - entry_price) * num_shares
    else: # SELL
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

# --- MAIN EXECUTION BLOCK (CLI) ---
if __name__ == "__main__":
    # This block allows the script to be run from the command line.
    parser = argparse.ArgumentParser(description="Run a full backtest using the main orchestrator.")
    
    # These arguments will be provided by your GitHub Actions workflow or CLI
    parser.add_argument("--batch_id", required=True, help="Unique ID for the backtest batch.")
    parser.add_argument("--start_date", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end_date", required=True, help="End date in YYYY-MM-DD format.")
    
    # The 'fresh_run' argument is converted from a string 'true'/'false' to a boolean
    parser.add_argument('--fresh_run', type=lambda x: (str(x).lower() == 'true'), default=True, help='Delete previous data for this batch_id? (true/false)')
    
    # Optional argument for a targeted backtest on specific stocks
    parser.add_argument("--tickers", nargs='+', required=False, help="Optional: Space-separated list of specific tickers to backtest.")

    args = parser.parse_args()

    # Call the main simulation function with all the parsed arguments
    run_simulation(
        batch_id=args.batch_id,
        start_date=args.start_date,
        end_date=args.end_date,
        is_fresh_run=args.fresh_run,
        tickers=args.tickers  # Pass the list of tickers (will be None if not provided)
    )
