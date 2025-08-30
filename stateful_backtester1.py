# stateful_backtester.py

import pandas as pd
from logger_config import log
import database_manager
import data_retriever
import config
import argparse
from config import PORTFOLIO_CONSTRAINTS

def _calculate_closed_trade(position: dict, closing_price: float, closing_reason: str, close_date: pd.Timestamp, previous_day_data: pd.Series):
    """
    Calculates the P&L of a closed trade and returns a performance document.
    """
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

    # Use the prior day's ATR to remove lookahead bias
    atr_on_prior_day = previous_day_data.get('ATRr_14', 0)
    slippage_per_share = atr_on_prior_day * costs['slippage_atr_fraction']
    total_slippage_cost = (slippage_per_share * num_shares) * 2 # Multiply by 2 for entry and exit

    total_transaction_costs = brokerage + stt + total_slippage_cost
    net_pnl = gross_pnl - total_transaction_costs

    initial_investment = entry_price * num_shares
    net_return_pct = (net_pnl / initial_investment) * 100 if initial_investment != 0 else 0

    performance_doc = {
        "prediction_id": position['prediction_id'], "ticker": position['ticker'], 
        "strategy": position['strategy'], "signal": signal, "status": "Closed", 
        "open_date": position['open_date'], "close_date": close_date.to_pydatetime(), 
        "closing_reason": closing_reason, 
        "net_pnl": round(net_pnl, 2),
        "net_return_pct": round(net_return_pct, 2),
        "batch_id": position['batch_id']
    }
    return performance_doc, net_pnl

def run_stateful_backtest(batch_id: str):
    log.info(f"--- 🚀 Starting Stateful Backtest for Batch: '{batch_id}' ---")

    strategies_to_test = [
        config.STRATEGY_ATR_3R,
        config.STRATEGY_AI_ATR,
        config.STRATEGY_AI_STRUCTURE
    ]

    database_manager.init_db(purpose='scheduler')
    query = {"batch_id": batch_id, "signal": {"$in": ["BUY", "SELL"]}}
    signals_cursor = database_manager.predictions_collection.find(query)
    all_signals_df = pd.DataFrame(list(signals_cursor))

    if all_signals_df.empty:
        log.warning(f"No 'BUY' or 'SELL' signals found for batch '{batch_id}'. Nothing to simulate.")
        database_manager.close_db_connection()
        return

    log.info(f"Found {len(all_signals_df)} trade signals to simulate.")
    all_signals_df['prediction_date'] = pd.to_datetime(all_signals_df['prediction_date'])
    unique_tickers = all_signals_df['ticker'].unique().tolist()
    last_signal_date = all_signals_df['prediction_date'].max()
    
    # --- BUG FIX: Use .get() to provide a default holding period if one isn't specified ---
    default_holding_period = 10 
    max_holding_days = max(s.get('holding_period', default_holding_period) for s in strategies_to_test)
    simulation_end_date = last_signal_date + pd.Timedelta(days=max_holding_days + 15)
    
    data_cache = {}
    log.info(f"Pre-loading historical data for {len(unique_tickers)} ticker(s)...")
    for ticker in unique_tickers:
        data = data_retriever.get_historical_stock_data(ticker, end_date=simulation_end_date.strftime('%Y-%m-%d'))
        if data is not None and not data.empty:
            data_cache[ticker] = data
        else:
            log.warning(f"Could not load data for {ticker}. Signals for this ticker will be skipped.")
            all_signals_df = all_signals_df[all_signals_df['ticker'] != ticker]
            
    log.info("✅ Setup complete.")

    all_results_for_db = []
    for strategy in strategies_to_test:
        log.info(f"--- Running Simulation for Strategy: '{strategy['name']}' ---")

        # --- START OF NEW VETO LOGIC BLOCK ---

        # 1. Filter by Conviction Score
        min_score = strategy.get('min_conviction_score', 30)
        signals_for_this_strategy = all_signals_df[all_signals_df['scrybeScore'].abs() >= min_score].copy()
        log.info(f"  [Veto] Start with {len(all_signals_df)} signals, {len(signals_for_this_strategy)} pass conviction score >= {min_score}")

        # 2. Filter by Market Regime
        def is_regime_ok(row):
            regime = row['market_regime_at_prediction']
            signal = row['signal']
            if signal == 'BUY' and regime == 'Bearish':
                return False
            if signal == 'SELL' and regime == 'Bullish':
                return False
            return True

        signals_for_this_strategy = signals_for_this_strategy[signals_for_this_strategy.apply(is_regime_ok, axis=1)]
        log.info(f"  [Veto] {len(signals_for_this_strategy)} signals pass market regime check")

        # 3. Filter by Risk/Reward Ratio (using the logic from Fix #2)
        def calculate_strategy_rr(row):
            trade_plan = row.get('tradePlan', {})
            entry_price = trade_plan.get('entryPrice')
            if not entry_price: return 0

            # Determine the correct stop loss for THIS strategy
            stop_loss_price = 0
            if "ATR_3R" in strategy['name']:
                stop_loss_price = trade_plan.get('atr_stop_loss_A', 0)
            elif "AI_ATR_Stop" in strategy['name']:
                stop_loss_price = trade_plan.get('atr_stop_loss_B', 0)
            elif "AI_Structure_Stop" in strategy['name']:
                stop_loss_price = trade_plan.get('structure_stop_loss_C', 0)

            if stop_loss_price <= 0: return 0

            # Use AI's predicted gain for a universal reward check
            ai_predicted_gain = row.get('predicted_gain_pct', 0)
            potential_reward = entry_price * (ai_predicted_gain / 100.0)
            potential_risk = abs(entry_price - stop_loss_price)

            return potential_reward / potential_risk if potential_risk > 0 else 0

        signals_for_this_strategy['rr'] = signals_for_this_strategy.apply(calculate_strategy_rr, axis=1)
        signals_for_this_strategy = signals_for_this_strategy[signals_for_this_strategy['rr'] >= 1.5]
        log.info(f"  [Veto] {len(signals_for_this_strategy)} signals pass R:R check >= 1.5")

        # --- END OF NEW VETO LOGIC BLOCK ---

        portfolio = {
            'equity': config.BACKTEST_PORTFOLIO_CONFIG['initial_capital'],
            'open_positions': [], 'daily_equity_log': [], 'closed_trades_for_report': []
        }

        # Important: Check if any signals are left after filtering
        if signals_for_this_strategy.empty:
            log.warning(f"No signals remained for strategy '{strategy['name']}' after applying vetoes. Skipping simulation.")
            continue # Skip to the next strategy

        simulation_start_date = signals_for_this_strategy['prediction_date'].min()
        simulation_end_date = signals_for_this_strategy['prediction_date'].max() + pd.Timedelta(days=30) # Adjusted end date
        simulation_days = pd.bdate_range(start=simulation_start_date, end=simulation_end_date)
        portfolio = {
            'equity': config.BACKTEST_PORTFOLIO_CONFIG['initial_capital'],
            'open_positions': [], 'daily_equity_log': [], 'closed_trades_for_report': []
        }
        simulation_start_date = all_signals_df['prediction_date'].min()
        simulation_days = pd.bdate_range(start=simulation_start_date, end=simulation_end_date)

        for day in simulation_days:
            # --- MANAGE EXITS FOR OPEN POSITIONS ---
            for position in portfolio['open_positions'][:]:
                ticker = position['ticker']
                try:
                    day_data = data_cache[ticker].loc[day]
                    # Get the index of the current day to find the previous day's data
                    current_day_index = data_cache[ticker].index.get_loc(day)
                    if current_day_index > 0:
                        previous_day_data = data_cache[ticker].iloc[current_day_index - 1]
                    else:
                        previous_day_data = day_data # Fallback if it's the first day in history

                except KeyError:
                    continue

                close_reason, closing_price = None, None
                
                # --- Set SL/TP based on the strategy currently being tested ---
                strategy_name = position['strategy']

                if "ATR_3R" in strategy_name:
                    position['stop_loss'] = position['initial_atr_stop_loss_A']
                    position['target'] = position['initial_atr_target_A']
                elif "AI_ATR_Stop" in strategy_name:
                    position['stop_loss'] = position['initial_atr_stop_loss_B']
                    position['target'] = position['initial_ai_target']
                elif "AI_Structure_Stop" in strategy_name:
                    position['stop_loss'] = position['initial_structure_stop_loss_C']
                    position['target'] = position['initial_ai_target']

                # --- Adaptive Trailing Stop Logic ---
                is_ai_strategy = "AI_ATR_Stop" in strategy_name or "AI_Structure_Stop" in strategy_name
                # Use .get() for safety
                if is_ai_strategy and strategy.get('use_adaptive_profit_take'):
                    if not position.get('adaptive_stop_activated') and day_data.high >= position['halfway_profit_price']:
                        position['adaptive_stop_activated'] = True
                        log.info(f"ADAPTIVE STOP ACTIVATED for {ticker} at {day_data.high:.2f}")

                    if position.get('adaptive_stop_activated'):
                        if 'ATRr_14' not in data_cache[ticker].columns:
                            data_cache[ticker].ta.atr(length=14, append=True)
                        current_atr = data_cache[ticker].loc[day]['ATRr_14']
                        atr_trail_amount = current_atr * strategy['adaptive_trail_atr_multiplier']
                        new_trailing_stop = day_data.high - atr_trail_amount

                        if new_trailing_stop > position['stop_loss']:
                            position['stop_loss'] = new_trailing_stop

                # --- Final Exit Condition Checks ---
                stop_loss = position['stop_loss']
                target = position['target']

                if position['signal'] == 'BUY':
                    # Priority 1: Check for a gap-down that blows past the stop-loss at the open
                    if day_data.open <= stop_loss:
                        close_reason, closing_price = "Stop-Loss Hit (Gap Down)", day_data.open
                    # Priority 2: Check for an intraday touch of the stop-loss
                    elif day_data.low <= stop_loss:
                        close_reason, closing_price = "Stop-Loss Hit (Intraday)", stop_loss
                    # Priority 3: Check for a gap-up that hits the target at the open
                    elif day_data.open >= target:
                        close_reason, closing_price = "Target Hit (Gap Up)", day_data.open
                    # Priority 4: Check for an intraday touch of the target
                    elif day_data.high >= target:
                        close_reason, closing_price = "Target Hit (Intraday)", target

                elif position['signal'] == 'SELL': # Logic is reversed for short positions
                    # Priority 1: Check for a gap-up that blows past the stop-loss at the open
                    if day_data.open >= stop_loss:
                        close_reason, closing_price = "Stop-Loss Hit (Gap Up)", day_data.open
                    # Priority 2: Check for an intraday touch of the stop-loss
                    elif day_data.high >= stop_loss:
                        close_reason, closing_price = "Stop-Loss Hit (Intraday)", stop_loss
                    # Priority 3: Check for a gap-down that hits the target at the open
                    elif day_data.open <= target:
                        close_reason, closing_price = "Target Hit (Gap Down)", day_data.open
                    # Priority 4: Check for an intraday touch of the target
                    elif day_data.low <= target:
                        close_reason, closing_price = "Target Hit (Intraday)", target

                if not close_reason:
                    days_held = (day - position['open_date']).days
                    max_holding_period = strategy.get('holding_period', 10) # Default to 10 if not set
                    if days_held >= max_holding_period:
                        close_reason = "Time Exit (Max Hold)"
                        closing_price = day_data.close # Exit at the day's closing price

                if close_reason:
                    closed_trade_doc, net_pnl = _calculate_closed_trade(position, closing_price, close_reason, day, previous_day_data)
                    log.info(f"[{strategy['name']}] CLOSING TRADE: {ticker} for {close_reason}. Net P&L: ₹{net_pnl:.2f}")
                    portfolio['equity'] += net_pnl
                    portfolio['closed_trades_for_report'].append(closed_trade_doc)
                    portfolio['open_positions'].remove(position)
            
            # --- ENTER NEW TRADES ---
            new_signals_for_today = signals_for_this_strategy[signals_for_this_strategy['prediction_date'].dt.date == day.date()]
            if not new_signals_for_today.empty:
                for index, signal in new_signals_for_today.iterrows():
                    if len(portfolio['open_positions']) >= PORTFOLIO_CONSTRAINTS['max_concurrent_trades']:
                        log.warning(f"SKIPPING SIGNAL for {signal['ticker']}: Max concurrent trade limit reached.")
                        continue
                    
                    if any(p['ticker'] == signal['ticker'] for p in portfolio['open_positions']):
                        log.warning(f"SKIPPING SIGNAL for {signal['ticker']}: Position already open.")
                        continue

                    risk_amount = portfolio['equity'] * (config.BACKTEST_PORTFOLIO_CONFIG['risk_per_trade_pct'] / 100.0)
                    trade_plan = signal.get('tradePlan', {})
                    entry_price = trade_plan.get('entryPrice')

                    # --- THIS IS THE DYNAMIC POSITION SIZING LOGIC ---
                    risk_per_share = 0
                    if "ATR_3R" in strategy['name']:
                        risk_per_share = abs(entry_price - trade_plan.get('atr_stop_loss_A', 0))
                    elif "AI_ATR_Stop" in strategy['name']:
                        risk_per_share = abs(entry_price - trade_plan.get('atr_stop_loss_B', 0))
                    elif "AI_Structure_Stop" in strategy['name']:
                        risk_per_share = abs(entry_price - trade_plan.get('structure_stop_loss_C', 0))
                    
                    if risk_per_share <= 0:
                        log.warning(f"SKIPPING SIGNAL for {signal['ticker']}: Invalid risk per share ({risk_per_share:.2f}) for strategy {strategy['name']}.")
                        continue

                    num_shares = int(risk_amount / risk_per_share)
                    # --- END OF DYNAMIC LOGIC ---

                    if num_shares == 0:
                        log.warning(f"SKIPPING SIGNAL for {signal['ticker']}: Calculated shares to trade is zero.")
                        continue

                    new_position = {
                        'prediction_id': signal['_id'], 'ticker': signal['ticker'], 
                        'signal': signal['signal'], 'entry_price': entry_price, 
                        'num_shares': num_shares, 'open_date': day, 'batch_id': signal['batch_id'],
                        'strategy': strategy['name'],
                        'initial_ai_target': trade_plan.get('ai_target'),
                        'initial_atr_target_A': trade_plan.get('atr_target_A'),
                        'initial_atr_stop_loss_A': trade_plan.get('atr_stop_loss_A'),
                        'initial_atr_stop_loss_B': trade_plan.get('atr_stop_loss_B'),
                        'initial_structure_stop_loss_C': trade_plan.get('structure_stop_loss_C'),
                        'halfway_profit_price': trade_plan.get('halfway_profit_price'),
                        'adaptive_stop_activated': False,
                    }

                    portfolio['open_positions'].append(new_position)
                    log.info(f"[{strategy['name']}] ENTERING TRADE: {signal['signal']} {num_shares} shares of {signal['ticker']} @ {entry_price:.2f}")

            portfolio['daily_equity_log'].append({
                'date': day, 
                'equity': portfolio['equity'],
                'open_positions_count': len(portfolio['open_positions']) # ADD THIS LINE
            })
        
        all_results_for_db.extend(portfolio['closed_trades_for_report'])
        
        # --- FINAL REPORTING FOR THIS STRATEGY ---
        if not portfolio['daily_equity_log']:
            log.warning(f"No equity data logged for strategy '{strategy['name']}'. Skipping report.")
            continue
            
        equity_df = pd.DataFrame(portfolio['daily_equity_log']).set_index('date')
        closed_trades_df = pd.DataFrame(portfolio['closed_trades_for_report'])
        
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
            downside_std = equity_df[equity_df['daily_return'] < 0]['daily_return'].std(ddof=0)
            sortino_ratio = (equity_df['daily_return'].mean() / downside_std) * (252**0.5) if downside_std > 0 else 0
            annualized_return = total_return_pct * (252 / len(equity_df)) if len(equity_df) > 0 else 0
            calmar_ratio = annualized_return / max_drawdown_pct if max_drawdown_pct > 0 else 0
        else:
            win_rate, profit_factor, sharpe_ratio, sortino_ratio, calmar_ratio = 0, 0, 0, 0, 0
        
        # --- NEW: Calculate Activity Metrics ---
        if total_trades > 0:
            # Calculate Average Holding Period
            closed_trades_df['open_date'] = pd.to_datetime(closed_trades_df['open_date'])
            closed_trades_df['close_date'] = pd.to_datetime(closed_trades_df['close_date'])
            closed_trades_df['holding_period'] = (closed_trades_df['close_date'] - closed_trades_df['open_date']).dt.days
            avg_holding_period = closed_trades_df['holding_period'].mean()

            # Calculate Exposure Time
            days_with_positions = equity_df[equity_df['open_positions_count'] > 0].shape[0]
            total_days_in_sim = equity_df.shape[0]
            exposure_time_pct = (days_with_positions / total_days_in_sim) * 100 if total_days_in_sim > 0 else 0

            # Calculate Annualized Trades
            years_in_sim = total_days_in_sim / 252.0
            annualized_trades = total_trades / years_in_sim if years_in_sim > 0 else 0
        else:
            avg_holding_period, exposure_time_pct, annualized_trades = 0, 0, 0

        print("\n" + "="*80)
        print(f"### Backtest Performance Summary: '{strategy['name']}' on Batch '{batch_id}' ###")
        print("="*80)
        print(f"{'Total Return:':<25} {total_return_pct:.2f}%")
        print(f"{'Final Portfolio Value:':<25} ₹{final_equity:,.2f}")
        print(f"{'Max Drawdown:':<25} {max_drawdown_pct:.2f}%")
        print("-" * 80)
        print("--- Risk & Return Metrics ---")
        print(f"{'Sharpe Ratio:':<25} {sharpe_ratio:.2f}")
        print(f"{'Sortino Ratio:':<25} {sortino_ratio:.2f}")
        print(f"{'Calmar Ratio:':<25} {calmar_ratio:.2f}")
        print("-" * 80)
        print("--- Trade Metrics ---")
        print(f"{'Total Trades:':<25} {total_trades}")
        print(f"{'Win Rate:':<25} {win_rate:.2f}%")
        print(f"{'Profit Factor:':<25} {profit_factor:.2f}")
        print("-" * 80)
        print("--- Activity Metrics ---") # NEW SECTION
        print(f"{'Exposure Time:':<25} {exposure_time_pct:.2f}%")
        print(f"{'Avg Holding Period (Days):':<25} {avg_holding_period:.1f}")
        print(f"{'Trades per Year:':<25} {annualized_trades:.1f}")
        print("="*80)
    
    if all_results_for_db:
        log.info(f"Writing {len(all_results_for_db)} closed trades to the database...")
        # We need to make sure each trade record is tagged with its strategy name
        final_records = []
        for trade_record in all_results_for_db:
            # The 'strategy' field is already in the trade_record from the _calculate_closed_trade function
            final_records.append(trade_record)

        database_manager.performance_collection.delete_many({"batch_id": batch_id})
        if final_records:
            database_manager.performance_collection.insert_many(final_records)
            log.info("✅ Batch database write complete.")

    database_manager.close_db_connection()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Stateful Backtester for a specific batch.")
    parser.add_argument('--batch_id', required=True, help='The unique ID of the backtest batch to process.')
    args = parser.parse_args()
    run_stateful_backtest(batch_id=args.batch_id)