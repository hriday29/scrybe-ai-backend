# stateful_backtester.py (Final Verified Version)

import pandas as pd
from logger_config import log
import database_manager
import data_retriever
import config
import argparse
from config import PORTFOLIO_CONSTRAINTS

def _calculate_closed_trade(position: dict, closing_price: float, closing_reason: str, close_date: pd.Timestamp):
    """
    Calculates the P&L of a closed trade and returns a performance document.
    This function DOES NOT modify the portfolio; it only does calculations.
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
    other_charges = turnover * (costs['slippage_pct'] / 100.0)
    total_transaction_costs = brokerage + stt + other_charges
    net_pnl = gross_pnl - total_transaction_costs
    
    performance_doc = {
        "prediction_id": position['prediction_id'], "ticker": position['ticker'], 
        "strategy": position['strategy'], "signal": signal, "status": "Closed", 
        "open_date": position['open_date'], "close_date": close_date.to_pydatetime(), 
        "closing_reason": closing_reason, "net_pnl": round(net_pnl, 2),
        "batch_id": position['batch_id']
    }
    return performance_doc, net_pnl

def run_stateful_backtest(batch_id: str):
    log.info(f"--- 🚀 Starting Stateful Backtest for Batch: '{batch_id}' ---")

    strategies_to_test = [
        {"name": "Flash Model Baseline (Hold: 10d, SL: 1.5x ATR)", "holding_period": 10}
    ]

    database_manager.init_db(purpose='scheduler')
    # We query for actual trade signals, not HOLDs or VETOED signals.
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
    max_holding_days = max(s['holding_period'] for s in strategies_to_test)
    simulation_end_date = last_signal_date + pd.Timedelta(days=max_holding_days + 5)
    data_cache = {}
    for ticker in unique_tickers:
        data = data_retriever.get_historical_stock_data(ticker, end_date=simulation_end_date.strftime('%Y-%m-%d'))
        if data is not None and not data.empty: data_cache[ticker] = data
        else: all_signals_df = all_signals_df[all_signals_df['ticker'] != ticker]
    log.info("✅ Setup complete.")

    all_results_for_db = []
    for strategy in strategies_to_test:
        log.info(f"--- Running Simulation for Strategy: '{strategy['name']}' ---")

        portfolio = {
            'equity': config.BACKTEST_PORTFOLIO_CONFIG['initial_capital'],
            'open_positions': [], 'daily_equity_log': [], 'closed_trades_for_report': []
        }
        simulation_start_date = all_signals_df['prediction_date'].min()
        simulation_days = pd.bdate_range(start=simulation_start_date, end=simulation_end_date)

        for day in simulation_days:
            for position in portfolio['open_positions'][:]:
                ticker = position['ticker']
                try: day_data = data_cache[ticker].loc[day]
                except KeyError: continue
                close_reason, closing_price = None, None
                if position['signal'] == 'BUY':
                    if day_data.low <= position['stop_loss']: close_reason, closing_price = "Stop-Loss Hit", position['stop_loss']
                    elif day_data.high >= position['target']: close_reason, closing_price = "Target Hit", position['target']
                elif position['signal'] == 'SELL':
                    if day_data.high >= position['stop_loss']: close_reason, closing_price = "Stop-Loss Hit", position['stop_loss']
                    elif day_data.low <= position['target']: close_reason, closing_price = "Target Hit", position['target']
                days_held = (day.date() - position['open_date'].date()).days
                if not close_reason and days_held >= strategy['holding_period']:
                    close_reason, closing_price = "Time Exit", day_data.close
                
                if close_reason:
                    closed_trade_doc, net_pnl = _calculate_closed_trade(position, closing_price, close_reason, day)
                    log.info(f"CLOSING TRADE: {position['ticker']} for {close_reason}. Net P&L: ₹{net_pnl:.2f}")
                    portfolio['equity'] += net_pnl
                    portfolio['closed_trades_for_report'].append(closed_trade_doc)
                    portfolio['open_positions'].remove(position)
            
            new_signals_for_today = all_signals_df[all_signals_df['prediction_date'].dt.date == day.date()]
            if not new_signals_for_today.empty:
                for index, signal in new_signals_for_today.iterrows():
                    # --- PORTFOLIO-LEVEL CHECK ---
                    if len(portfolio['open_positions']) >= PORTFOLIO_CONSTRAINTS['max_concurrent_trades']:
                        log.warning(f"SKIPPING SIGNAL for {signal['ticker']}: Max concurrent trade limit ({PORTFOLIO_CONSTRAINTS['max_concurrent_trades']}) reached.")
                        continue

                    risk_amount = portfolio['equity'] * (config.BACKTEST_PORTFOLIO_CONFIG['risk_per_trade_pct'] / 100.0)
                    trade_plan = signal.get('tradePlan', {})
                    entry_price, stop_loss_price = trade_plan.get('entryPrice'), trade_plan.get('stopLoss')
                    if not all([entry_price, stop_loss_price]): continue
                    risk_per_share = abs(entry_price - stop_loss_price)
                    if risk_per_share <= 0: continue
                    num_shares = int(risk_amount / risk_per_share)
                    if num_shares == 0: continue
                    new_position = {
                        'prediction_id': signal['_id'], 'ticker': signal['ticker'], 'signal': signal['signal'],
                        'entry_price': entry_price, 'num_shares': num_shares, 'stop_loss': stop_loss_price,
                        'target': trade_plan.get('target'), 'open_date': day, 'holding_period': strategy['holding_period'],
                        'strategy': strategy['name'], 'batch_id': signal['batch_id']
                    }
                    portfolio['open_positions'].append(new_position)
                    log.info(f"ENTERING TRADE: {signal['signal']} {num_shares} shares of {signal['ticker']} @ {entry_price:.2f}")
            portfolio['daily_equity_log'].append({'date': day, 'equity': portfolio['equity']})
        
        all_results_for_db.extend(portfolio['closed_trades_for_report'])
        
        # --- FINAL REPORTING FOR THIS STRATEGY ---
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
            sharpe_ratio = (equity_df['daily_return'].mean() / equity_df['daily_return'].std()) * (252**0.5) if equity_df['daily_return'].std() != 0 else 0
            downside_std = equity_df[equity_df['daily_return'] < 0]['daily_return'].std()
            sortino_ratio = (equity_df['daily_return'].mean() / downside_std) * (252**0.5) if downside_std > 0 else 0
            annualized_return = total_return_pct * (252 / len(equity_df))
            calmar_ratio = annualized_return / max_drawdown_pct if max_drawdown_pct > 0 else 0
        else:
            win_rate, profit_factor, sharpe_ratio, sortino_ratio, calmar_ratio = 0, 0, 0, 0, 0
        print("\n" + "="*80)
        print(f"### Backtest Performance Summary: '{strategy['name']}' on Batch '{batch_id}' ###")
        print("="*80)
        print(f"{'Total Return:':<25} {total_return_pct:.2f}%")
        print(f"{'Final Portfolio Value:':<25} ₹{final_equity:,.2f}")
        print(f"{'Max Drawdown:':<25} {max_drawdown_pct:.2f}%")
        print("-" * 80)
        print(f"{'Sharpe Ratio:':<25} {sharpe_ratio:.2f}")
        print(f"{'Sortino Ratio:':<25} {sortino_ratio:.2f}")
        print(f"{'Calmar Ratio:':<25} {calmar_ratio:.2f}")
        print("-" * 80)
        print(f"{'Total Trades:':<25} {total_trades}")
        print(f"{'Win Rate:':<25} {win_rate:.2f}%")
        print(f"{'Profit Factor:':<25} {profit_factor:.2f}")
        print("="*80)
    
    if all_results_for_db:
        log.info(f"Writing {len(all_results_for_db)} closed trades to the database...")
        database_manager.performance_collection.delete_many({"batch_id": batch_id})
        database_manager.performance_collection.insert_many(all_results_for_db)
        log.info("✅ Batch database write complete.")

    database_manager.close_db_connection()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Stateful Backtester for a specific batch.")
    parser.add_argument('--batch_id', required=True, help='The unique ID of the backtest batch to process.')
    args = parser.parse_args()
    run_stateful_backtest(batch_id=args.batch_id)