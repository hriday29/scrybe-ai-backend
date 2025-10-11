import pandas as pd
import config
from logger_config import log

def calculate_historical_performance(historical_data: pd.DataFrame) -> dict:
    """
    Calculates the stock's price performance over standard trading day periods.
    """
    if historical_data.empty:
        return {}

    log.info("Calculating historical performance snapshot...")
    performance = {}
    # Use standard trading days instead of calendar days
    periods = {'1-Week': 5, '1-Month': 21, '3-Month': 63, '6-Month': 126, '1-Year': 252}
    latest_price = historical_data['close'].iloc[-1]
    
    for name, days in periods.items():
        if len(historical_data) > days:
            old_price = historical_data['close'].iloc[-days]
            change_pct = ((latest_price - old_price) / old_price) * 100
            performance[name] = {'change_percent': f"{change_pct:.2f}%"}
        else:
            performance[name] = {'change_percent': 'N/A'}
    
    log.info("Successfully calculated performance snapshot.")
    return performance

def calculate_correlations(stock_historical_data: pd.DataFrame, benchmarks_data: pd.DataFrame) -> dict:
    """
    Calculates the 90-day correlation between a stock's returns and the returns
    of key benchmarks.
    """
    log.info("Calculating inter-market correlations...")
    if stock_historical_data is None or benchmarks_data is None or stock_historical_data.empty or benchmarks_data.empty:
        return {}
        
    try:
        # Using pct_change() for returns is the more robust method
        stock_returns = stock_historical_data['close'].pct_change().rename('Stock')
        benchmark_returns = benchmarks_data.pct_change(fill_method=None)
        
        combined_returns = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
        
        if len(combined_returns) < 90:
            log.warning("Not enough data for 90-day correlation.")
            return {}

        correlation_matrix = combined_returns.tail(90).corr()
        stock_correlations = correlation_matrix['Stock'].drop('Stock')

        formatted_correlations = {
            f"{k} Correlation": f"{v:.2f}" for k, v in stock_correlations.items()
        }
        log.info(f"Successfully calculated {len(formatted_correlations)} correlations.")
        return formatted_correlations
        
    except Exception as e:
        log.error(f"An error occurred during correlation calculation: {e}")
        return {}
    
def generate_backtest_report(portfolio: dict, batch_id: str) -> dict:
    """
    Calculates and prints a comprehensive performance summary for a completed backtest.
    """
    log.info(f"--- Generating Final Performance Report for Batch: {batch_id} ---")

    if not portfolio['daily_equity_log']:
        log.warning("[REPORT] No equity data logged. Skipping final report.")
        return {}

    equity_df = pd.DataFrame(portfolio['daily_equity_log']).set_index('date')
    closed_trades_df = pd.DataFrame(portfolio['closed_trades'])
    initial_capital = config.BACKTEST_PORTFOLIO_CONFIG['initial_capital']

    # --- Core Metric Calculations ---
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

    # --- Print the Report to Console ---
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

    # --- Return a dictionary of metrics for potential database saving ---
    report_metrics = {
        "total_return_pct": round(total_return_pct, 2),
        "final_equity": round(final_equity, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2)
    }
    return report_metrics