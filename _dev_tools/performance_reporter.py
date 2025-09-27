# exhaustive_reporter.py (FIXED & FINAL VERSION)
# Description: This script performs a comprehensive, multi-faceted analysis of
# backtest results from a MongoDB database. It generates a detailed text report
# and saves it to a file, covering everything from high-level portfolio metrics
# to deep statistical analysis and individual trade logs.

import argparse
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
from tabulate import tabulate

# --- CONFIGURATION & INITIALIZATION ---
try:
    from config import BACKTEST_PORTFOLIO_CONFIG
except ImportError:
    print("FATAL ERROR: config.py not found. Make sure it's in the same directory.")
    sys.exit(1)

load_dotenv()
MONGO_URI = os.getenv("SCHEDULER_DB_URI")
DB_NAME = "scheduler_db"
COLLECTION_NAME = "performance"
REPORT_DIR = "reports"


def get_augmented_data_from_db(batch_id: str) -> pd.DataFrame:
    """
    Connects to MongoDB, fetches all trades for a batch, and augments the
    DataFrame with a rich set of calculated fields for deep analysis.
    """
    print(f"🔗 Connecting to MongoDB...")
    try:
        client = MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        print(f"🔎 Fetching all trades for batch_id: '{batch_id}'...")
        trades_list = list(collection.find({"batch_id": batch_id}))
        
        if not trades_list:
            print(f"❌ No trades found for batch_id '{batch_id}'. Exiting.")
            sys.exit(1) # Exit if no data
            
        print(f"✅ Found {len(trades_list)} trades. Augmenting data for deep analysis...")
        df = pd.DataFrame(trades_list).sort_values(by='close_date').reset_index(drop=True)
        
        # --- Data Type Conversion and Basic Augmentation ---
        df['open_date'] = pd.to_datetime(df['open_date'])
        df['close_date'] = pd.to_datetime(df['close_date'])
        
        # --- Advanced Augmentation ---
        # Reverse-engineer invested capital for percentage-based metrics
        df['invested_capital'] = abs(df.apply(
            lambda r: (r['net_pnl'] / (r['net_return_pct'] / 100)) if r['net_return_pct'] != 0 else 0, axis=1
        ))
        mean_capital = df[df['invested_capital'] > 0]['invested_capital'].mean()
        # FIX: Avoid chained assignment warning
        df['invested_capital'] = df['invested_capital'].replace(0, mean_capital)
        
        df['is_win'] = (df['net_pnl'] > 0).astype(int)
        df['holding_period'] = (df['close_date'] - df['open_date']).dt.days
        df['pnl_per_day'] = df['net_pnl'] / df['holding_period'].replace(0, 1)
        
        # Columns for temporal analysis
        df['year'] = df['close_date'].dt.year
        df['year_month'] = df['close_date'].dt.strftime('%Y-%B')
        df['entry_day_of_week'] = df['open_date'].dt.day_name()
        
        return df

    except Exception as e:
        print(f"FATAL ERROR connecting or fetching data: {e}")
        sys.exit(1)
    finally:
        if 'client' in locals(): client.close()


def calculate_drawdown_details(equity_curve: pd.Series) -> dict:
    """Calculates detailed information about the max drawdown period."""
    high_water_mark = equity_curve.cummax()
    drawdown = (equity_curve - high_water_mark) / high_water_mark
    
    trough_date = drawdown.idxmin()
    peak_date = equity_curve.loc[:trough_date].idxmax()
    
    # Attempt to find recovery date after the trough
    recovery_df = equity_curve.loc[trough_date:]
    recovery_date_series = recovery_df[recovery_df >= equity_curve[peak_date]].first_valid_index()
    
    # --- 🐞 ERROR FIX IS HERE ---
    # Handle the case where the portfolio never recovers
    if pd.notnull(recovery_date_series):
        recovery_date = recovery_date_series.date()
        duration = (recovery_date_series - peak_date).days
    else:
        recovery_date = "Not Recovered"
        duration = "N/A"
    
    return {
        "start_date": peak_date.date(),
        "trough_date": trough_date.date(),
        "recovery_date": recovery_date,
        "duration_days": duration,
        "max_dd_pct": abs(drawdown.min())
    }

def main(batch_id: str):
    """
    Main function to orchestrate data fetching, analysis, and report generation.
    """
    trades_df = get_augmented_data_from_db(batch_id)
    
    # --- Setup File Output ---
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_filename = os.path.join(REPORT_DIR, f"report_{batch_id}_{timestamp}.txt")
    
    # Use a list to build the report content, then join and write once.
    report_content = []
    
    # --- Analysis & Report Building ---
    initial_capital = BACKTEST_PORTFOLIO_CONFIG['initial_capital']
    winning_trades = trades_df[trades_df['is_win'] == 1]
    losing_trades = trades_df[trades_df['is_win'] == 0]
    win_rate = len(winning_trades) / len(trades_df)

    # --- Equity Curve & Drawdown ---
    trades_df['equity'] = initial_capital + trades_df['net_pnl'].cumsum()
    daily_equity = pd.Series(trades_df['equity'].values, index=pd.to_datetime(trades_df['close_date']))
    # FIX: Remove redundant and deprecated fillna(method='ffill')
    daily_equity = daily_equity.resample('D').ffill().fillna(initial_capital)
    
    # --- Header ---
    report_content.append("="*90)
    report_content.append(f"### Definitive Performance Report for Batch: '{batch_id}' ###")
    report_content.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append("="*90)

    # --- 1. Executive Summary ---
    report_content.append("\n\n--- 1. Executive Summary ---\n")
    report_content.append(tabulate([
        ("Final Portfolio Value", f"₹{trades_df['equity'].iloc[-1]:,.2f}"),
        ("Net Profit (Absolute)", f"₹{trades_df['net_pnl'].sum():,.2f}"),
        ("Net Profit (%)", f"{(trades_df['net_pnl'].sum() / initial_capital):.2%}"),
        ("Total Trades", f"{len(trades_df)}"),
        ("Win Rate", f"{win_rate:.2%}"),
        ("Profit Factor", f"{winning_trades['net_pnl'].sum() / abs(losing_trades['net_pnl'].sum()):.2f}"),
        ("Expectancy per Trade (₹)", f"₹{(win_rate * winning_trades['net_pnl'].mean()) - ((1-win_rate) * abs(losing_trades['net_pnl'].mean())):,.2f}")
    ], headers=["Metric", "Value"], tablefmt="grid"))
    
    # --- 2. P&L Deep Dive & Return Statistics ---
    report_content.append("\n\n--- 2. P&L Deep Dive & Return Statistics ---\n")
    report_content.append(tabulate([
        ("Metric", "Value (₹)", "Value (%)"),
        ("Gross Profit", f"₹{winning_trades['net_pnl'].sum():,.2f}", ""),
        ("Gross Loss", f"₹{losing_trades['net_pnl'].sum():,.2f}", ""),
        ("Avg Win", f"₹{winning_trades['net_pnl'].mean():,.2f}", f"{winning_trades['net_return_pct'].mean():.2f}%"),
        ("Avg Loss", f"₹{losing_trades['net_pnl'].mean():,.2f}", f"{losing_trades['net_return_pct'].mean():.2f}%"),
        ("Largest Win", f"₹{trades_df['net_pnl'].max():,.2f}", f"{trades_df['net_return_pct'].max():.2f}%"),
        ("Largest Loss", f"₹{trades_df['net_pnl'].min():,.2f}", f"{trades_df['net_return_pct'].min():.2f}%"),
        ("Std. Dev of P&L", f"₹{trades_df['net_pnl'].std():,.2f}", f"{trades_df['net_return_pct'].std():.2f}%"),
        ("Skewness of P&L", f"{trades_df['net_pnl'].skew():.2f}", f"{trades_df['net_return_pct'].skew():.2f}"),
        ("Kurtosis of P&L", f"{trades_df['net_pnl'].kurt():.2f}", f"{trades_df['net_return_pct'].kurt():.2f}")
    ], headers="firstrow", tablefmt="grid"))

    # --- 3. Risk, Drawdown & Volatility ---
    report_content.append("\n\n--- 3. Risk, Drawdown & Volatility ---\n")
    dd_details = calculate_drawdown_details(daily_equity)
    duration_years = (daily_equity.index.max() - daily_equity.index.min()).days / 365.25
    annualized_return = ((daily_equity.iloc[-1] / initial_capital) ** (1 / duration_years) - 1) if duration_years > 0 else 0
    annualized_vol = daily_equity.pct_change().std() * np.sqrt(252)
    report_content.append(tabulate([
        ("Max Drawdown (%)", f"{dd_details['max_dd_pct']:.2%}"),
        ("Drawdown Start", dd_details['start_date']),
        ("Drawdown Trough", dd_details['trough_date']),
        ("Drawdown Recovery", dd_details['recovery_date']),
        ("Drawdown Duration", f"{dd_details['duration_days']} days"),
        ("Annualized Return", f"{annualized_return:.2%}"),
        ("Annualized Volatility", f"{annualized_vol:.2%}"),
        ("Sharpe Ratio (Annualized)", f"{(annualized_return / annualized_vol):.2f}" if annualized_vol > 0 else "N/A"),
        ("Calmar Ratio (Ann. Return / Max DD)", f"{(annualized_return / dd_details['max_dd_pct']):.2f}" if dd_details['max_dd_pct'] > 0 else "N/A")
    ], headers=["Metric", "Value"], tablefmt="grid"))

    # --- 4. Positional & Holding Analysis ---
    report_content.append("\n\n--- 4. Positional & Holding Analysis ---\n")
    report_content.append("\nPerformance by Closing Reason:")
    reason_analysis = trades_df.groupby('closing_reason').agg(Count=('ticker', 'size'), Win_Rate=('is_win', 'mean'), Avg_PnL_Pct=('net_return_pct', 'mean'))
    report_content.append(tabulate(reason_analysis, headers="keys", tablefmt="pretty"))
    
    report_content.append("\n\nPerformance by Signal Type:")
    signal_analysis = trades_df.groupby('signal').agg(Count=('ticker', 'size'), Win_Rate=('is_win', 'mean'), Avg_PnL_Pct=('net_return_pct', 'mean'))
    report_content.append(tabulate(signal_analysis, headers="keys", tablefmt="pretty"))
    
    bins = [0, 5, 10, 20, 40, 90, 1000]
    labels = ['0-5d', '6-10d', '11-20d', '21-40d', '41-90d', '90d+']
    trades_df['holding_bucket'] = pd.cut(trades_df['holding_period'], bins=bins, labels=labels, right=False)
    report_content.append("\n\nPerformance by Holding Period:")
    holding_analysis = trades_df.groupby('holding_bucket', observed=True).agg(Count=('ticker', 'size'), Win_Rate=('is_win', 'mean'), Avg_PnL_Pct=('net_return_pct', 'mean'))
    report_content.append(tabulate(holding_analysis, headers="keys", tablefmt="pretty"))

    # --- 5. Time-Based Analysis ---
    report_content.append("\n\n--- 5. Time-Based Analysis ---\n")
    report_content.append("\nPerformance by Year:")
    yearly_analysis = trades_df.groupby('year').agg(Trades=('ticker','size'), Win_Rate=('is_win','mean'), Net_PnL=('net_pnl','sum'))
    report_content.append(tabulate(yearly_analysis, headers="keys", tablefmt="pretty"))

    report_content.append("\n\nPerformance by Month:")
    monthly_analysis = trades_df.groupby('year_month').agg(Trades=('ticker','size'), Win_Rate=('is_win','mean'), Net_PnL=('net_pnl','sum'))
    report_content.append(tabulate(monthly_analysis, headers="keys", tablefmt="pretty"))
    
    report_content.append("\n\nPerformance by Entry Day of Week:")
    day_analysis = trades_df.groupby('entry_day_of_week').agg(Trades=('ticker','size'), Win_Rate=('is_win','mean'), Net_PnL=('net_pnl','sum'))
    report_content.append(tabulate(day_analysis.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']), headers="keys", tablefmt="pretty"))

    # --- 6. Ticker-Level Breakdown (Complete) ---
    report_content.append("\n\n--- 6. Ticker-Level Breakdown ---\n")
    ticker_analysis = trades_df.groupby('ticker').agg(
        Trades=('ticker','size'),
        Win_Rate=('is_win','mean'),
        Net_PnL=('net_pnl','sum'),
        Avg_PnL_Pct=('net_return_pct', 'mean')
    ).sort_values('Net_PnL', ascending=False)
    report_content.append(tabulate(ticker_analysis, headers="keys", tablefmt="pretty"))

    # --- 7. Complete Unabridged Trade Log ---
    report_content.append("\n\n--- 7. Complete Unabridged Trade Log ---\n")
    report_content.append(tabulate(trades_df[[
        'ticker', 'signal', 'open_date', 'close_date', 'holding_period', 
        'net_pnl', 'net_return_pct', 'invested_capital', 'closing_reason'
    ]].round(2), headers="keys", tablefmt="pretty"))

    # --- Write to file ---
    try:
        with open(report_filename, 'w', encoding='utf-8') as f:  # FIXED
            f.write("\n".join(report_content))
        print(f"\n✅ Exhaustive report successfully generated and saved to: {report_filename}")
    except Exception as e:
        print(f"\n❌ ERROR: Could not write report to file. Error: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the definitive, exhaustive performance report for a backtest batch.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--batch_id', required=True, help='The unique ID of the batch to analyze.')
    args = parser.parse_args()
    
    # Execute the main analysis and reporting function
    main(args.batch_id)