import os
import pandas as pd
import numpy as np

# --- Configuration ---
TRADE_LOG_FILE = 'trade_log.csv'
INITIAL_CAPITAL = 100000.0  # Match this to your backtest config
REPORTS_DIR = 'reports'
REPORT_FILE = os.path.join(REPORTS_DIR, 'backtest_analysis_summary.txt')

# --- Create reports directory if not exists ---
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- Load Data ---
try:
    df = pd.read_csv(TRADE_LOG_FILE)
    print(f"Successfully loaded {len(df)} trades from '{TRADE_LOG_FILE}'.")
except FileNotFoundError:
    print(f"ERROR: Trade log file '{TRADE_LOG_FILE}' not found.")
    print("Please export your trade data to CSV and place it in the same directory.")
    exit()
except Exception as e:
    print(f"ERROR: Could not load or parse '{TRADE_LOG_FILE}': {e}")
    exit()

# --- Data Cleaning & Preparation ---
required_cols = ['open_date', 'close_date', 'net_pnl', 'net_return_pct', 'closing_reason', 'signal']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"ERROR: CSV file is missing required columns: {', '.join(missing_cols)}")
    exit()

# Convert dates and calculate holding period
try:
    df['open_date'] = pd.to_datetime(df['open_date'], errors='coerce')
    df['close_date'] = pd.to_datetime(df['close_date'], errors='coerce')
    df.dropna(subset=['open_date', 'close_date'], inplace=True)
    if df.empty:
        print("ERROR: No valid date entries found after cleaning. Check date format in CSV.")
        exit()
    df['holding_period_days'] = (df['close_date'] - df['open_date']).dt.days
except Exception as e:
    print(f"ERROR: Could not process date columns: {e}")
    exit()

# --- Redirect print output to both console and file ---
class DualWriter:
    def __init__(self, *targets):
        self.targets = targets
    def write(self, text):
        for t in self.targets:
            t.write(text)
    def flush(self):
        for t in self.targets:
            t.flush()

with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    writer = DualWriter(f, os.sys.stdout)
    print("\n" + "=" * 80, file=writer)
    print("### Overall Backtest Performance Summary ###", file=writer)
    print("=" * 80, file=writer)

    total_trades = len(df)
    wins = df[df['net_pnl'] > 0]
    losses = df[df['net_pnl'] <= 0]
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    total_net_pnl = df['net_pnl'].sum()
    total_return_pct = (total_net_pnl / INITIAL_CAPITAL) * 100

    gross_profit = wins['net_pnl'].sum()
    gross_loss = abs(losses['net_pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

    avg_win_pnl = wins['net_pnl'].mean() if not wins.empty else 0
    avg_loss_pnl = losses['net_pnl'].mean() if not losses.empty else 0
    avg_win_pct = wins['net_return_pct'].mean() if not wins.empty else 0
    avg_loss_pct = losses['net_return_pct'].mean() if not losses.empty else 0
    expectancy = (win_rate/100 * avg_win_pnl) + ((100-win_rate)/100 * avg_loss_pnl)
    expectancy_pct = (win_rate/100 * avg_win_pct) + ((100-win_rate)/100 * avg_loss_pct)

    print(f"{'Total Trades:':<30} {total_trades}", file=writer)
    print(f"{'Total Net P&L:':<30} ₹{total_net_pnl:,.2f}", file=writer)
    print(f"{'Approx Total Return %:':<30} {total_return_pct:.2f}%", file=writer)
    print("-" * 40, file=writer)
    print(f"{'Win Rate:':<30} {win_rate:.2f}%", file=writer)
    print(f"{'Profit Factor:':<30} {profit_factor:.2f}", file=writer)
    print(f"{'Avg Win P&L:':<30} ₹{avg_win_pnl:,.2f} ({avg_win_pct:.2f}%)", file=writer)
    print(f"{'Avg Loss P&L:':<30} ₹{avg_loss_pnl:,.2f} ({avg_loss_pct:.2f}%)", file=writer)
    print(f"{'Avg Holding Period (Days):':<30} {df['holding_period_days'].mean():.1f}", file=writer)
    print(f"{'Expectancy per Trade:':<30} ₹{expectancy:,.2f} ({expectancy_pct:.2f}%)", file=writer)

    # --- Performance by Holding Period ---
    print("\n" + "=" * 80, file=writer)
    print("### Performance by Holding Period ###", file=writer)
    print("=" * 80, file=writer)

    bins = [0, 15, 30, 45, 60, 90, float('inf')]
    labels = ['1-15 Days', '16-30 Days', '31-45 Days', '46-60 Days', '61-90 Days', '91+ Days']
    df['holding_period_group'] = pd.cut(df['holding_period_days'], bins=bins, labels=labels, right=True, include_lowest=True)

    grouped_holding = df.groupby('holding_period_group', observed=False).agg(
        num_trades=('ticker', 'size'),
        total_pnl=('net_pnl', 'sum'),
        win_rate=('net_pnl', lambda x: (x > 0).mean() * 100),
        avg_return_pct=('net_return_pct', 'mean'),
        median_return_pct=('net_return_pct', 'median'),
        profit_factor=('net_pnl', lambda x: x[x > 0].sum() / abs(x[x <= 0].sum())
                       if abs(x[x <= 0].sum()) > 0 else (float('inf') if x[x > 0].sum() > 0 else 0))
    ).reset_index()

    print(grouped_holding.to_string(float_format="%.2f", index=False), file=writer)

    # --- Performance by Closing Reason ---
    print("\n" + "=" * 80, file=writer)
    print("### Performance by Closing Reason ###", file=writer)
    print("=" * 80, file=writer)

    grouped_reason = df.groupby('closing_reason').agg(
        num_trades=('ticker', 'size'),
        total_pnl=('net_pnl', 'sum'),
        win_rate=('net_pnl', lambda x: (x > 0).mean() * 100),
        avg_return_pct=('net_return_pct', 'mean'),
        median_return_pct=('net_return_pct', 'median'),
        profit_factor=('net_pnl', lambda x: x[x > 0].sum() / abs(x[x <= 0].sum())
                       if abs(x[x <= 0].sum()) > 0 else (float('inf') if x[x > 0].sum() > 0 else 0))
    ).sort_values('total_pnl', ascending=False).reset_index()

    print(grouped_reason.to_string(float_format="%.2f", index=False), file=writer)

    # --- Performance by Signal Type ---
    print("\n" + "=" * 80, file=writer)
    print("### Performance by Signal Type ###", file=writer)
    print("=" * 80, file=writer)

    if 'signal' in df.columns:
        grouped_signal = df.groupby('signal').agg(
            num_trades=('ticker', 'size'),
            total_pnl=('net_pnl', 'sum'),
            win_rate=('net_pnl', lambda x: (x > 0).mean() * 100),
            avg_return_pct=('net_return_pct', 'mean'),
            median_return_pct=('net_return_pct', 'median'),
            profit_factor=('net_pnl', lambda x: x[x > 0].sum() / abs(x[x <= 0].sum())
                           if abs(x[x <= 0].sum()) > 0 else (float('inf') if x[x > 0].sum() > 0 else 0))
        ).reset_index()
        print(grouped_signal.to_string(float_format="%.2f", index=False), file=writer)
    else:
        print("Column 'signal' not found in CSV.", file=writer)

    # --- Biggest Winners and Losers ---
    print("\n" + "=" * 80, file=writer)
    print("### Top 5 Winning Trades (by P&L) ###", file=writer)
    print("=" * 80, file=writer)
    print(df.nlargest(5, 'net_pnl')[['ticker', 'signal', 'open_date', 'close_date',
          'holding_period_days', 'net_pnl', 'net_return_pct']].to_string(index=False), file=writer)

    print("\n" + "=" * 80, file=writer)
    print("### Top 5 Losing Trades (by P&L) ###", file=writer)
    print("=" * 80, file=writer)
    print(df.nsmallest(5, 'net_pnl')[['ticker', 'signal', 'open_date', 'close_date',
          'holding_period_days', 'net_pnl', 'net_return_pct']].to_string(index=False), file=writer)

    print("\n" + "=" * 80, file=writer)
    print("### Analysis Complete ###", file=writer)
    print("=" * 80, file=writer)

print(f"\nAnalysis summary saved to: {REPORT_FILE}")
