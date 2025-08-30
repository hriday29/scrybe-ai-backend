# analyze_batch_results.py
import pandas as pd
import database_manager
import argparse
from logger_config import log

def analyze_results(batch_id: str):
    """
    Fetches results for a specific batch_id and prints a performance
    summary for each strategy within that batch.
    """
    log.info(f"--- 📊 Starting Analysis for Batch: '{batch_id}' ---")

    try:
        database_manager.init_db(purpose='scheduler')

        # Find all closed trades for the specified batch
        query = {"batch_id": batch_id, "status": "Closed"}
        all_trades = list(database_manager.performance_collection.find(query))

        if not all_trades:
            log.warning(f"No closed trades found in the database for batch_id '{batch_id}'.")
            return

        df = pd.DataFrame(all_trades)
        log.info(f"Found a total of {len(df)} closed trades to analyze.")

        # Group the DataFrame by the 'strategy' column
        strategy_groups = df.groupby('strategy')

        print("\n" + "="*80)
        print("### Strategy Performance Leaderboard ###")
        print("="*80)

        for name, group_df in strategy_groups:
            # --- Calculate Core Metrics ---
            total_trades = len(group_df)
            net_pnl = group_df['net_pnl'].sum()
            
            wins = group_df[group_df['net_pnl'] > 0]
            losses = group_df[group_df['net_pnl'] < 0]
            
            win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
            
            gross_profit = wins['net_pnl'].sum()
            gross_loss = abs(losses['net_pnl'].sum())
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            avg_return_pct = group_df['net_return_pct'].mean()
            median_return_pct = group_df['net_return_pct'].median()

            # --- Print the Report for this Strategy ---
            print(f"\n📈 Strategy: [ {name} ]")
            print("-" * 40)
            print(f"{'Total Net P&L:':<25} ₹{net_pnl:,.2f}")
            print(f"{'Total Trades:':<25} {total_trades}")
            print(f"{'Win Rate:':<25} {win_rate:.2f}%")
            print(f"{'Profit Factor:':<25} {profit_factor:.2f}")
            print(f"{'Average Return / Trade:':<25} {avg_return_pct:.2f}%")
            print(f"{'Median Return / Trade:':<25} {median_return_pct:.2f}%")

        print("\n" + "="*80)

    finally:
        database_manager.close_db_connection()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the performance of strategies for a given batch ID.")
    parser.add_argument('--batch_id', required=True, help='The batch_id to analyze.')
    args = parser.parse_args()
    
    analyze_results(args.batch_id)