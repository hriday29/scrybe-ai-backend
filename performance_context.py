# performance_context.py
"""
performance_context.py

Purpose
- Provides lightweight context snippets for reports and UIs, such as a 30-day
    strategy summary and the last few closed trades for a ticker within a batch.

How it fits
- Consumed by the analysis pipeline or API layer to add recent-history blurbs to
    AI prompts or human-readable summaries during reporting.

Main role
- Query performance data and render compact, human-friendly strings to embed in
    downstream outputs without heavy post-processing.
"""
from logger_config import log
import database_manager
import pandas as pd

def get_strategy_performance_summary(batch_id: str, point_in_time: pd.Timestamp) -> str:
    """
    Queries the database for the last 30 days of trades within a backtest
    to generate a high-level performance summary.
    """
    try:
        thirty_days_prior = point_in_time - pd.Timedelta(days=30)
        query = {
            "batch_id": batch_id,
            "close_date": {"$gte": thirty_days_prior.to_pydatetime(), "$lt": point_in_time.to_pydatetime()}
        }
        
        # This function assumes the database connection is already handled by the calling pipeline.
        recent_trades = list(database_manager.performance_collection.find(query))
        
        if not recent_trades:
            return "No trading history in the last 30 days."

        df = pd.DataFrame(recent_trades)
        total_signals = len(df)
        win_rate = (df['net_return_pct'] > 0).mean() * 100
        avg_win_pct = df[df['net_return_pct'] > 0]['net_return_pct'].mean()
        avg_loss_pct = df[df['net_return_pct'] < 0]['net_return_pct'].mean()
        
        review = (
            f"30-Day Strategy Review: Total Trades={total_signals}, Win Rate={win_rate:.1f}%, "
            f"Avg Win={avg_win_pct:.2f}%, Avg Loss={avg_loss_pct:.2f}%"
        )
        return review
    except Exception as e:
        log.warning(f"Could not generate strategy performance summary: {e}")
        return "Error fetching strategy performance."


def get_ticker_trade_history(ticker: str, batch_id: str, point_in_time: pd.Timestamp) -> str:
    """
    Queries the database for the last 3 closed trades for a specific stock
    within a backtest.
    """
    try:
        query = {
            "batch_id": batch_id,
            "ticker": ticker,
            "close_date": {"$lt": point_in_time.to_pydatetime()}
        }
        recent_trades = list(database_manager.performance_collection.find(query).sort("close_date", -1).limit(3))
        
        if not recent_trades:
            return "No recent trade history for this stock."
        
        history_lines = [
            f"- Outcome: {t.get('net_return_pct', 0):.2f}% ({t.get('closing_reason')})"
            for t in recent_trades
        ]
        return "Recent Trade History for this Stock:\n" + "\n".join(history_lines)
    except Exception as e:
        log.warning(f"Could not generate ticker trade history for {ticker}: {e}")
        return "Error fetching ticker trade history."