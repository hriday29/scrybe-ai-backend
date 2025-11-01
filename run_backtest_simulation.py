"""
run_backtest_simulation.py

Purpose
- Modular, risk-based backtest simulator that sizes positions by risk % of current equity (compounding),
    computes absolute and percentage P&L, and writes Closed trades to the `performance` collection.

How it fits
- Phase 2 of the backtest flow: after Phase 1 writes predictions for a `batch_id`, this simulator
    replays entries/exits per trade plan (SL/target/time) and persists results used by reporting.

Main role
- Convert signal documents into portfolio outcomes with realistic costs and position sizing; update
    prediction statuses and prepare data for `analyze_backtest_db` and the final report.

Notes
- Additive module that leaves earlier backtesting logic intact.
- Relies on saved predictions for the given `batch_id` and historical price data in cache/backends.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

import config
import database_manager
from logger_config import log
import data_retriever


# -------- Data Models ---------
@dataclass
class Position:
    prediction_id: any
    ticker: str
    signal: str  # 'BUY' or 'SELL'/'SHORT'
    entry_price: float
    num_shares: int
    stop_loss: float
    target: Optional[float]
    open_date: pd.Timestamp
    holding_period: int
    strategy: str
    batch_id: str


def _calc_closed_trade(position: Position, closing_price: float, close_reason: str, close_date: pd.Timestamp) -> tuple[dict, float]:
    """Compute gross/net PnL and build a performance document with absolute P&L.

    Returns (performance_doc, net_pnl)
    """
    entry_price = position.entry_price
    num_shares = position.num_shares
    signal = position.signal

    if signal in ("SELL", "SHORT"):
        gross_pnl = (entry_price - closing_price) * num_shares
    else:  # BUY
        gross_pnl = (closing_price - entry_price) * num_shares

    costs = config.BACKTEST_CONFIG
    turnover = (entry_price * num_shares) + (closing_price * num_shares)
    brokerage = turnover * (costs['brokerage_pct'] / 100.0)
    stt = turnover * (costs['stt_pct'] / 100.0)
    other_charges = turnover * (costs['slippage_pct'] / 100.0)
    total_costs = brokerage + stt + other_charges
    net_pnl = gross_pnl - total_costs

    invested_capital = entry_price * num_shares
    net_return_pct = (net_pnl / invested_capital) * 100 if invested_capital > 0 else 0.0

    performance_doc = {
        "prediction_id": position.prediction_id,
        "ticker": position.ticker,
        "strategy": position.strategy,
        "signal": "SHORT" if position.signal == "SELL" else position.signal,
        "status": "Closed",
        "open_date": position.open_date.to_pydatetime(),
        "close_date": close_date.to_pydatetime(),
        "closing_reason": close_reason,
        # Absolute and relative P&L
        "net_pnl": round(net_pnl, 2),
        "net_return_pct": round(net_return_pct, 2),
        # Helpful additional fields (non-breaking additions)
        "entry_price": round(entry_price, 4),
        "exit_price": round(closing_price, 4),
        "num_shares": int(num_shares),
        "invested_capital": round(invested_capital, 2),
        "batch_id": position.batch_id,
    }
    return performance_doc, net_pnl


def _preload_history(tickers: List[str], end_date: Optional[str]) -> Dict[str, pd.DataFrame]:
    cache: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = data_retriever.get_historical_stock_data(ticker, end_date=end_date)
        if df is not None and not df.empty:
            cache[ticker] = df
        else:
            log.warning(f"Historical data missing for {ticker}; skipping trades for this ticker.")
    return cache


def run_backtest_simulation(batch_id: str):
    """
    Risk-based, stateful backtest simulator.
    - Risk per trade is a percent of CURRENT equity (compounded).
    - Shares sized as floor(risk_amount / risk_per_share).
    - Stops/targets/time exits applied.
    - Writes closed trades with net_pnl and net_return_pct to performance collection.
    """
    log.info(f"--- 🚀 Starting Risk-Based Backtest Simulation for Batch: '{batch_id}' ---")

    # Initialize DB and fetch signals
    database_manager.init_db(purpose='scheduler')

    # Support both 'SELL' and 'SHORT' historical values
    signal_filter = {"$in": ["BUY", "SELL", "SHORT"]}
    signals_df = pd.DataFrame(list(database_manager.predictions_collection.find({
        "batch_id": batch_id,
        "signal": signal_filter,
        "status": {"$in": ["open", "Open", "OPEN"]}
    })))

    if signals_df.empty:
        log.warning(f"No actionable signals found for batch '{batch_id}'.")
        database_manager.close_db_connection()
        return

    # Dates and data cache
    signals_df['prediction_date'] = pd.to_datetime(signals_df['prediction_date'])
    unique_tickers = sorted(signals_df['ticker'].unique().tolist())
    last_signal_date = signals_df['prediction_date'].max()
    holding_days = config.APEX_SWING_STRATEGY['holding_period']
    sim_end_date = last_signal_date + pd.Timedelta(days=holding_days + 5)

    history_cache = _preload_history(unique_tickers, end_date=sim_end_date.strftime('%Y-%m-%d'))

    # Drop any signals for which we have no history
    signals_df = signals_df[signals_df['ticker'].isin(history_cache.keys())].copy()
    if signals_df.empty:
        log.warning("No signals left after filtering for available historical data.")
        database_manager.close_db_connection()
        return

    # Portfolio state
    equity = float(config.BACKTEST_PORTFOLIO_CONFIG['initial_capital'])
    risk_pct = float(config.BACKTEST_PORTFOLIO_CONFIG['risk_per_trade_pct']) / 100.0
    max_concurrent = int(config.PORTFOLIO_CONSTRAINTS.get('max_concurrent_trades', 10))

    open_positions: List[Position] = []
    closed_docs: List[dict] = []
    equity_log: List[dict] = []

    bdays = pd.bdate_range(start=signals_df['prediction_date'].min(), end=sim_end_date)

    for day in bdays:
        # 1) Manage open positions: check SL/Target/Time exit
        for pos in open_positions[:]:
            day_data = history_cache[pos.ticker].loc[day] if day in history_cache[pos.ticker].index else None
            if day_data is None:
                continue
            close_reason = None
            close_price = None

            if pos.signal in ("SELL", "SHORT"):
                # Short
                if day_data.high >= pos.stop_loss:
                    close_reason, close_price = "Stop-Loss Hit", pos.stop_loss
                elif pos.target is not None and day_data.low <= pos.target:
                    close_reason, close_price = "Target Hit", pos.target
            else:
                # Long
                if day_data.low <= pos.stop_loss:
                    close_reason, close_price = "Stop-Loss Hit", pos.stop_loss
                elif pos.target is not None and day_data.high >= pos.target:
                    close_reason, close_price = "Target Hit", pos.target

            days_held = (day.date() - pos.open_date.date()).days
            if close_reason is None and days_held >= pos.holding_period:
                close_reason, close_price = "Time Exit", float(day_data.close)

            if close_reason is not None and close_price is not None:
                perf_doc, net_pnl = _calc_closed_trade(pos, float(close_price), close_reason, day)
                equity += net_pnl
                closed_docs.append(perf_doc)
                open_positions.remove(pos)

        # 2) Enter new positions (signals that occur today)
        todays = signals_df[signals_df['prediction_date'].dt.date == day.date()]
        if not todays.empty:
            for _, s in todays.iterrows():
                if len(open_positions) >= max_concurrent:
                    log.warning(
                        f"Skipping {s['ticker']} on {day.date()}: max concurrent positions ({max_concurrent}) reached."
                    )
                    continue

                plan = s.get('tradePlan', {}) or {}
                entry = plan.get('entryPrice')
                stop = plan.get('stopLoss')
                target = plan.get('target')
                if not entry or not stop:
                    continue

                entry = float(entry)
                stop = float(stop)
                target_f = float(target) if target else None

                risk_per_share = abs(entry - stop)
                if risk_per_share <= 0:
                    continue

                risk_amount = equity * risk_pct
                shares = int(risk_amount // risk_per_share)
                if shares <= 0:
                    continue

                pos = Position(
                    prediction_id=s['_id'],
                    ticker=s['ticker'],
                    signal=s['signal'],
                    entry_price=entry,
                    num_shares=shares,
                    stop_loss=stop,
                    target=target_f,
                    open_date=day,
                    holding_period=holding_days,
                    strategy=s.get('strategy', config.APEX_SWING_STRATEGY['name']),
                    batch_id=s['batch_id'],
                )
                open_positions.append(pos)
                log.info(f"ENTER: {pos.signal} {pos.num_shares}x {pos.ticker} @ {pos.entry_price:.2f} (risk {risk_pct*100:.2f}% of ₹{equity:,.0f})")

        # 3) Daily equity mark
        equity_log.append({"date": day, "equity": equity})

    # Persist results
    if closed_docs:
        log.info(f"Writing {len(closed_docs)} closed trades to performance collection for batch '{batch_id}'...")
        # Replace any previous run for this batch id (safe upsert behaviour)
        database_manager.performance_collection.delete_many({"batch_id": batch_id})
        database_manager.performance_collection.insert_many(closed_docs)

        # Mark predictions as processed/Closed
        pred_ids = [d["prediction_id"] for d in closed_docs if d.get("prediction_id") is not None]
        if pred_ids:
            database_manager.predictions_collection.update_many(
                {"_id": {"$in": pred_ids}}, {"$set": {"status": "Closed"}}
            )
        log.info("✅ Batch performance write complete.")
    else:
        log.warning("No positions were closed during the simulation; nothing to write.")

    database_manager.close_db_connection()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run risk-based backtest simulation for a given batch.")
    parser.add_argument("--batch_id", required=True, help="Unique batch identifier to simulate.")
    args = parser.parse_args()
    run_backtest_simulation(batch_id=args.batch_id)
