# run_daily_jobs.py (v1.2 - Production Ready & Fully Synced)

import argparse
import json
import time
import random
from datetime import datetime, timezone
from typing import List, Dict, Tuple

import pandas as pd
import pandas_ta as ta

import config
import data_retriever
import database_manager
import quantitative_screener
import sector_analyzer
from ai_analyzer import AIAnalyzer
from logger_config import log
from utils import APIKeyManager
from sector_analyzer import CORE_SECTOR_INDICES, BENCHMARK_INDEX


# ------------------------------
# Helpers ported from main_orchestrator.py
# ------------------------------

def _build_live_context(ticker: str, point_in_time_data: pd.DataFrame, market_regime: str, nifty_data: pd.DataFrame) -> dict:
    """
    Constructs the complete, point-in-time context dictionary for the AI,
    mirroring _build_backtest_context from main_orchestrator.py.
    """
    failed_indicators = []
    try:
        data = point_in_time_data.copy()

        try:
            macd = data.ta.macd()
            if macd is not None and not macd.empty:
                data = data.join(macd)
        except Exception as e:
            failed_indicators.append("MACD")
            log.warning(f"MACD failed for {ticker}: {e}")

        try:
            bbands = data.ta.bbands()
            if bbands is not None and not bbands.empty:
                data = data.join(bbands)
        except Exception as e:
            failed_indicators.append("Bollinger Bands")
            log.warning(f"Bollinger Bands failed for {ticker}: {e}")

        try:
            supertrend = data.ta.supertrend()
            if supertrend is not None and not supertrend.empty:
                data = data.join(supertrend)
        except Exception as e:
            failed_indicators.append("Supertrend")
            log.warning(f"Supertrend failed for {ticker}: {e}")

        try:
            rsi = data.ta.rsi(length=14)
            if rsi is not None and not rsi.empty:
                data = data.join(rsi)
        except Exception as e:
            failed_indicators.append("RSI")
            log.warning(f"RSI failed for {ticker}: {e}")

        try:
            adx = data.ta.adx(length=14)
            if adx is not None and not adx.empty:
                data = data.join(adx)
        except Exception as e:
            failed_indicators.append("ADX")
            log.warning(f"ADX failed for {ticker}: {e}")

        latest_row = data.iloc[-1]

        technicals = {"daily_close": latest_row.get("close", None)}

        if "RSI_14" in data.columns:
            technicals["RSI_14"] = f"{latest_row['RSI_14']:.2f}"

        if "ADX_14" in data.columns:
            technicals["ADX_14_trend_strength"] = f"{latest_row['ADX_14']:.2f}"

        if "MACD_12_26_9" in data.columns and "MACDs_12_26_9" in data.columns:
            technicals["MACD_status"] = {
                "value": f"{latest_row['MACD_12_26_9']:.2f}",
                "signal_line": f"{latest_row['MACDs_12_26_9']:.2f}",
                "interpretation": (
                    "Bullish Crossover" if latest_row['MACD_12_26_9'] > latest_row['MACDs_12_26_9'] else "Bearish Crossover"
                ),
            }

        if all(col in data.columns for col in ["BBU_20_2.0", "BBL_20_2.0", "BBM_20_2.0"]):
            technicals["bollinger_bands"] = {
                "price_position": (
                    "Above Upper Band"
                    if latest_row['close'] > latest_row['BBU_20_2.0']
                    else "Below Lower Band"
                    if latest_row['close'] < latest_row['BBL_20_2.0']
                    else "Inside Bands"
                ),
                "upper_band": f"{latest_row['BBU_20_2.0']:.2f}",
                "lower_band": f"{latest_row['BBL_20_2.0']:.2f}",
                "band_width_pct": f"{((latest_row['BBU_20_2.0'] - latest_row['BBL_20_2.0']) / latest_row['BBM_20_2.0']) * 100:.2f}%",
            }

        if "SUPERT_7_3.0" in data.columns and "SUPERTd_7_3.0" in data.columns:
            technicals["supertrend_7_3"] = {
                "trend": "Uptrend" if latest_row['SUPERTd_7_3.0'] == 1 else "Downtrend",
                "value": f"{latest_row['SUPERT_7_3.0']:.2f}",
            }

        if failed_indicators:
            technicals["errors"] = f"Failed to calculate: {', '.join(failed_indicators)}"

        if len(technicals) <= 1:
            raise ValueError("No valid technical indicators were generated.")

    except Exception as e:
        log.error(
            f"Indicator calculation failed for {ticker} on {point_in_time_data.index[-1].strftime('%Y-%m-%d')}: {e}"
        )
        technicals = {
            "error": f"Indicator calculation failed. Failed indicators: {', '.join(failed_indicators) if failed_indicators else 'All'}"
        }

    # Relative Strength
    try:
        nifty_slice = nifty_data.loc[:point_in_time_data.index[-1]]
        nifty_5d_change = (nifty_slice['close'].iloc[-1] / nifty_slice['close'].iloc[-6] - 1) * 100
        stock_5d_change = (point_in_time_data['close'].iloc[-1] / point_in_time_data['close'].iloc[-6] - 1) * 100
        relative_strength = "Outperforming" if stock_5d_change > nifty_5d_change else "Underperforming"
    except (IndexError, KeyError):
        relative_strength = "Data Not Available"

    context = {
        "layer_1_macro_context": {"nifty_50_regime": market_regime},
        "layer_2_relative_strength": {"relative_strength_vs_nifty50": relative_strength},
        "layer_3_fundamental_moat": data_retriever.get_fundamental_proxies(point_in_time_data),
        "layer_4_technicals": technicals,
        "layer_5_options_sentiment": {"sentiment": "Unavailable in backtest"},
        "layer_6_news_catalyst": {"summary": "Unavailable in backtest"},
    }
    return context


def sanitize_context(context: dict) -> dict:
    sanitized_context = {}
    for layer, details in context.items():
        if isinstance(details, dict):
            sanitized_details = {}
            for k, v in details.items():
                if not v or str(v).strip().lower() in [
                    "unavailable",
                    "n/a",
                    "none",
                    "null",
                    "unavailable in backtest",
                ]:
                    sanitized_details[k] = "Data Not Available"
                else:
                    sanitized_details[k] = v
            sanitized_context[layer] = sanitized_details
        else:
            sanitized_context[layer] = details
    return sanitized_context


def _get_30_day_performance_review(current_day: pd.Timestamp, batch_id: str) -> str:
    thirty_days_prior = current_day - pd.Timedelta(days=30)
    query = {
        "batch_id": batch_id,
        "close_date": {
            "$gte": thirty_days_prior.to_pydatetime(),
            "$lt": current_day.to_pydatetime(),
        },
    }
    recent_trades = list(database_manager.performance_collection.find(query))
    if not recent_trades:
        return "No trading history in the last 30 days."
    df = pd.DataFrame(recent_trades)
    total_signals = len(df)
    win_rate = (df['net_return_pct'] > 0).mean() * 100
    gross_profit = df[df['net_return_pct'] > 0]['net_return_pct'].sum()
    gross_loss = abs(df[df['net_return_pct'] < 0]['net_return_pct'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    review = (
        f"30-Day Performance Review:\n- Total Signals: {total_signals}\n- Win Rate: {win_rate:.1f}%\n- Profit Factor: {profit_factor:.2f}"
    )
    return review


def _get_1_day_tactical_lookback(current_day: pd.Timestamp, ticker: str, batch_id: str) -> str:
    previous_trading_day = current_day - pd.Timedelta(days=3)
    query = {
        "batch_id": batch_id,
        "ticker": ticker,
        "prediction_date": {
            "$gte": previous_trading_day.to_pydatetime(),
            "$lt": current_day.to_pydatetime(),
        },
    }
    last_analysis = database_manager.predictions_collection.find_one(
        query, sort=[("prediction_date", -1)]
    )
    if not last_analysis:
        return "No analysis for this stock on the previous trading day."
    lookback = (
        f"Previous Day's Note ({ticker}):\n- Signal: {last_analysis.get('signal', 'N/A')}\n- Scrybe Score: {last_analysis.get('scrybeScore', 0)}\n- Key Insight: \"{last_analysis.get('keyInsight', 'N/A')}\""
    )
    return lookback


def _get_per_stock_trade_history(ticker: str, batch_id: str, current_day: pd.Timestamp) -> str:
    query = {
        "batch_id": batch_id,
        "ticker": ticker,
        "close_date": {"$lt": current_day.to_pydatetime()},
    }
    recent_trades = list(
        database_manager.performance_collection.find(query).sort("close_date", -1).limit(3)
    )
    if not recent_trades:
        return "No recent trade history for this stock."
    history_lines = [
        f"{i+1}. Signal: {t.get('signal')}, Outcome: {t.get('net_return_pct'):.2f}% ({t.get('closing_reason')})"
        for i, t in enumerate(recent_trades)
    ]
    return "\n".join(history_lines)


# ------------------------------
# Daily Tasks (Screen, Analyze, Manage) - Self-Contained
# ------------------------------

def run_screening_task(output_file: str):
    """
    Mirrors orchestrator: diagnose regime/volatility, compute strong sectors, then choose screener.
    Self-contained - no database connection needed, only fetches API data and writes JSON.
    """
    log.info("--- Starting Daily Screening Task (Synced & Self-Contained) ---")
    
    try:
        # Load sector/benchmark data cache
        index_symbols = list(CORE_SECTOR_INDICES.values()) + [BENCHMARK_INDEX]
        index_cache = {
            ticker: data_retriever.get_historical_stock_data(ticker) for ticker in index_symbols
        }

        # VIX and volatility regime
        vix_data = data_retriever.get_historical_stock_data("^INDIAVIX")
        vix_slice = vix_data.loc[: pd.Timestamp.now()] if vix_data is not None else None
        volatility_regime = data_retriever.get_volatility_regime(vix_slice)

        # Market regime (trend)
        nifty = index_cache.get(BENCHMARK_INDEX)
        market_regime = (
            data_retriever.calculate_regime_from_data(nifty.loc[: pd.Timestamp.now()])
            if nifty is not None
            else "Neutral"
        )

        # Sector strength (point in time = now)
        strong_sectors = sector_analyzer.get_top_performing_sectors(
            index_cache, pd.Timestamp.now()
        )

        # Load stock data cache for NIFTY 50 universe (live)
        stock_universe = config.NIFTY_50_TICKERS
        stock_cache = {
            t: data_retriever.get_historical_stock_data(t) for t in stock_universe
        }

        # Screener selection synced with orchestrator logic
        if market_regime == "Bearish":
            log.warning("Market regime is Bearish. Mean-reversion setups only.")
            stocks_for_today = quantitative_screener.screen_for_mean_reversion(
                strong_sectors, stock_cache, pd.Timestamp.now()
            )
        elif market_regime == "Bullish":
            if volatility_regime == "High-Risk":
                stocks_for_today = quantitative_screener.screen_for_pullbacks(
                    strong_sectors, stock_cache, pd.Timestamp.now()
                )
            else:
                stocks_for_today = quantitative_screener.screen_for_momentum(
                    strong_sectors, stock_cache, pd.Timestamp.now()
                )
        else:  # Neutral
            stocks_for_today = quantitative_screener.screen_for_mean_reversion(
                strong_sectors, stock_cache, pd.Timestamp.now()
            )

        if not stocks_for_today:
            log.warning("No stocks passed the screening funnel today.")
            watchlist: List[Dict] = []
        else:
            watchlist = [{"ticker": t, "reason": r} for t, r in stocks_for_today]

        with open(output_file, "w") as f:
            json.dump(watchlist, f)

        log.info(
            f"✅ Screening complete. Regime={market_regime}, Vol={volatility_regime}. Saved {len(watchlist)} candidates to {output_file}."
        )

    except Exception as e:
        log.error(f"Screening task failed: {e}")
        # Create empty watchlist on error to ensure downstream jobs don't fail
        with open(output_file, "w") as f:
            json.dump([], f)

    
    log.info("✅ Screening task complete - no database connection needed.")


def run_analysis_task(input_file: str, batch_id: str):
    """
    Processes the watchlist, builds full context, runs AI analysis with retry/backoff,
    applies veto logic, and saves to live collection with trade plan fields.
    Self-contained with its own database connection management.
    """
    log.info(f"--- Starting AI Analysis Task from {input_file} (Synced & Self-Contained) ---")
    
    database_manager.init_db("analysis")
    
    try:
        try:
            with open(input_file, "r") as f:
                watchlist = json.load(f)
        except FileNotFoundError:
            log.error(f"Watchlist file {input_file} not found. Cannot proceed.")
            return

        if not watchlist:
            log.info("Watchlist is empty. Nothing to analyze.")
            return

        # Keys management
        key_manager = APIKeyManager(api_keys=config.GEMINI_API_KEY_POOL)
        analyzer = AIAnalyzer(api_key=key_manager.get_key())

        # Shared overlays
        vix_data = data_retriever.get_historical_stock_data("^INDIAVIX")
        latest_vix = vix_data.iloc[-1]["close"] if vix_data is not None and not vix_data.empty else None
        market_is_high_risk = latest_vix is not None and latest_vix > getattr(config, "HIGH_RISK_VIX_THRESHOLD", 9999)

        benchmark_data = data_retriever.get_historical_stock_data(BENCHMARK_INDEX)
        market_regime = data_retriever.calculate_regime_from_data(benchmark_data) if benchmark_data is not None else "Neutral"

        portfolio_config = getattr(config, "BACKTEST_PORTFOLIO_CONFIG", {"initial_capital": 100000, "risk_per_trade_pct": 1.0})
        active_strategy = config.APEX_SWING_STRATEGY

        for item in watchlist:
            ticker = item["ticker"]
            screener_reason = item["reason"]
            log.info(f"--- Analyzing {ticker} | Reason: {screener_reason} ---")

            try:
                # Historical data
                point_in_time_data = data_retriever.get_historical_stock_data(ticker)
                if point_in_time_data is None or len(point_in_time_data) < 252:
                    log.warning(f"Skipping {ticker}: insufficient historical data.")
                    continue

                # Build context
                strategic_review = _get_30_day_performance_review(pd.Timestamp.now(), batch_id)
                tactical_lookback = _get_1_day_tactical_lookback(pd.Timestamp.now(), ticker, batch_id)
                per_stock_history = _get_per_stock_trade_history(ticker, batch_id, pd.Timestamp.now())

                full_context = _build_live_context(
                    ticker,
                    point_in_time_data.loc[: pd.Timestamp.now()].copy(),
                    market_regime,
                    benchmark_data,
                )
                sanitized_full_context = sanitize_context(full_context)

                # ATR
                point_in_time_data.ta.atr(length=14, append=True)
                atr_at_prediction = point_in_time_data["ATRr_14"].iloc[-1]
                if pd.isna(atr_at_prediction):
                    log.warning(f"ATR NaN for {ticker}. Skipping.")
                    continue

                # AI Analysis with retry/backoff
                final_analysis = None
                max_attempts = len(config.GEMINI_API_KEY_POOL)
                current_attempt, delay = 0, 5

                log.info(f"Analyzing {ticker} with Primary Model ({config.PRO_MODEL})")
                while current_attempt < max_attempts:
                    try:
                        final_analysis = analyzer.get_apex_analysis(
                            ticker,
                            sanitized_full_context,
                            strategic_review,
                            tactical_lookback,
                            per_stock_history,
                            model_name=config.PRO_MODEL,
                            screener_reason=screener_reason,
                        )
                        if final_analysis:
                            log.info(f"✅ Got analysis for {ticker} (Primary).")
                            final_analysis["modelUsed"] = "pro"
                            break
                    except Exception as e:
                        log.error(f"Primary attempt #{current_attempt + 1} for {ticker} failed: {e}")
                        if any(x in str(e).lower() for x in ["429", "quota", "500"]) or isinstance(e, ValueError):
                            analyzer = AIAnalyzer(api_key=key_manager.rotate_key())
                            current_attempt += 1
                            log.info(f"Backing off {delay:.2f}s before retry...")
                            time.sleep(delay)
                            delay = min(delay * 2, 60) + random.uniform(0, 1)
                        else:
                            break

                if not final_analysis:
                    log.warning(f"⚡ Fallback model ({config.FLASH_MODEL}) for {ticker}.")
                    try:
                        final_analysis = analyzer.get_apex_analysis(
                            ticker,
                            sanitized_full_context,
                            strategic_review,
                            tactical_lookback,
                            per_stock_history,
                            model_name=config.FLASH_MODEL,
                            screener_reason=screener_reason,
                        )
                        if final_analysis:
                            log.info(f"✅ Got analysis for {ticker} (Fallback).")
                            final_analysis["modelUsed"] = "flash"
                    except Exception as e:
                        log.error(f"Fallback also failed for {ticker}: {e}")
                        continue

                log.info("Pacing API calls (35s delay).")
                time.sleep(35)

                if not final_analysis:
                    continue

                # --- Veto logic ---
                original_signal = final_analysis.get("signal")
                scrybe_score = final_analysis.get("scrybeScore", 0)
                final_signal, veto_reason = original_signal, None

                entry_price = point_in_time_data["close"].iloc[-1]
                potential_risk_per_share = active_strategy["stop_loss_atr_multiplier"] * atr_at_prediction
                potential_reward_per_share = potential_risk_per_share * active_strategy["profit_target_rr_multiple"]
                rr_ratio = potential_reward_per_share / potential_risk_per_share if potential_risk_per_share > 0 else 0

                if original_signal in ["BUY", "SELL"]:
                    is_conviction_ok = abs(scrybe_score) >= active_strategy["min_conviction_score"]
                    is_regime_ok = (
                        (original_signal == "BUY" and market_regime != "Bearish")
                        or (original_signal == "SELL" and market_regime != "Bullish")
                    )
                    is_rr_ok = rr_ratio >= active_strategy["profit_target_rr_multiple"]

                    if original_signal == "BUY" and market_is_high_risk:
                        final_signal, veto_reason = "HOLD", f"VETOED BUY: High VIX {latest_vix:.2f}"
                    elif not is_regime_ok:
                        final_signal, veto_reason = "HOLD", f"VETOED {original_signal}: Contradicts regime ({market_regime})"
                    elif not is_conviction_ok:
                        final_signal, veto_reason = "HOLD", f"VETOED {original_signal}: Conviction {scrybe_score}"
                    elif not is_rr_ok:
                        final_signal, veto_reason = "HOLD", f"VETOED {original_signal}: Poor RR {rr_ratio:.2f}R"

                # --- Save doc ---
                prediction_doc = final_analysis.copy()
                prediction_doc["signal"] = final_signal

                if final_signal in ["BUY", "SELL"]:
                    num_shares_to_trade = int(
                        (portfolio_config["initial_capital"] * (portfolio_config["risk_per_trade_pct"] / 100.0))
                        / potential_risk_per_share
                    ) if potential_risk_per_share > 0 else 0
                    position_size_pct = (
                        (num_shares_to_trade * entry_price / portfolio_config["initial_capital"]) * 100
                        if portfolio_config["initial_capital"] > 0 else 0
                    )
                    stop_loss_price = (
                        entry_price - potential_risk_per_share if final_signal == "BUY" else entry_price + potential_risk_per_share
                    )
                    target_price = (
                        entry_price + potential_reward_per_share if final_signal == "BUY" else entry_price - potential_reward_per_share
                    )

                    prediction_doc["tradePlan"] = {
                        "entryPrice": round(entry_price, 2),
                        "target": round(target_price, 2),
                        "stopLoss": round(stop_loss_price, 2),
                    }
                    prediction_doc.update({
                        "ticker": ticker,
                        "prediction_date": datetime.now().replace(tzinfo=timezone.utc),
                        "price_at_prediction": entry_price,
                        "status": "open",
                        "strategy": "ApexSwing_v5_HighConviction",
                        "atr_at_prediction": float(atr_at_prediction),
                        "position_size_pct": float(position_size_pct),
                        "batch_id": batch_id,
                    })
                else:
                    prediction_doc.update({
                        "ticker": ticker,
                        "prediction_date": datetime.now().replace(tzinfo=timezone.utc),
                        "price_at_prediction": entry_price,
                        "status": "vetoed" if veto_reason else "hold",
                        "strategy": "ApexSwing_v5_HighConviction",
                        "atr_at_prediction": float(atr_at_prediction),
                        "veto_reason": veto_reason,
                        "batch_id": batch_id,
                    })

                database_manager.save_live_prediction(prediction_doc)

            except Exception as e:
                log.error(f"Analysis error for {ticker}: {e}", exc_info=True)
                if any(x in str(e).lower() for x in ["429", "quota"]):
                    analyzer = AIAnalyzer(api_key=key_manager.rotate_key())
                continue

    except Exception as e:
        log.error(f"Error in analysis task: {e}", exc_info=True)
    finally:
        database_manager.close_db_connection()
        log.info("✅ AI analysis complete (Synced & Self-Contained).")

def manage_open_trades():
    """
    Manage open live trades saved by run_analysis_task using synchronized tradePlan fields.
    Self-contained with its own database connection management.
    """
    log.info("--- Starting Open Trade Management Task (Synced & Self-Contained) ---")
    
    # Initialize database connection for this task
    database_manager.init_db('analysis')
    
    try:
        # Find open live predictions with a tradePlan
        open_positions = database_manager.live_predictions_collection.find({"status": "open", "tradePlan": {"$ne": None}})

        for position in open_positions:
            try:
                ticker = position['ticker']
                plan = position['tradePlan']
                current = data_retriever.get_live_financial_data(ticker)
                if not current or not current.get('curatedData'):
                    log.warning(f"No live quote for {ticker}; skipping.")
                    continue

                current_price = current['curatedData']['currentPrice']
                stop_loss = float(plan['stopLoss']) if plan.get('stopLoss') is not None else None
                target = float(plan['target']) if plan.get('target') is not None else None

                close_reason, close_price = None, None
                side = position.get('signal', 'BUY')  # default BUY if missing

                if side == 'BUY':
                    if current_price <= stop_loss:
                        close_reason, close_price = "Stop-Loss Hit", stop_loss
                    elif current_price >= target:
                        close_reason, close_price = "Target Hit", target
                
                elif side == 'SELL': # Correctly de-dented to be at the same level as the 'if'
                    if current_price >= stop_loss:
                        close_reason, close_price = "Stop-Loss Hit", stop_loss
                    elif current_price <= target:
                        close_reason, close_price = "Target Hit", target

                # Optional: expiry check if your schema includes it
                expiry = position.get('expiry_date')
                if not close_reason and expiry:
                    now_utc = datetime.now(timezone.utc)
                    if now_utc > expiry:
                        close_reason, close_price = "Expired", current_price

                if close_reason:
                    log.info(f"CLOSING TRADE: {ticker} due to {close_reason} @ {close_price}")
                    database_manager.close_live_trade(ticker, position, close_reason, close_price)

            except Exception as e:
                log.error(f"Trade management error for {position.get('ticker', 'UNKNOWN')}: {e}")
                continue

    except Exception as e:
        log.error(f"Error in trade management task: {e}", exc_info=True)
    finally:
        # Clean up database connection
        database_manager.close_db_connection()
        log.info("✅ Open trade management complete (Synced & Self-Contained).")


# ------------------------------
# CLI Entrypoint
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrybe AI Daily Job Runner (Synced with Orchestrator)")
    parser.add_argument('--task', required=True, choices=['manage', 'screen', 'analyze'], help='The specific task to run.')
    parser.add_argument('--watchlist', default='watchlist.json', help='Path to watchlist JSON for analyze task.')
    parser.add_argument('--batch_id', default='LIVE', help='Batch ID to tag live runs for parity with backtests.')
    args = parser.parse_args()

    log.info(f"--- Executing Task: {args.task} ---")
    
    # DB connections are now handled inside each task function
    if args.task == 'manage':
        manage_open_trades()
    elif args.task == 'screen':
        run_screening_task(output_file=args.watchlist)
    elif args.task == "analyze":
        live_batch_id = f"LIVE_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        run_analysis_task(input_file=args.watchlist, batch_id=live_batch_id)
    
    log.info(f"--- Task '{args.task}' Completed ---")