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

STATE_FILE = 'simulation_state.json'

def _build_backtest_context(ticker: str, point_in_time_data: pd.DataFrame, market_regime: str, nifty_data: pd.DataFrame) -> dict:
    """
    Constructs the complete, point-in-time context dictionary for the AI,
    now with a RICH set of technical indicators.
    """
    # --- Step 1: Calculate all required indicators ---
    try:
        # Use a copy to avoid SettingWithCopyWarning
        data = point_in_time_data.copy()
        
        # Standard indicators
        data.ta.macd(append=True)
        data.ta.bbands(append=True)
        data.ta.supertrend(append=True)
        data.ta.rsi(length=14, append=True) # Keep RSI as it's very useful context
        data.ta.adx(length=14, append=True)
        
        # Get the latest row AFTER all calculations
        latest_row = data.iloc[-1]
        
        # --- Step 2: Create a rich, descriptive technical summary ---
        # Provide not just the value, but a simple interpretation to help the AI.
        technicals = {
            "daily_close": latest_row['close'],
            "RSI_14": f"{latest_row['RSI_14']:.2f}",
            "ADX_14_trend_strength": f"{latest_row['ADX_14']:.2f}",
            "MACD_status": {
                "value": f"{latest_row['MACD_12_26_9']:.2f}",
                "signal_line": f"{latest_row['MACDs_12_26_9']:.2f}",
                "interpretation": "Bullish Crossover" if latest_row['MACD_12_26_9'] > latest_row['MACDs_12_26_9'] else "Bearish Crossover"
            },
            "bollinger_bands": {
                "price_position": "Above Upper Band" if latest_row['close'] > latest_row['BBU_20_2.0'] else \
                                  "Below Lower Band" if latest_row['close'] < latest_row['BBL_20_2.0'] else "Inside Bands",
                "upper_band": f"{latest_row['BBU_20_2.0']:.2f}",
                "lower_band": f"{latest_row['BBL_20_2.0']:.2f}",
                "band_width_pct": f"{((latest_row['BBU_20_2.0'] - latest_row['BBL_20_2.0']) / latest_row['BBM_20_2.0'])*100:.2f}%"
            },
            "supertrend_7_3": {
                "trend": "Uptrend" if latest_row['SUPERTd_7_3.0'] == 1 else "Downtrend",
                "value": f"{latest_row['SUPERT_7_3.0']:.2f}"
            }
        }
    except Exception as e:
        log.error(f"Indicator calculation failed for {ticker} on {point_in_time_data.index[-1].strftime('%Y-%m-%d')}: {e}")
        # Return a fallback context if calculation fails
        technicals = {"error": "Indicator calculation failed."}

    # --- Step 3: Calculate Relative Strength (no changes here) ---
    try:
        nifty_slice = nifty_data.loc[:point_in_time_data.index[-1]]
        nifty_5d_change = (nifty_slice['close'].iloc[-1] / nifty_slice['close'].iloc[-6] - 1) * 100
        stock_5d_change = (point_in_time_data['close'].iloc[-1] / point_in_time_data['close'].iloc[-6] - 1) * 100
        relative_strength = "Outperforming" if stock_5d_change > nifty_5d_change else "Underperforming"
    except (IndexError, KeyError):
        relative_strength = "Data Not Available"

    # --- Step 4: Assemble the final, enriched context dictionary ---
    context = {
        "layer_1_macro_context": {"nifty_50_regime": market_regime},
        "layer_2_relative_strength": {"relative_strength_vs_nifty50": relative_strength},
        "layer_3_fundamental_moat": data_retriever.get_fundamental_proxies(point_in_time_data),
        "layer_4_technicals": technicals, # Use our new rich technicals packet
        "layer_5_options_sentiment": {"sentiment": "Unavailable in backtest"},
        "layer_6_news_catalyst": {"summary": "Unavailable in backtest"}
    }
    return context

def sanitize_context(context: dict) -> dict:
    """Sanitizes the context dictionary to replace null-like values."""
    sanitized_context = {}
    for layer, details in context.items():
        if isinstance(details, dict):
            sanitized_details = {}
            for k, v in details.items():
                # Check for various null-like or placeholder strings
                if not v or str(v).strip().lower() in ["unavailable", "n/a", "none", "null", "unavailable in backtest"]:
                    sanitized_details[k] = "Data Not Available"
                else:
                    sanitized_details[k] = v
            sanitized_context[layer] = sanitized_details
        else:
            # If the layer's value is not a dictionary, keep it as is
            sanitized_context[layer] = details
    return sanitized_context

def _get_30_day_performance_review(current_day: pd.Timestamp, batch_id: str) -> str:
    thirty_days_prior = current_day - pd.Timedelta(days=30)
    query = {"batch_id": batch_id, "close_date": {"$gte": thirty_days_prior.to_pydatetime(), "$lt": current_day.to_pydatetime()}}
    recent_trades = list(database_manager.performance_collection.find(query))
    if not recent_trades: return "No trading history in the last 30 days."
    df = pd.DataFrame(recent_trades)
    total_signals = len(df)
    win_rate = (df['net_return_pct'] > 0).mean() * 100
    gross_profit = df[df['net_return_pct'] > 0]['net_return_pct'].sum()
    gross_loss = abs(df[df['net_return_pct'] < 0]['net_return_pct'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    review = (f"30-Day Performance Review:\n- Total Signals: {total_signals}\n- Win Rate: {win_rate:.1f}%\n- Profit Factor: {profit_factor:.2f}")
    return review

def _get_1_day_tactical_lookback(current_day: pd.Timestamp, ticker: str, batch_id: str) -> str:
    previous_trading_day = current_day - pd.Timedelta(days=3) # Look back 3 days to be safe
    query = {"batch_id": batch_id, "ticker": ticker, "prediction_date": {"$gte": previous_trading_day.to_pydatetime(), "$lt": current_day.to_pydatetime()}}
    last_analysis = database_manager.predictions_collection.find_one(query, sort=[("prediction_date", -1)])
    if not last_analysis: return "No analysis for this stock on the previous trading day."
    lookback = (f"Previous Day's Note ({ticker}):\n- Signal: {last_analysis.get('signal', 'N/A')}\n- Scrybe Score: {last_analysis.get('scrybeScore', 0)}\n"
                f"- Key Insight: \"{last_analysis.get('keyInsight', 'N/A')}\"")
    return lookback

def _get_per_stock_trade_history(ticker: str, batch_id: str, current_day: pd.Timestamp) -> str:
    query = { "batch_id": batch_id, "ticker": ticker, "close_date": {"$lt": current_day.to_pydatetime()} }
    recent_trades = list(database_manager.performance_collection.find(query).sort("close_date", -1).limit(3))
    if not recent_trades: return "No recent trade history for this stock."
    history_lines = [f"{i+1}. Signal: {t.get('signal')}, Outcome: {t.get('net_return_pct'):.2f}% ({t.get('closing_reason')})" for i, t in enumerate(recent_trades)]
    return "\n".join(history_lines)

def save_state(next_day_to_run):
    with open(STATE_FILE, 'w') as f: json.dump({'next_start_date': next_day_to_run.strftime('%Y-%m-%d')}, f)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            log.warning(f"Resuming from saved state. Next day to run: {state['next_start_date']}")
            return state['next_start_date']
    return None

def run_simulation(batch_id: str, start_date: str, end_date: str, is_fresh_run: bool = False):
    """
    Runs the backtest simulation with a dynamic, point-in-time stock universe
    to eliminate survivorship bias using nifty50_historical_constituents.csv.
    """
    log.info(f"### STARTING APEX SIMULATION FOR BATCH: {batch_id} ###")
    log.info(f"Period: {start_date} to {end_date}")

    # --- STEP 1: Load historical constituents file ---
    try:
        constituents_df = pd.read_csv('nifty50_historical_constituents.csv')
        constituents_df['date'] = pd.to_datetime(constituents_df['date'])
        log.info("✅ Successfully loaded 'nifty50_historical_constituents.csv'.")
    except FileNotFoundError:
        log.error("CRITICAL: Constituents file missing. Aborting simulation to avoid survivorship bias.")
        return

    # --- NEW: Define indices required for every simulation run ---
    from sector_analyzer import CORE_SECTOR_INDICES, BENCHMARK_INDEX
    required_indices = list(CORE_SECTOR_INDICES.values()) + [BENCHMARK_INDEX]

    database_manager.init_db(purpose='scheduler')
    if is_fresh_run:
        log.warning("FRESH RUN ENABLED: Deleting all previous data.")
        database_manager.clear_scheduler_data()

    try:
        key_manager = APIKeyManager(api_keys=config.GEMINI_API_KEY_POOL)
        analyzer = AIAnalyzer(api_key=key_manager.get_key())

        # --- STEP 2: Initialize on-demand cache for stocks + indices ---
        point_in_time_data_cache = {}

        # Preload VIX only (others will be pulled dynamically as needed)
        vix_data = data_retriever.get_historical_stock_data("^INDIAVIX", end_date=end_date)
        log.info("✅ Pre-loading of India VIX index data is complete.")

    except Exception as e:
        log.fatal(f"Failed during pre-run initialization. Error: {e}")
        return

    simulation_days = pd.bdate_range(start=start_date, end=end_date)
    portfolio_config = config.BACKTEST_PORTFOLIO_CONFIG
    current_equity = portfolio_config['initial_capital']
    active_strategy = config.APEX_SWING_STRATEGY

    api_call_timestamps = deque()
    RPM_LIMIT = 8

    total_stocks_processed_by_screener = 0
    total_stocks_passed_screener = 0
    total_ai_buy_sell_signals = 0
    total_signals_vetoed = 0
    total_trades_executed = 0

    if not simulation_days.empty:
        log.info(f"\n--- Starting simulation for {len(simulation_days)} trading days ---")
        for i, current_day in enumerate(simulation_days):
            day_str = current_day.strftime('%Y-%m-%d')
            log.info(f"\n--- Simulating Day {i+1}/{len(simulation_days)}: {day_str} ---")

            # --- STEP 3: Determine universe for THIS day ---
            available_dates = constituents_df[constituents_df['date'] <= current_day]['date']
            if available_dates.empty:
                log.warning(f"No constituent data available on or before {day_str}. Skipping day.")
                continue

            latest_constituent_date = available_dates.max()
            stock_universe_for_today = constituents_df[constituents_df['date'] == latest_constituent_date]['ticker'].tolist()
            log.info(f"Using NIFTY50 constituents as of {latest_constituent_date.strftime('%Y-%m-%d')} (Universe size: {len(stock_universe_for_today)})")

            # --- STEP 4: Ensure data for both STOCKS and INDICES is loaded ---
            tickers_to_load_today = stock_universe_for_today + required_indices
            for ticker in tickers_to_load_today:
                if ticker not in point_in_time_data_cache:
                    data = data_retriever.get_historical_stock_data(ticker, end_date=end_date)
                    if data is not None and len(data) > 252:
                        point_in_time_data_cache[ticker] = data

            fundamentally_approved_stocks = stock_universe_for_today
            approved_stock_data_cache = {t: point_in_time_data_cache[t] for t in fundamentally_approved_stocks if t in point_in_time_data_cache}

            # --- NEW: Create context cache (stocks + indices) ---
            full_context_data_cache = approved_stock_data_cache.copy()
            for ticker in required_indices:
                if ticker in point_in_time_data_cache:
                    full_context_data_cache[ticker] = point_in_time_data_cache[ticker]

            # --- STEP 5: Risk overlay (VIX) ---
            try:
                latest_vix = vix_data.loc[:day_str].iloc[-1]['close']
                market_is_high_risk = latest_vix > config.HIGH_RISK_VIX_THRESHOLD
                if market_is_high_risk:
                    log.warning(f"!! MASTER RISK OVERLAY ENGAGED !! VIX={latest_vix:.2f}. No new BUYs today.")
            except (KeyError, IndexError):
                market_is_high_risk = False

            # --- STEP 6: Market regime + sector analysis ---
            nifty_data_slice = full_context_data_cache.get(BENCHMARK_INDEX)
            market_regime = data_retriever.calculate_regime_from_data(
                nifty_data_slice.loc[:current_day]
            ) if nifty_data_slice is not None else "Neutral"

            strong_sectors = sector_analyzer.get_top_performing_sectors(full_context_data_cache, current_day)
            vix_slice = vix_data.loc[:current_day] if vix_data is not None else None
            volatility_regime = data_retriever.get_volatility_regime(vix_slice)
            log.info(f"Market State Diagnosis | Trend: {market_regime}, Volatility: {volatility_regime}")

            stocks_for_today = []
            if market_regime == "Bearish":
                log.warning("Market regime is Bearish. Screening for high-probability Mean Reversion setups only.")
                # Allow the agent to look for oversold opportunities even in a downtrend
                stocks_for_today = quantitative_screener.screen_for_mean_reversion( #
                    strong_sectors, approved_stock_data_cache, current_day
                )
                
            elif market_regime == "Bullish":
                if volatility_regime == "High-Risk":
                    # In volatile uptrends, buying dips is safer.
                    stocks_for_today = quantitative_screener.screen_for_pullbacks(
                        strong_sectors, approved_stock_data_cache, current_day
                    )
                else:  # Low or Normal Volatility
                    # In calm, grinding uptrends, momentum breakouts work well.
                    stocks_for_today = quantitative_screener.screen_for_momentum(
                        strong_sectors, approved_stock_data_cache, current_day
                    )

            elif market_regime == "Neutral":
                # In sideways markets, mean reversion plays are safer.
                stocks_for_today = quantitative_screener.screen_for_mean_reversion(
                    strong_sectors, approved_stock_data_cache, current_day
                )

            total_stocks_processed_by_screener += len(fundamentally_approved_stocks)
            total_stocks_passed_screener += len(stocks_for_today)

            if not stocks_for_today:
                log.warning("No stocks passed the funnel today. Skipping.")
                continue

            # --- STEP 7: Collect potential trades ---
            potential_trades_today = []
            for ticker, screener_reason in stocks_for_today:
                # Get the correct point-in-time data slice for this specific stock
                point_in_time_data = approved_stock_data_cache.get(ticker).loc[:day_str].copy()
                if len(point_in_time_data) < 252:
                    log.warning(f"Skipping {ticker} due to insufficient point-in-time historical data on {day_str}.")
                    continue

                # Get historical context for the AI
                strategic_review = _get_30_day_performance_review(current_day, batch_id)
                tactical_lookback = _get_1_day_tactical_lookback(current_day, ticker, batch_id)
                per_stock_history = _get_per_stock_trade_history(ticker, batch_id, current_day)

                # --- Build the context using the new centralized helper ---
                full_context = _build_backtest_context(ticker, point_in_time_data, market_regime, nifty_data_slice)
                sanitized_full_context = sanitize_context(full_context)

                # --- Calculate ATR separately (for trade planning, not AI context) ---
                point_in_time_data.ta.atr(length=14, append=True)
                atr_at_prediction = point_in_time_data['ATRr_14'].iloc[-1]

                # --- AI Analysis with retry + exponential backoff + fallback ---
                final_analysis = None
                max_attempts = len(config.GEMINI_API_KEY_POOL)
                current_attempt = 0
                delay = 5  # initial backoff delay

                log.info(f"--- Analyzing {ticker} with Primary Model ({config.PRO_MODEL}) ---")

                while current_attempt < max_attempts:
                    try:
                        final_analysis = analyzer.get_apex_analysis(
                            ticker, sanitized_full_context, strategic_review, tactical_lookback,
                            per_stock_history, model_name=config.PRO_MODEL, screener_reason=screener_reason
                        )
                        if final_analysis:
                            log.info(f"✅ Got analysis for {ticker} (Primary model).")
                            final_analysis['modelUsed'] = 'pro'
                            break
                    except Exception as e:
                        log.error(f"Primary attempt #{current_attempt + 1} for {ticker} failed. Error: {e}")
                        if any(x in str(e).lower() for x in ["429", "quota", "500"]) or isinstance(e, ValueError):
                            log.warning("Recoverable error. Rotating API key and retrying with exponential backoff...")
                            analyzer = AIAnalyzer(api_key=key_manager.rotate_key())
                            current_attempt += 1
                            log.info(f"Backing off for {delay:.2f} seconds...")
                            time.sleep(delay)
                            delay = min(delay * 2, 60) + random.uniform(0, 1)  # exponential + jitter
                        else:
                            break

                # --- Fallback to Flash model ---
                if not final_analysis:
                    log.warning(f"⚡ Fallback model ({config.FLASH_MODEL}) for {ticker}.")
                    try:
                        final_analysis = analyzer.get_apex_analysis(
                            ticker, sanitized_full_context, strategic_review, tactical_lookback,
                            per_stock_history, model_name=config.FLASH_MODEL, screener_reason=screener_reason
                        )
                        if final_analysis:
                            log.info(f"✅ Got analysis for {ticker} (Fallback).")
                            final_analysis['modelUsed'] = 'flash'
                    except Exception as e:
                        log.error(f"CRITICAL: Fallback model also failed for {ticker}. Error: {e}")
                        final_analysis = None

                log.info("Pacing API calls (35s delay).")
                time.sleep(35)

                if not final_analysis:
                    continue

                # --- Veto logic ---
                original_signal = final_analysis.get('signal')
                scrybe_score = final_analysis.get('scrybeScore', 0)
                final_signal = original_signal
                veto_reason = None

                if original_signal in ['BUY', 'SELL']:
                    total_ai_buy_sell_signals += 1
                    is_conviction_ok = abs(scrybe_score) >= active_strategy['min_conviction_score']
                    is_regime_ok = (original_signal == 'BUY' and market_regime != 'Bearish') or \
                                (original_signal == 'SELL' and market_regime != 'Bullish')

                    entry_price = point_in_time_data['close'].iloc[-1]
                    potential_risk_per_share = active_strategy['stop_loss_atr_multiplier'] * atr_at_prediction
                    potential_reward_per_share = potential_risk_per_share * active_strategy['profit_target_rr_multiple']
                    risk_reward_ratio = potential_reward_per_share / potential_risk_per_share if potential_risk_per_share > 0 else 0
                    is_rr_ok = risk_reward_ratio >= active_strategy['profit_target_rr_multiple']

                    if original_signal == 'BUY' and market_is_high_risk:
                        final_signal, veto_reason = 'HOLD', f"VETOED BUY: High VIX {latest_vix:.2f}"
                    elif not is_regime_ok:
                        final_signal, veto_reason = 'HOLD', f"VETOED {original_signal}: Contradicts regime ({market_regime})"
                    elif not is_conviction_ok:
                        final_signal, veto_reason = 'HOLD', f"VETOED {original_signal}: Conviction {scrybe_score}"
                    elif not is_rr_ok:
                        final_signal, veto_reason = 'HOLD', f"VETOED {original_signal}: Poor RR {risk_reward_ratio:.2f}R"

                    if veto_reason:
                        total_signals_vetoed += 1

                prediction_doc = final_analysis.copy()
                prediction_doc['signal'] = final_signal

                entry_price = point_in_time_data['close'].iloc[-1]
                if final_signal in ['BUY', 'SELL']:
                    num_shares_to_trade = int((current_equity * (portfolio_config['risk_per_trade_pct'] / 100.0)) / potential_risk_per_share) if potential_risk_per_share > 0 else 0
                    position_size_pct = (num_shares_to_trade * entry_price / current_equity) * 100
                    stop_loss_price = (entry_price - potential_risk_per_share) if final_signal == 'BUY' else (entry_price + potential_risk_per_share)
                    target_price = (entry_price + potential_reward_per_share) if final_signal == 'BUY' else (entry_price - potential_reward_per_share)

                    prediction_doc['tradePlan'] = {
                        "entryPrice": round(entry_price, 2),
                        "target": round(target_price, 2),
                        "stopLoss": round(stop_loss_price, 2)
                    }
                    prediction_doc.update({
                        'ticker': ticker,
                        'prediction_date': current_day.to_pydatetime(),
                        'price_at_prediction': entry_price,
                        'status': 'open',
                        'strategy': "ApexSwing_v5_HighConviction",
                        'atr_at_prediction': atr_at_prediction,
                        'position_size_pct': position_size_pct
                    })
                    potential_trades_today.append(prediction_doc)
                else:
                    prediction_doc.update({
                        'ticker': ticker,
                        'prediction_date': current_day.to_pydatetime(),
                        'price_at_prediction': entry_price,
                        'status': 'vetoed' if veto_reason else 'hold',
                        'strategy': "ApexSwing_v5_HighConviction",
                        'atr_at_prediction': atr_at_prediction,
                        'veto_reason': veto_reason
                    })
                    database_manager.save_prediction_for_backtesting(prediction_doc, batch_id)

            # --- STEP 8: Rank & execute trades ---
            if potential_trades_today:
                sorted_trades = sorted(potential_trades_today, key=lambda x: x.get('scrybeScore', 0), reverse=True)
                trades_to_execute = sorted_trades[:PORTFOLIO_CONSTRAINTS['max_concurrent_trades']]
                total_trades_executed += len(trades_to_execute)

                for trade_doc in trades_to_execute:
                    trade_doc['analysis_id'] = str(uuid.uuid4())
                    database_manager.save_prediction_for_backtesting(trade_doc, batch_id)


    log.info("\n--- ✅ APEX Dynamic Simulation Finished! ---")
    log.info("="*50)
    log.info("### FUNNEL ANALYSIS REPORT ###")
    log.info(f"Total Stocks Processed: {total_stocks_processed_by_screener}")
    log.info(f"Total Passed Screener: {total_stocks_passed_screener} "
             f"({(total_stocks_passed_screener/total_stocks_processed_by_screener*100) if total_stocks_processed_by_screener > 0 else 0 :.2f}%)")
    log.info(f"Total AI BUY/SELL Signals: {total_ai_buy_sell_signals}")
    log.info(f"Total Signals Vetoed: {total_signals_vetoed}")
    log.info(f"Total Trades Executed: {total_trades_executed}")
    log.info("="*50)

    database_manager.close_db_connection()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the APEX Dynamic Funnel historical backtest.")
    parser.add_argument('--batch_id', required=True)
    parser.add_argument('--start_date', required=True)
    parser.add_argument('--resume_date', required=False, default=None)
    parser.add_argument('--end_date', required=True)
    parser.add_argument('--fresh_run', type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()
    
    effective_start_date = args.start_date
    if args.resume_date and not args.fresh_run:
        log.warning(f"RESUME DATE PROVIDED. Overriding start date. The simulation will resume from: {args.resume_date}")
        effective_start_date = args.resume_date

    run_simulation(
        batch_id=args.batch_id, 
        start_date=effective_start_date, 
        end_date=args.end_date, 
        is_fresh_run=args.fresh_run
    )