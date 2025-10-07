# analysis_pipeline.py
from logger_config import log
import config
import database_manager
from utils import setup_api_key_iterator, sanitize_context
from ai_analyzer import AIAnalyzer
import market_regime_analyzer
import sector_analyzer
import quantitative_screener
import technical_analyzer
import time             
import random           
from collections import deque
import pandas_ta as ta
import uuid 
from datetime import datetime, timezone
import performance_context
import pandas as pd
import yfinance as yf

BENCHMARK_TICKERS = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Gold": "GC=F",
    "Crude Oil": "CL=F",
    "USD/INR": "INR=X"
}

def _fetch_benchmark_performance(full_data_cache, point_in_time):
    """Calculates recent performance for global benchmarks."""
    performance = {}
    for name, ticker in BENCHMARK_TICKERS.items():
        data = full_data_cache.get(ticker)
        if data is None: continue
        
        data_slice = data.loc[:point_in_time]
        if len(data_slice) > 21:
            try:
                perf_5d = (data_slice['close'].iloc[-1] / data_slice['close'].iloc[-6] - 1) * 100
                perf_21d = (data_slice['close'].iloc[-1] / data_slice['close'].iloc[-22] - 1) * 100
                performance[name] = {"5D_pct_change": round(perf_5d, 2), "21D_pct_change": round(perf_21d, 2)}
            except IndexError:
                continue # Not enough data in the slice
    return performance

def _calculate_market_breadth(full_data_cache, point_in_time):
    """Calculates the percentage of Nifty 50 stocks above key moving averages."""
    nifty50_stocks = [t for t in full_data_cache.keys() if ".NS" in t and t != "^NSEI"]
    if not nifty50_stocks: return {}

    above_50dma = 0
    above_200dma = 0
    
    for ticker in nifty50_stocks:
        data = full_data_cache.get(ticker)
        if data is None: continue

        data_slice = data.loc[:point_in_time].copy()
        if len(data_slice) < 200: continue
        
        data_slice['50_dma'] = data_slice['close'].rolling(window=50).mean()
        data_slice['200_dma'] = data_slice['close'].rolling(window=200).mean()
        
        latest = data_slice.iloc[-1]
        if pd.notna(latest['50_dma']) and latest['close'] > latest['50_dma']:
            above_50dma += 1
        if pd.notna(latest['200_dma']) and latest['close'] > latest['200_dma']:
            above_200dma += 1
    
    total_stocks = len(nifty50_stocks)
    return {
        "pct_above_50dma": round((above_50dma / total_stocks) * 100, 2),
        "pct_above_200dma": round((above_200dma / total_stocks) * 100, 2)
    }

class AnalysisPipeline:
    """
    A class to orchestrate the entire analysis pipeline, from data fetching
    to AI analysis and finally saving the results. This is the new engine
    for both backtesting and live daily jobs.
    """
    def __init__(self):
        """Initializes the pipeline components."""
        log.info("Initializing Analysis Pipeline...")
        self.db = None
        self.ai_analyzer = None
        self.key_iterator = None
        self.active_strategy = config.APEX_SWING_STRATEGY
        self.api_call_timestamps = deque()

    def _setup(self, mode: str):
        """Connects to DB and initializes API clients."""
        log.info(f"Setting up pipeline for '{mode}' mode.")
        db_purpose = 'analysis' if mode == 'live' else 'scheduler'
        database_manager.init_db(purpose=db_purpose)
        
        self.key_iterator = setup_api_key_iterator()
        current_api_key = next(self.key_iterator)
        self.ai_analyzer = AIAnalyzer(api_key=current_api_key)
        log.info("✅ Pipeline setup complete.")

    def _diagnose_market(self, point_in_time, full_data_cache):
        """Determines the overall market state, including deep macro context."""
        log.info("--- Diagnosing Market State (Deep Macro Analysis) ---")
        
        # 1. Analyze VIX and Market Trend/Volatility (Existing Logic)
        vix_data = full_data_cache.get("^INDIAVIX")
        latest_vix = 0.0
        market_is_high_risk = False
        try:
            if vix_data is not None:
                latest_vix = vix_data.loc[:point_in_time].iloc[-1]['close']
                market_is_high_risk = latest_vix > config.HIGH_RISK_VIX_THRESHOLD
        except (TypeError, IndexError, KeyError):
            log.warning("Could not determine VIX risk level from data.")

        market_regime_context = market_regime_analyzer.get_market_regime_context()
        market_regime = market_regime_context.get('market_regime', {}).get('regime_status', 'Neutral')
        volatility_regime = market_regime_context.get('volatility_regime', {}).get('volatility_status', 'Normal')

        # 2. NEW: Fetch deep macro, breadth, and sector data
        benchmark_performance = _fetch_benchmark_performance(full_data_cache, point_in_time)
        market_breadth = _calculate_market_breadth(full_data_cache, point_in_time)
        strong_sectors = sector_analyzer.get_top_performing_sectors(full_data_cache, point_in_time)

        # 3. Package into the upgraded 'market_state' object
        market_state = {
            "market_regime": market_regime,
            "volatility_regime": volatility_regime,
            "market_is_high_risk": market_is_high_risk,
            "latest_vix": latest_vix,
            "strong_sectors": strong_sectors,
            "benchmark_performance": benchmark_performance, # <-- NEW
            "market_breadth": market_breadth # <-- NEW
        }
        
        log.info(f"✅ Market Diagnosis: Trend={market_regime}, Breadth(>50D): {market_breadth.get('pct_above_50dma')}%, HighRisk={market_is_high_risk}")
        return market_state

    def _generate_candidates(self, market_state, full_data_cache, point_in_time):
        """Runs basic filters to find all stocks worth analyzing."""
        log.info("--- Generating Analyzable Universe ---")
        
        # In our new workflow, the "candidates" are all healthy, liquid stocks.
        # The AI will do the actual screening for trade setups.
        candidates = quantitative_screener.get_analyzable_universe(
            full_data_cache, point_in_time
        )
        
        # The output is now a simple list of tickers, not a tuple with a reason.
        return [(ticker, "Unbiased Analysis") for ticker in candidates]

    def _run_ai_analysis(self, candidates, full_data_cache, market_state, point_in_time, is_backtest, batch_id=None):
        """Performs a deep-dive AI analysis on the candidates."""
        log.info(f"--- Running Deep-Dive AI Analysis for {len(candidates)} Candidate(s) ---")
        
        ai_results = []
        
        for ticker, screener_reason in candidates:
            log.info(f"--- Analyzing {ticker} (Reason: {screener_reason}) ---")
            
            # --- 1. Get Point-in-Time Data ---
            point_in_time_data = full_data_cache.get(ticker).loc[:point_in_time].copy()
            if len(point_in_time_data) < 252:
                log.warning(f"Skipping {ticker} due to insufficient point-in-time historical data.")
                continue

            # --- 2. Build Unified Context ---
            full_context = technical_analyzer.build_analysis_context(
                ticker=ticker,
                historical_data=point_in_time_data,
                market_state=market_state, # Pass the whole market_state object
                is_backtest=is_backtest
            )
            sanitized_context = sanitize_context(full_context)
            
            # --- 3. Run AI with Robust Error Handling & Key Rotation ---
            try:
                # Smart rate limiting
                now = time.time()
                while self.api_call_timestamps and self.api_call_timestamps[0] < now - 60:
                    self.api_call_timestamps.popleft()
                
                RPM_LIMIT = 10 
                if len(self.api_call_timestamps) >= RPM_LIMIT:
                    sleep_time = 60 - (now - self.api_call_timestamps[0])
                    log.warning(f"Rate limit approaching. Pausing for {sleep_time:.1f} seconds.")
                    time.sleep(sleep_time)
                
                self.api_call_timestamps.append(time.time())

                # Call the AI analyzer
                if is_backtest:
                    strategic_review = performance_context.get_strategy_performance_summary(batch_id, point_in_time)
                    per_stock_history = performance_context.get_ticker_trade_history(ticker, batch_id, point_in_time)
                else:
                    strategic_review = "Performance review not applicable in live mode."
                    per_stock_history = "Performance review not applicable in live mode."

                # Call the AI analyzer with the new performance context
                analysis = self.ai_analyzer.get_apex_analysis(
                    ticker=ticker, 
                    full_context=sanitized_context, 
                    screener_reason=screener_reason,
                    strategic_review=strategic_review,     
                    tactical_lookback=None,               
                    per_stock_history=per_stock_history,    
                    model_name=config.PRO_MODEL
                )

                if analysis:
                    # Store the result along with other necessary data for the next step
                    ai_results.append({
                        "ticker": ticker,
                        "ai_analysis": analysis,
                        "point_in_time_data": point_in_time_data
                    })
                    log.info(f"✅ AI analysis successful for {ticker}.")

            except Exception as e:
                if "429" in str(e).lower() or "quota" in str(e).lower():
                    log.error("API quota error, attempting to rotate key...")
                    try:
                        new_key = next(self.key_iterator)
                        self.ai_analyzer.update_api_key(new_key)
                        log.info("Successfully rotated to a new API key. The failed stock will be skipped.")
                    except StopIteration:
                        log.critical("All API keys are exhausted! Aborting AI analysis loop.")
                        break # Exit the for loop
                else:
                    log.error(f"An unexpected error occurred during AI analysis for {ticker}: {e}", exc_info=True)
        
        log.info(f"✅ AI analysis complete for {len(ai_results)} candidate(s).")
        return ai_results

    def _apply_strategy_overlay(self, ai_results, market_state, is_backtest):
        """
        Filters the AI's unbiased analysis to find high-probability trades
        that align with our master strategy (market regime, risk, etc.).
        """
        log.info("--- Filtering AI Analysis with Strategy Overlay ---")
        
        final_trade_signals = []
        
        for result in ai_results:
            ticker = result['ticker']
            ai_analysis = result['ai_analysis']
            point_in_time_data = result['point_in_time_data']
            
            ai_signal = ai_analysis.get('signal')
            scrybe_score = ai_analysis.get('scrybeScore', 0)
            
            result['final_signal'] = 'HOLD' # Default to HOLD
            veto_reason = None
            
            if ai_signal == 'BUY' and scrybe_score >= self.active_strategy['min_conviction_score']:
                if market_state['market_is_high_risk']:
                    veto_reason = f"FILTERED: High VIX ({market_state['latest_vix']:.2f})"
                elif market_state['market_regime'] == 'Bearish':
                    veto_reason = "FILTERED: Does not align with Bearish market"
                
                # --- !! CRITICAL FIX: LIVE FUNDAMENTAL VETO !! ---
                elif not is_backtest:
                    try:
                        company_info = yf.Ticker(ticker).info
                        roe = company_info.get('returnOnEquity')
                        margins = company_info.get('profitMargins')
                        if (roe is not None and roe < 0.15) or (margins is not None and margins < 0.10):
                            veto_reason = f"FILTERED: Weak Live Fundamentals (ROE: {roe:.2%}, Margins: {margins:.2%})"
                    except Exception as e:
                        log.error(f"Error during live fundamental check for {ticker}: {e}")
                # --- END OF FIX ---

                if not veto_reason:
                    result['final_signal'] = 'BUY'

            elif ai_signal == 'SELL' and scrybe_score <= -self.active_strategy['min_conviction_score']:
                # Only veto if the strategy config explicitly disallows short-selling
                if not self.active_strategy.get('allow_short_selling', False):
                    veto_reason = "FILTERED: Short-selling disabled by strategy config"

            if veto_reason:
                result['veto_reason'] = veto_reason

            if result['final_signal'] in ['BUY', 'SELL']:
                point_in_time_data.ta.atr(length=14, append=True)
                atr = point_in_time_data['ATRr_14'].iloc[-1]
                if atr and atr > 0:
                    entry_price = point_in_time_data['close'].iloc[-1]
                    risk_per_share = self.active_strategy['stop_loss_atr_multiplier'] * atr
                    reward_per_share = risk_per_share * self.active_strategy['profit_target_rr_multiple']
                    
                    if result['final_signal'] == 'BUY':
                        stop_loss = entry_price - risk_per_share
                        target = entry_price + reward_per_share
                    else: # SELL
                        stop_loss = entry_price + risk_per_share
                        target = entry_price - reward_per_share

                    result['trade_plan'] = {"entryPrice": round(entry_price, 2), "target": round(target, 2), "stopLoss": round(stop_loss, 2)}
                    log.info(f"✅ {ticker} AI signal '{result['final_signal']}' passed strategy filters. Trade plan generated.")
                else:
                    result['final_signal'] = 'HOLD'
                    result['veto_reason'] = "FILTERED: Invalid ATR for trade plan"

            final_trade_signals.append(result)

        actionable_signals = [s for s in final_trade_signals if s.get('final_signal') in ['BUY', 'SELL']]
        log.info(f"✅ Strategy overlay complete. {len(actionable_signals)} signal(s) are actionable.")
        return final_trade_signals
    
    def _persist_results(self, final_signals, is_backtest, batch_id=None):
        """Saves the final analysis and actionable signals to the database."""
        log.info(f"--- Persisting {len(final_signals)} Final Signal(s) to Database ---")
        
        if is_backtest:
            # --- BACKTEST PERSISTENCE LOGIC ---
            for result in final_signals:
                prediction_doc = {
                    **result['ai_analysis'],
                    'ticker': result['ticker'],
                    'prediction_date': result['point_in_time_data'].index[-1].to_pydatetime(),
                    'price_at_prediction': result['point_in_time_data']['close'].iloc[-1],
                    'status': 'vetoed' if result.get('veto_reason') else ('hold' if result['final_signal'] != 'BUY' else 'open'),
                    'strategy': self.active_strategy['name'],
                    'signal': result['final_signal'],
                    'veto_reason': result.get('veto_reason'),
                    'tradePlan': result.get('trade_plan')
                }
                database_manager.save_prediction_for_backtesting(prediction_doc, batch_id)
        else:
            # --- LIVE PERSISTENCE LOGIC ---
            for result in final_signals:
                ticker = result['ticker']
                # 1. Combine all results into a single document for saving
                analysis_to_save = result['ai_analysis']
                analysis_to_save['strategy_signal'] = {
                    "type": result.get('screener_reason', 'Daily Review'),
                    "signal": result['final_signal'],
                    "veto_reason": result.get('veto_reason'),
                    "trade_plan": result.get('trade_plan')
                }
                # For the live app, we save the full analysis for user review
                database_manager.save_vst_analysis(ticker, analysis_to_save, {"ticker": ticker})

                # 2. If the signal is an actionable BUY, set it as an active trade
                if result['final_signal'] == 'BUY':
                    trade_plan = result.get('trade_plan', {})
                    active_trade_object = {
                        'signal': 'BUY',
                        'entry_date': datetime.now(timezone.utc),
                        'expiry_date': datetime.now(timezone.utc) + pd.Timedelta(days=self.active_strategy['holding_period']),
                        'entry_price': trade_plan.get('entryPrice'),
                        'target': trade_plan.get('target'),
                        'stop_loss': trade_plan.get('stopLoss'),
                        'strategy': self.active_strategy.get('name', 'Live')
                    }
                    database_manager.set_active_trade(ticker, active_trade_object)

        log.info("✅ Results saved successfully.")

    def run(self, point_in_time, full_data_cache, is_backtest=False, batch_id=None):
        """
        Executes the full analysis pipeline for a given day.

        Args:
            point_in_time (pd.Timestamp): The date to run the analysis for.
            full_data_cache (dict): A dictionary of pre-fetched historical data.
            is_backtest (bool): Flag to indicate if this is a backtest run.
            batch_id (str): The batch ID for backtesting.
        """
        try:
            # Step 1: Diagnose the market
            market_state = self._diagnose_market(point_in_time, full_data_cache)

            # Step 2: Generate candidates using the screener
            candidates = self._generate_candidates(market_state, full_data_cache, point_in_time)
            
            if not candidates:
                log.info("No candidates passed the screener. Pipeline finished for today.")
                return

            # Step 3: Run the AI analysis on the candidates
            ai_results = self._run_ai_analysis(candidates, full_data_cache, market_state, point_in_time, is_backtest, batch_id)

            # Step 4: Apply the final strategy filter (veto logic)
            final_signals = self._apply_strategy_overlay(ai_results, market_state, is_backtest)
            
            # Step 5: Save everything to the database
            self._persist_results(final_signals, is_backtest=is_backtest, batch_id=batch_id)

        except Exception as e:
            log.critical(f"A critical error occurred in the analysis pipeline: {e}", exc_info=True)
        
    def close(self):
        """Closes any open connections."""
        database_manager.close_db_connection()
        log.info("--- 🔒 Pipeline resources closed. ---")