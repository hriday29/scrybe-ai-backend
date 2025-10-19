# analysis_pipeline.py
from logger_config import log
import config
import database_manager
from utils import sanitize_context
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
        self.active_strategy = config.APEX_SWING_STRATEGY
        self.api_call_timestamps = deque()

    def _setup(self, mode: str):
        """Connects to DB and initializes API clients."""
        log.info(f"Setting up pipeline for '{mode}' mode.")
        db_purpose = 'analysis' if mode == 'live' else 'scheduler'
        database_manager.init_db(purpose=db_purpose)
        
        self.ai_analyzer = AIAnalyzer()
        log.info("✅ Pipeline setup complete.")

    def _diagnose_market(self, point_in_time, full_data_cache):
        """
        Determines the overall market state and now symmetrically selects the
        strongest or weakest sectors based on the market regime.
        """
        log.info("--- Diagnosing Market State (Symmetrical Analysis) ---")
        
        # 1. Analyze VIX and Market Trend
        vix_data = full_data_cache.get("^INDIAVIX")
        latest_vix = 0.0
        market_is_high_risk = False
        try:
            if vix_data is not None:
                latest_vix = vix_data.loc[:point_in_time].iloc[-1]['close']
                market_is_high_risk = latest_vix > config.HIGH_RISK_VIX_THRESHOLD
        except (TypeError, IndexError, KeyError):
            log.warning("Could not determine VIX risk level from data.")

        # Fetch the correct point-in-time data from the cache
        nifty_df = full_data_cache.get("^NSEI")
        vix_df = full_data_cache.get("^INDIAVIX")
        
        # Pass the data into the analyzer function
        market_regime_context = market_regime_analyzer.get_market_regime_context(nifty_df, vix_df)
        market_regime = market_regime_context.get('market_regime', {}).get('regime_status', 'Sideways')
        volatility_regime = market_regime_context.get('volatility_regime', {}).get('volatility_status', 'Normal')

        # 2. Fetch Macro and Breadth Data
        benchmark_performance = _fetch_benchmark_performance(full_data_cache, point_in_time)
        market_breadth = _calculate_market_breadth(full_data_cache, point_in_time)

        # 3. --- NEW: All-Weather Sector Analysis ---
        log.info(f"Regime is '{market_regime}'. Identifying both strongest and weakest sectors to find all opportunities.")
        strong_sectors = sector_analyzer.get_top_performing_sectors(full_data_cache, point_in_time)
        weak_sectors = sector_analyzer.get_bottom_performing_sectors(full_data_cache, point_in_time)

        # Combine and remove duplicates to create a comprehensive list of actionable sectors
        actionable_sectors = list(set(strong_sectors + weak_sectors))

        log.info(f"Combined universe: {len(strong_sectors)} strong sectors and {len(weak_sectors)} weak sectors considered.")

        # 4. Package into the final 'market_state' object
        market_state = {
            "market_regime": market_regime_context.get('market_regime'), # Pass the full dict
            "volatility_regime": market_regime_context.get('volatility_regime'),
            "market_is_high_risk": market_is_high_risk,
            "latest_vix": latest_vix,
            "actionable_sectors": actionable_sectors, # Use the new generic, actionable key
            "benchmark_performance": benchmark_performance,
            "market_breadth": market_breadth
        }
        
        log.info(f"✅ Market Diagnosis: Trend={market_regime}, ActionableSectors={len(actionable_sectors)}, HighRisk={market_is_high_risk}")
        return market_state

    def _generate_candidates(self, market_state, full_data_cache, point_in_time):
        """
        Calls the upgraded quantitative screener, which now acts as a master
        dispatcher to get a list of high-probability, regime-aligned candidates.
        """
        log.info("--- Generating Candidates via Master Screener Dispatcher ---")
        
        # This now calls our upgraded function and passes the market state.
        # It directly returns the (ticker, reason) tuples we need.
        candidates = quantitative_screener.get_strategy_candidates(
            market_state, full_data_cache, point_in_time
        )
        
        return candidates

    def _run_ai_analysis(self, candidates, full_data_cache, market_state, point_in_time, is_backtest, batch_id=None):
        """
        Runs the "Committee of Experts" AI analysis pipeline using the new
        Volatility/Futures analyst instead of Sentiment. Handles potentially more candidates.
        """
        log.info(f"--- Running 'Committee of Experts' AI Analysis for {len(candidates)} Candidate(s) ---") # Handles larger list

        final_synthesized_results = []
        nifty_data_for_rs = full_data_cache.get("^NSEI") # Fetch Nifty data once for RS calc

        for i, (ticker, screener_reason) in enumerate(candidates):
            log.info(f"--- ({i+1}/{len(candidates)}) Convening Committee for {ticker} (Reason: {screener_reason}) ---")

            # --- Build Full Context ---
            point_in_time_data = full_data_cache.get(ticker)
            if point_in_time_data is None:
                 log.warning(f"Skipping {ticker}: No data found in cache for point_in_time.")
                 continue

            # Ensure data slicing is correct
            point_in_time_data_slice = point_in_time_data.loc[:point_in_time].copy()

            if len(point_in_time_data_slice) < 50: # Check length *after* slicing
                log.warning(f"Skipping {ticker}: Insufficient historical data ({len(point_in_time_data_slice)} days) for analysis.")
                continue

            # Build the context using the sliced data and pass the full cache for futures lookup
            full_context = technical_analyzer.build_analysis_context(
                ticker=ticker,
                historical_data=point_in_time_data_slice, # Use the sliced data
                market_state=market_state,
                is_backtest=is_backtest,
                full_nifty_data=nifty_data_for_rs,
                full_data_cache=full_data_cache # Pass the main cache here
            )

            # --- Call Specialist Analysts ---
            log.info(f"[AI Committee] Getting Technical Verdict for {ticker}...")
            tech_verdict = self.ai_analyzer.get_technical_verdict(full_context.get('technical_indicators', {}), ticker)

            log.info(f"[AI Committee] Getting Fundamental Verdict for {ticker}...")
            fund_verdict = self.ai_analyzer.get_fundamental_verdict(full_context.get('fundamental_data', {}), ticker)

            # --- MODIFIED: Call New Volatility/Futures Analyst ---
            log.info(f"[AI Committee] Getting Volatility/Futures Verdict for {ticker}...")
            vol_fut_verdict = self.ai_analyzer.get_volatility_and_futures_verdict(full_context.get('volatility_futures_data', {}), ticker)
            # --- END MODIFICATION ---

            # --- Sanity Check on Specialist Reports ---
            # --- MODIFIED: Check new verdict variable ---
            if "error" in tech_verdict or "error" in fund_verdict or "error" in vol_fut_verdict:
                log.error(f"Could not proceed with final synthesis for {ticker} due to an error in a specialist analysis. Skipping.")
                # Optionally log the specific error details here
                if "error" in tech_verdict: log.error(f"  Technical Error: {tech_verdict['error']}")
                if "error" in fund_verdict: log.error(f"  Fundamental Error: {fund_verdict['error']}")
                if "error" in vol_fut_verdict: log.error(f"  Vol/Futures Error: {vol_fut_verdict['error']}")
                continue
            # --- END MODIFICATION ---

            # --- Call Head of Strategy for Final Synthesis ---
            try:
                log.info(f"[AI Committee] Passing reports to Head of Strategy for {ticker}...")
                # --- MODIFIED: Pass new verdict variable ---
                final_analysis = self.ai_analyzer.get_apex_analysis(
                    ticker=ticker,
                    technical_verdict=tech_verdict,
                    fundamental_verdict=fund_verdict,
                    volatility_futures_verdict=vol_fut_verdict, # Pass the new verdict
                    market_state=market_state,
                    screener_reason=screener_reason
                )
                # --- END MODIFICATION ---

                if final_analysis and "error" not in final_analysis : # Check for errors from apex call itself
                    final_synthesized_results.append({
                        "ticker": ticker,
                        "ai_analysis": final_analysis,
                        "point_in_time_data": point_in_time_data_slice # Save the data used for analysis
                    })
                elif final_analysis and "error" in final_analysis:
                     log.error(f"Final APEX synthesis for {ticker} failed: {final_analysis.get('error')}")
                else:
                    log.error(f"Final APEX synthesis for {ticker} returned no result.")

            except Exception as e:
                log.critical(f"The Head of Strategy (APEX Synthesis) failed critically for {ticker}: {e}", exc_info=True)
                continue # Continue to the next candidate

        log.info(f"--- AI Committee Concluded. {len(final_synthesized_results)} stocks have a final verdict. ---")
        return final_synthesized_results
    
    def _apply_strategy_overlay(self, ai_results, market_state, is_backtest):
        """
        Filters the AI's unbiased analysis to find high-probability trades
        that align with our master strategy (market regime, risk, etc.).
        This is the Portfolio Manager's final decision gate.
        """
        log.info("--- Filtering AI Analysis with Strategy Overlay ---")
        
        final_trade_signals = []
        
        for result in ai_results:
            ticker = result['ticker']
            ai_analysis = result['ai_analysis']
            point_in_time_data = result['point_in_time_data']
            
            ai_signal = ai_analysis.get('signal')
            scrybe_score = ai_analysis.get('scrybeScore', 0)
            
            result['final_signal'] = 'HOLD'  # Default to HOLD
            veto_reason = None
            
            # --- PROCESS BUY SIGNALS ---
            if ai_signal == 'BUY' and scrybe_score >= self.active_strategy['min_conviction_score']:
                if market_state['market_regime'] == 'Bearish':  # This line should now be the first check
                    veto_reason = "FILTERED (BUY): Does not align with Bearish market"
                
                # Live fundamental check (optional but good practice)
                # elif not is_backtest:
                #     try:
                #         company_info = yf.Ticker(ticker).info
                #         roe = company_info.get('returnOnEquity')
                #         margins = company_info.get('profitMargins')
                #         if (roe is not None and roe < 0.15) or (margins is not None and margins < 0.10):
                #             veto_reason = (
                #                 f"FILTERED (BUY): Weak Live Fundamentals (ROE: {roe:.2%}, "
                #                 f"Margins: {margins:.2%})"
                #             )
                #     except Exception as e:
                #         log.error(f"Error during live fundamental check for {ticker}: {e}")

                if not veto_reason:
                    result['final_signal'] = 'BUY'

            # --- PROCESS SHORT SIGNALS (NEW LOGIC) ---
            elif ai_signal == 'SHORT' and scrybe_score <= -self.active_strategy['min_conviction_score']:
                if not self.active_strategy.get('allow_short_selling', False):
                    veto_reason = "FILTERED (SHORT): Short-selling disabled by strategy config"
                elif market_state['market_regime'] in ['Bullish', 'Uptrend']:
                    veto_reason = "FILTERED (SHORT): Does not align with Bullish market"
                
                if not veto_reason:
                    result['final_signal'] = 'SHORT'
            
            # Deprecated 'SELL' can be treated as a low-priority exit signal if needed, but for now we ignore it.

            if veto_reason:
                result['veto_reason'] = veto_reason

            # --- GENERATE TRADE PLAN FOR ACTIONABLE SIGNALS ---
            if result['final_signal'] in ['BUY', 'SHORT']:
                point_in_time_data.ta.atr(length=14, append=True)
                atr = point_in_time_data['ATRr_14'].iloc[-1]
                
                if atr and atr > 0:
                    entry_price = point_in_time_data['close'].iloc[-1]
                    risk_per_share = self.active_strategy['stop_loss_atr_multiplier'] * atr
                    reward_per_share = risk_per_share * self.active_strategy['profit_target_rr_multiple']
                    
                    if result['final_signal'] == 'BUY':
                        stop_loss = entry_price - risk_per_share
                        target = entry_price + reward_per_share
                    else:  # SHORT
                        stop_loss = entry_price + risk_per_share
                        target = entry_price - reward_per_share

                    result['trade_plan'] = {
                        "entryPrice": round(entry_price, 2),
                        "target": round(target, 2),
                        "stopLoss": round(stop_loss, 2),
                    }
                    log.info(f"✅ {ticker} AI signal '{result['final_signal']}' passed strategy filters. Trade plan generated.")
                else:
                    result['final_signal'] = 'HOLD'
                    result['veto_reason'] = "FILTERED: Invalid ATR for trade plan generation"

            final_trade_signals.append(result)

        actionable_signals = [s for s in final_trade_signals if s.get('final_signal') in ['BUY', 'SHORT']]
        log.info(f"✅ Strategy overlay complete. {len(actionable_signals)} signal(s) are actionable.")
        return final_trade_signals
    
    def _persist_results(self, final_signals, is_backtest, batch_id=None):
        """Saves the final analysis and actionable signals to the database."""
        log.info(f"--- Persisting {len(final_signals)} Final Signal(s) to Database ---")
        
        if is_backtest:
            for result in final_signals:
                final_signal = result['final_signal']
                
                # Determine status based on the final signal and any veto reason
                status = 'open'
                if final_signal not in ['BUY', 'SHORT']:
                    status = 'hold'
                if result.get('veto_reason'):
                    status = 'vetoed'

                prediction_doc = {
                    **result['ai_analysis'],
                    'ticker': result['ticker'],
                    'prediction_date': result['point_in_time_data'].index[-1].to_pydatetime(),
                    'price_at_prediction': result['point_in_time_data']['close'].iloc[-1],
                    'status': status,
                    'strategy': self.active_strategy['name'],
                    'signal': final_signal,  # This will now correctly be 'BUY', 'SHORT', or 'HOLD'
                    'veto_reason': result.get('veto_reason'),
                    'tradePlan': result.get('trade_plan')
                }
                database_manager.save_prediction_for_backtesting(prediction_doc, batch_id)
        else:
            # LIVE PERSISTENCE LOGIC
            for result in final_signals:
                ticker = result['ticker']
                analysis_to_save = result['ai_analysis']
                analysis_to_save['strategy_signal'] = {
                    "type": result.get('screener_reason', 'Daily Review'),
                    "signal": result['final_signal'],
                    "veto_reason": result.get('veto_reason'),
                    "trade_plan": result.get('trade_plan')
                }
                database_manager.save_vst_analysis(ticker, analysis_to_save, {"ticker": ticker})

                # If the signal is actionable, set it as an active trade
                if result['final_signal'] in ['BUY', 'SHORT']:
                    trade_plan = result.get('trade_plan', {})
                    active_trade_object = {
                        'signal': result['final_signal'],
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
