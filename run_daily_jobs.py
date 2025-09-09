# run_daily_jobs.py (FINAL - Full Parity with Orchestrator)

import pandas as pd
import index_manager
import quantitative_screener
import technical_analyzer
import data_retriever
import market_regime_analyzer
import database_manager
import sector_analyzer
from ai_analyzer import AIAnalyzer
from logger_config import log
import config
from datetime import datetime
from utils import setup_api_key_iterator, sanitize_context
from sector_analyzer import CORE_SECTOR_INDICES, BENCHMARK_INDEX

def run_daily_stock_analysis():
    """
    Orchestrates the daily analysis with full logical parity to the advanced
    regime-adaptive backtest orchestrator.
    """
    log.info("--- 🚀 Starting Daily Stock Analysis Job (Full Parity Mode) ---")
    
    database_manager.init_db(purpose='analysis')
    key_iterator = setup_api_key_iterator()
    current_api_key = next(key_iterator)
    ai_analyzer = AIAnalyzer(api_key=current_api_key)
    active_strategy = config.APEX_SWING_STRATEGY

    log.info("Fetching Nifty 50 universe...")
    nifty50_tickers = index_manager.get_nifty50_tickers()
    if not nifty50_tickers:
        log.error("Could not get Nifty 50 tickers. Aborting.")
        return

    required_indices = list(CORE_SECTOR_INDICES.values()) + [BENCHMARK_INDEX]
    tickers_to_load = nifty50_tickers + required_indices
    log.info(f"Pre-caching historical data for {len(tickers_to_load)} assets...")
    
    full_data_cache = {
        ticker: data_retriever.get_historical_stock_data(ticker, end_date=None)
        for ticker in tickers_to_load
    }
    full_data_cache = {k: v for k, v in full_data_cache.items() if v is not None and not v.empty}
    log.info(f"Successfully cached data for {len(full_data_cache)} assets.")
    
    point_in_time = pd.Timestamp.now().floor('D')
    
    # --- LOGIC MIRROR 1: VIX RISK OVERLAY ---
    vix_data = data_retriever.get_historical_stock_data("^INDIAVIX")
    try:
        latest_vix = vix_data.iloc[-1]['close']
        market_is_high_risk = latest_vix > config.HIGH_RISK_VIX_THRESHOLD
        if market_is_high_risk:
            log.warning(f"!! MASTER RISK OVERLAY ENGAGED !! VIX={latest_vix:.2f}. No new BUYs will be actioned today.")
    except (TypeError, IndexError):
        market_is_high_risk = False

    # --- LOGIC MIRROR 2: MARKET STATE DIAGNOSIS ---
    market_regime_context = market_regime_analyzer.get_market_regime_context()
    market_regime = market_regime_context.get('market_regime', {}).get('regime_status', 'Neutral')
    volatility_regime = market_regime_context.get('volatility_regime', {}).get('volatility_status', 'Normal')
    log.info(f"Market State Diagnosis | Trend: {market_regime}, Volatility: {volatility_regime}")
    
    strong_sectors = sector_analyzer.get_top_performing_sectors(full_data_cache, point_in_time)
    if not strong_sectors:
        log.warning("No strong performing sectors identified today. Aborting.")
        database_manager.close_db_connection()
        return

    # --- LOGIC MIRROR 3: REGIME-ADAPTIVE SCREENER SELECTION ---
    stocks_for_today = []
    # (This entire if/elif/else block is identical to main_orchestrator.py)
    if market_regime == "Bearish":
        log.info("Regime is Bearish. Screening for Mean Reversion setups only.")
        stocks_for_today = quantitative_screener.screen_for_mean_reversion(strong_sectors, full_data_cache, point_in_time)
    elif market_regime == "Bullish":
        if volatility_regime == "High-Risk":
            log.info("Regime is Bullish but Volatile. Screening for Pullback setups.")
            stocks_for_today = quantitative_screener.screen_for_pullbacks(strong_sectors, full_data_cache, point_in_time)
        else:
            log.info("Regime is Bullish and Stable. Screening for Momentum setups.")
            stocks_for_today = quantitative_screener.screen_for_momentum(strong_sectors, full_data_cache, point_in_time)
    elif market_regime == "Neutral":
        log.info("Regime is Neutral. Screening for Mean Reversion setups.")
        stocks_for_today = quantitative_screener.screen_for_mean_reversion(strong_sectors, full_data_cache, point_in_time)

    if not stocks_for_today:
        log.warning("No stocks passed the selected screener today. Finishing job.")
        database_manager.close_db_connection()
        return

    all_candidates = {ticker: reason for ticker, reason in stocks_for_today}
    candidate_tickers = list(all_candidates.keys())
    log.info(f"Found {len(candidate_tickers)} unique candidates for AI analysis: {candidate_tickers}")

    sector_performance_context = data_retriever.get_nifty50_performance()

    for i, ticker in enumerate(candidate_tickers):
        log.info(f"--- Analyzing candidate {i+1}/{len(candidate_tickers)}: {ticker} ---")
        screener_reason = all_candidates[ticker]
        
        try:
            historical_data = full_data_cache.get(ticker)
            if historical_data is None: continue

            full_context = technical_analyzer.build_live_context(ticker, historical_data, market_regime_context, sector_performance_context)
            sanitized_context = sanitize_context(full_context)
            
            ai_analysis_result = ai_analyzer.get_apex_analysis(
                ticker=ticker, full_context=sanitized_context, screener_reason=screener_reason,
                strategic_review=None, tactical_lookback=None, per_stock_history=None, model_name=config.PRO_MODEL
            )

            if not ai_analysis_result:
                log.warning(f"AI analysis returned no result for {ticker}. Skipping.")
                continue

            # --- LOGIC MIRROR 4: ADVANCED VETO SYSTEM ---
            original_signal = ai_analysis_result.get('signal')
            scrybe_score = ai_analysis_result.get('scrybeScore', 0)
            final_signal = original_signal
            veto_reason = None
            
            if original_signal == 'BUY':
                is_conviction_ok = abs(scrybe_score) >= active_strategy['min_conviction_score']
                is_regime_ok = market_regime != 'Bearish'
                
                technicals = full_context.get("technical_analysis", {})
                atr = technicals.get("ATR_14", 0)
                entry_price = historical_data['close'].iloc[-1]
                
                if atr > 0:
                    risk_per_share = active_strategy['stop_loss_atr_multiplier'] * atr
                    reward_per_share = risk_per_share * active_strategy['profit_target_rr_multiple']
                    rr_ratio = reward_per_share / risk_per_share
                    is_rr_ok = rr_ratio >= active_strategy['profit_target_rr_multiple']
                else:
                    is_rr_ok = False

                if market_is_high_risk:
                    final_signal, veto_reason = 'HOLD', f"VETOED BUY: High VIX ({latest_vix:.2f})"
                elif not is_regime_ok:
                    final_signal, veto_reason = 'HOLD', f"VETOED BUY: Contradicts Bearish regime"
                elif not is_conviction_ok:
                    final_signal, veto_reason = 'HOLD', f"VETOED BUY: Low Conviction ({scrybe_score})"
                elif not is_rr_ok:
                    final_signal, veto_reason = 'HOLD', f"VETOED BUY: Poor Risk/Reward"

                if veto_reason:
                    log.warning(f"VETO APPLIED on {ticker}: {original_signal} -> {final_signal}. Reason: {veto_reason}")
                    ai_analysis_result['signal'] = final_signal
                    ai_analysis_result['veto_reason'] = veto_reason
            
            log.info(f"Saving full analysis for {ticker} (Final Signal: {final_signal}) to the database.")
            database_manager.save_vst_analysis(ticker, ai_analysis_result)

            if final_signal == "BUY":
                log.info(f"✅ Final signal for {ticker} is BUY. Generating live prediction record.")
                prediction_doc = {
                    "ticker": ticker, "prediction_date": datetime.utcnow(),
                    "signal": final_signal, "strategy_name": active_strategy['name'],
                    "screener_reason": screener_reason, "scrybe_score": scrybe_score,
                    "details": ai_analysis_result
                }
                database_manager.save_live_prediction(prediction_doc)
        
        except Exception as e:
            if "429" in str(e):
                log.error("Quota exhausted, rotating API key...")
                try:
                    current_api_key = next(key_iterator)
                    ai_analyzer = AIAnalyzer(api_key=current_api_key)
                    log.info("Successfully rotated to a new API key.")
                except StopIteration:
                    log.critical("All API keys are exhausted! Aborting run.")
                    break
            else:
                log.error(f"An unexpected error occurred while analyzing {ticker}: {e}", exc_info=True)
                continue

    database_manager.close_db_connection()
    log.info("--- ✅ Daily Stock Analysis Job Finished ---")

if __name__ == "__main__":
    run_daily_stock_analysis()