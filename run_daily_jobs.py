# run_daily_jobs.py (FINAL - Unified Analysis with Highlights)

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
from datetime import datetime, timezone
from utils import setup_api_key_iterator, sanitize_context
from sector_analyzer import CORE_SECTOR_INDICES, BENCHMARK_INDEX

def run_unified_daily_analysis():
    """
    Orchestrates the daily analysis using the "Unified Analysis with Highlights" plan.
    It performs one intelligent loop to generate a comprehensive analysis for all stocks,
    and adds a special 'strategy_signal' highlight for those meeting the backtested criteria.
    """
    log.info("--- 🚀 Starting Unified Daily Analysis Job ---")
    
    try:
        # --- 1. INITIAL SETUP (No Changes) ---
        database_manager.init_db(purpose='analysis')
        key_iterator = setup_api_key_iterator()
        current_api_key = next(key_iterator)
        ai_analyzer = AIAnalyzer(api_key=current_api_key)
        active_strategy = config.APEX_SWING_STRATEGY

        # --- 2. GATHER UNIVERSE & MARKET DATA (No Changes) ---
        log.info("Fetching Nifty 50 universe for daily analysis...")
        stock_universe = index_manager.get_nifty50_tickers()
        if not stock_universe:
            log.error("Could not get the stock universe. Aborting.")
            return

        required_indices = list(CORE_SECTOR_INDICES.values()) + [BENCHMARK_INDEX]
        tickers_to_load = stock_universe + required_indices
        log.info(f"Pre-caching historical data for {len(tickers_to_load)} assets...")
        full_data_cache = {
            ticker: data_retriever.get_historical_stock_data(ticker, end_date=None)
            for ticker in tickers_to_load
        }
        full_data_cache = {k: v for k, v in full_data_cache.items() if v is not None and not v.empty}
        log.info(f"Successfully cached data for {len(full_data_cache)} assets.")
        
        stock_sector_map = quantitative_screener._get_stock_sector_map(stock_universe)

        # --- 3. DIAGNOSE MARKET STATE (No Changes) ---
        point_in_time = pd.Timestamp.now().floor('D')
        vix_data = data_retriever.get_historical_stock_data("^INDIAVIX")
        latest_vix = 0.0
        try:
            latest_vix = vix_data.iloc[-1]['close']
            market_is_high_risk = latest_vix > config.HIGH_RISK_VIX_THRESHOLD
            if market_is_high_risk:
                log.warning(f"!! MASTER RISK OVERLAY ENGAGED !! VIX={latest_vix:.2f}. No new BUYs will be actioned.")
        except (TypeError, IndexError):
            market_is_high_risk = False

        market_regime_context = market_regime_analyzer.get_market_regime_context()
        market_regime = market_regime_context.get('market_regime', {}).get('regime_status', 'Neutral')
        volatility_regime = market_regime_context.get('volatility_regime', {}).get('volatility_status', 'Normal')
        log.info(f"Market State Diagnosis | Trend: {market_regime}, Volatility: {volatility_regime}")
        
        strong_sectors = sector_analyzer.get_top_performing_sectors(full_data_cache, point_in_time)
        sector_performance_context = data_retriever.get_nifty50_performance()

        # =================================================================================
        # --- 4. SINGLE INTELLIGENT ANALYSIS LOOP (WITH PERFORMANCE REVAMP) ---
        # =================================================================================
        log.info(f"--- Starting Unified Analysis for {len(stock_universe)} stocks ---")

        # --- PHOENIX REVAMP: Create an info cache to avoid redundant API calls ---
        info_cache = {}
        log.info("Pre-caching company info objects for performance...")
        import yfinance as yf
        for ticker in stock_universe:
            try:
                info_cache[ticker] = yf.Ticker(ticker).info
            except Exception:
                info_cache[ticker] = {} # Store empty dict on failure
        log.info("Company info caching complete.")
        # --- END REVAMP ---

        for i, ticker in enumerate(stock_universe):
            log.info(f"--- Analyzing {i+1}/{len(stock_universe)}: {ticker} ---")
            
            try:
                historical_data = full_data_cache.get(ticker)
                stock_sector = stock_sector_map.get(ticker, "Other")

                # --- PHOENIX REVAMP: Use the cached info object ---
                company_info = info_cache.get(ticker, {})
                company_name = company_info.get('longName', ticker)
                # --- END REVAMP ---

                if historical_data is None:
                    log.warning(f"No cached data for {ticker}, skipping analysis.")
                    continue

                # Pass the cached info object to the screener for efficiency
                screener_reason = quantitative_screener.check_strategy_candidate(
                    ticker=ticker, data=historical_data, stock_sector=stock_sector, strong_sectors=strong_sectors, 
                    market_regime=market_regime, volatility_regime=volatility_regime, point_in_time=point_in_time,
                    company_info=company_info # Pass cached info
                )

                analysis_reason = screener_reason if screener_reason else "Comprehensive Daily Analysis"
                log.info(f"AI analysis reason for {ticker}: {analysis_reason}")

                # Pass the cached info to the context builder as well
                full_context = technical_analyzer.build_live_context(
                    ticker, historical_data, market_regime_context, 
                    sector_performance_context, company_info
                )
                full_context['companyName'] = company_name
                sanitized_context = sanitize_context(full_context)
                
                try:
                    atr = sanitized_context.get("technical_analysis", {}).get("ATR_14", 0)
                    if atr and atr > 0:
                        entry_price = historical_data['close'].iloc[-1]
                        risk_per_share = active_strategy['stop_loss_atr_multiplier'] * atr
                        reward_per_share = risk_per_share * active_strategy['profit_target_rr_multiple']

                        # Add these metrics to the technical_analysis object to be saved
                        sanitized_context["technical_analysis"]["potential_stop_loss"] = round(entry_price - risk_per_share, 2)
                        sanitized_context["technical_analysis"]["potential_target"] = round(entry_price + reward_per_share, 2)
                except Exception as e:
                    log.warning(f"Could not calculate theoretical metrics for {ticker}: {e}")

                ai_analysis_result = ai_analyzer.get_apex_analysis(
                    ticker=ticker, full_context=sanitized_context, screener_reason=analysis_reason,
                    strategic_review=None, tactical_lookback=None, per_stock_history=None, model_name=config.PRO_MODEL
                )

                if not ai_analysis_result:
                    log.warning(f"AI analysis returned no result for {ticker}. Skipping.")
                    continue

                strategy_signal_obj = None
                
                if screener_reason and ai_analysis_result.get('signal') == 'BUY':
                    # ... (The trade plan logic you added is PERFECT, no changes needed here)
                    log.info(f"{ticker} is a '{screener_reason}' candidate with a BUY signal. Applying veto checks...")
                    scrybe_score = ai_analysis_result.get('scrybeScore', 0)
                    is_conviction_ok = abs(scrybe_score) >= active_strategy['min_conviction_score']
                    is_regime_ok = market_regime != 'Bearish'
                    atr = sanitized_context.get("technical_analysis", {}).get("ATR_14", 0)
                    is_rr_ok = atr > 0
                    
                    veto_reason = None
                    if market_is_high_risk: veto_reason = f"High VIX ({latest_vix:.2f})"
                    elif not is_regime_ok: veto_reason = "Contradicts Bearish regime"
                    elif not is_conviction_ok: veto_reason = f"Low Conviction ({scrybe_score})"
                    elif not is_rr_ok: veto_reason = "Poor Risk/Reward (ATR is zero)"

                    if veto_reason:
                        log.warning(f"VETO APPLIED on {ticker}: BUY -> HOLD. Reason: {veto_reason}")
                        ai_analysis_result['signal'] = 'HOLD'
                        strategy_signal_obj = {"type": screener_reason, "signal": "HOLD", "veto_reason": veto_reason}
                    else:
                        log.info(f"✅ {ticker} PASSED all checks. Flagging as actionable BUY.")
                        entry_price = historical_data['close'].iloc[-1]
                        risk_per_share = active_strategy['stop_loss_atr_multiplier'] * atr
                        reward_per_share = risk_per_share * active_strategy['profit_target_rr_multiple']

                        strategy_signal_obj = {
                            "type": screener_reason, 
                            "signal": "BUY", 
                            "scrybe_score": scrybe_score,
                            "trade_plan": {
                                "entry_price": round(entry_price, 2),
                                "stop_loss": round(entry_price - risk_per_share, 2),
                                "target_price": round(entry_price + reward_per_share, 2),
                                "risk_reward_ratio": round(active_strategy['profit_target_rr_multiple'], 1),
                                "holding_period_days": active_strategy['holding_period']
                            }
                        }
                
                ai_analysis_result['strategy_signal'] = strategy_signal_obj
                database_manager.save_vst_analysis(ticker, ai_analysis_result, sanitized_context)

                if strategy_signal_obj and strategy_signal_obj.get('signal') == 'BUY':
                    # --- THIS PART IS THE FIX ---
                    log.info(f"✅ Actionable BUY signal for {ticker}. Setting as an active trade.")
                    
                    # 1. Get the trade plan details from the object we already created
                    trade_plan = strategy_signal_obj.get('trade_plan', {})
                    
                    # 2. Create a complete trade object that the frontend page expects
                    active_trade_object = {
                        'signal': 'BUY',
                        'entry_date': datetime.now(timezone.utc),
                        'expiry_date': datetime.now(timezone.utc) + pd.Timedelta(days=active_strategy['holding_period']),
                        'entry_price': trade_plan.get('entry_price'),
                        'target': trade_plan.get('target_price'),
                        'stop_loss': trade_plan.get('stop_loss'),
                        'risk_reward_ratio': trade_plan.get('risk_reward_ratio'),
                        'strategy': active_strategy.get('name', 'VST_Live')
                    }

                    # 3. Call the correct database function to set this as an open position
                    database_manager.set_active_trade(ticker, active_trade_object)
                    
                    # --- This part below is optional but recommended for logging ---
                    log.info(f"Generating live prediction record for {ticker} as a historical log.")
                    prediction_doc = { 
                        "ticker": ticker, 
                        "prediction_date": datetime.now(timezone.utc), 
                        "signal": "BUY", 
                        "strategy_name": active_strategy['name'], 
                        "screener_reason": screener_reason, 
                        "scrybe_score": scrybe_score, 
                        "details": ai_analysis_result 
                    }
                    database_manager.save_live_prediction(prediction_doc)

            except Exception as e:
                # ... (error handling logic is fine, no changes needed)
                if "429" in str(e):
                    log.error("Quota exhausted, rotating API key...")
                    try:
                        current_api_key = next(key_iterator)
                        ai_analyzer = AIAnalyzer(api_key=current_api_key)
                        log.info("Successfully rotated to a new API key.")
                    except StopIteration:
                        log.critical("All API keys are exhausted! Aborting run.")
                        raise 
                else:
                    log.error(f"An unexpected error occurred while analyzing {ticker}: {e}", exc_info=True)
                    continue

    except Exception as e:
        log.critical(f"A critical failure occurred in the main script loop: {e}", exc_info=True)
    
    finally:
        database_manager.close_db_connection()
        log.info("--- ✅ Unified Daily Analysis Job Finished ---")

if __name__ == "__main__":
    run_unified_daily_analysis()