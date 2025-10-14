# _dev_tools/full_system_test.py

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import json

# Add the root directory to the Python path to allow for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger_config import log
import config
import data_retriever
from ai_analyzer import AIAnalyzer
from analysis_pipeline import AnalysisPipeline
import database_manager

# --- Configuration ---
TEST_TICKER = "RELIANCE.NS" # A reliable, data-rich stock for testing
TEST_DATE = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')

def print_header(title):
    log.info("=" * 80)
    log.info(f"### {title.upper()} ###")
    log.info("=" * 80)

def print_result(test_name, success, result=None):
    if success:
        log.info(f"[SUCCESS] ✅ {test_name}")
        if result:
            # Pretty-print JSON results for better readability
            log.info(json.dumps(result, indent=2))
    else:
        log.error(f"[FAILURE] ❌ {test_name}")
        if result:
            log.error(f"Error Details: {result}")
    log.info("-" * 80)

def main():
    """Main function to run all system tests."""
    
    # =========================================================================
    # 1. INITIALIZATION & DATA RETRIEVAL TEST
    # =========================================================================
    print_header("Test 1: Initialization & Data Retrieval")
    
    ai_analyzer = None
    historical_data = None
    
    try:
        # Test AI Analyzer Initialization (checks for Azure credentials)
        ai_analyzer = AIAnalyzer()
        print_result("AI Analyzer Initialization", True)
    except Exception as e:
        print_result("AI Analyzer Initialization", False, str(e))
        return # Cannot proceed if this fails

    try:
        # Test Data Retriever
        historical_data = data_retriever.get_historical_stock_data(TEST_TICKER, end_date=TEST_DATE)
        if historical_data is not None and not historical_data.empty:
            print_result(f"Data Retrieval for {TEST_TICKER}", True, f"Fetched {len(historical_data)} rows.")
        else:
            raise ValueError("Data retrieval returned empty or None.")
    except Exception as e:
        print_result(f"Data Retrieval for {TEST_TICKER}", False, str(e))
        return # Cannot proceed without data

    # =========================================================================
    # 2. SPECIALIST ANALYST TESTS (INDIVIDUAL AI CALLS)
    # =========================================================================
    print_header("Test 2: Specialist Analyst AI Functions")

    # Mock data for testing
    mock_technicals = {"RSI_14": 65, "EMA_50": 2800, "Price": 2850}
    mock_fundamentals = {"returnOnEquity": 0.18, "profitMargins": 0.12, "debtToEquity": 80}
    mock_sentiment = {"put_call_ratio_oi": 0.8, "news_headline": "Reliance announces record profits"}
    
    # Test Technical Verdict
    tech_verdict = ai_analyzer.get_technical_verdict(mock_technicals, TEST_TICKER)
    print_result("AI Technical Verdict", "error" not in tech_verdict, tech_verdict)
    
    # Test Fundamental Verdict
    fund_verdict = ai_analyzer.get_fundamental_verdict(mock_fundamentals, TEST_TICKER)
    print_result("AI Fundamental Verdict", "error" not in fund_verdict, fund_verdict)
    
    # Test Sentiment Verdict
    sent_verdict = ai_analyzer.get_sentiment_verdict(mock_sentiment, TEST_TICKER)
    print_result("AI Sentiment Verdict", "error" not in sent_verdict, sent_verdict)

    # =========================================================================
    # 3. APEX SYNTHESIS TEST (COMPLEX AI CALL)
    # =========================================================================
    print_header("Test 3: APEX Synthesis AI Function")

    apex_analysis = None

    if "error" in tech_verdict or "error" in fund_verdict or "error" in sent_verdict:
        print_result("APEX Synthesis", False, "Skipped due to specialist failure.")
    else:
        mock_market_state = {"market_regime": {"regime_status": "Bullish"}}
        apex_analysis = ai_analyzer.get_apex_analysis(
            ticker=TEST_TICKER,
            technical_verdict=tech_verdict,
            fundamental_verdict=fund_verdict,
            sentiment_verdict=sent_verdict,
            market_state=mock_market_state,
            screener_reason="Test Case: Bullish Momentum"
        )
        print_result("APEX Synthesis", "error" not in apex_analysis, apex_analysis)

    # =========================================================================
    # 4. CONVERSATIONAL TEST
    # =========================================================================
    print_header("Test 4: Conversational AI Function")

    mock_context = {"ticker": TEST_TICKER, **(apex_analysis or {})}
    question = "What is the main reason for the BUY signal?"
    
    answer = ai_analyzer.get_conversational_answer(question, mock_context)
    print_result("Conversational Answer", answer is not None, answer)

    # =========================================================================
    # 5. FULL PIPELINE TEST (SINGLE-DAY RUN)
    # =========================================================================
    print_header("Test 5: Full Analysis Pipeline (Single-Day Run)")
    
    try:
        database_manager.init_db(purpose='scheduler')
        pipeline = AnalysisPipeline()
        pipeline._setup(mode='scheduler') # Re-setup inside to ensure fresh objects
        
        # Prepare data cache for the pipeline run
        data_cache_for_pipeline = {
            TEST_TICKER: historical_data,
            # Add other required indices/tickers for a more robust test if needed
            "^NSEI": data_retriever.get_historical_stock_data("^NSEI", end_date=TEST_DATE)
        }
        
        pipeline.run(
            point_in_time=pd.to_datetime(TEST_DATE),
            full_data_cache=data_cache_for_pipeline,
            is_backtest=True,
            batch_id="full_system_test"
        )
        print_result("Full Pipeline Execution", True, "Pipeline run completed without critical errors.")
    except Exception as e:
        print_result("Full Pipeline Execution", False, str(e))
    finally:
        database_manager.close_db_connection()
        log.info("Database connection closed.")


if __name__ == "__main__":
    main()