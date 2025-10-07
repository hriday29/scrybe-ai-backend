"""
Enhanced Angel One Connection Test Script
Tests authentication, symbol lookup, and data fetching for multiple asset types
"""

from logger_config import log
import config
from angelone_retriever import (
    initialize_angelone_session,
    get_full_quote,
    get_index_quote,
    get_historical_data
)
from datetime import datetime, timedelta

def test_environment_variables():
    """Verify all required environment variables are present"""
    log.info("--- Verifying required environment variables... ---")
    
    required_vars = {
        "ANGELONE_API_KEY": config.ANGELONE_API_KEY,
        "ANGELONE_CLIENT_ID": config.ANGELONE_CLIENT_ID,
        "ANGELONE_PASSWORD": config.ANGELONE_PASSWORD,
        "ANGELONE_TOTP_SECRET": config.ANGELONE_TOTP_SECRET,
    }
    
    all_present = True
    for var_name, var_value in required_vars.items():
        if var_value:
            log.info(f"✅ {var_name}... FOUND")
        else:
            log.error(f"❌ {var_name}... MISSING!")
            all_present = False
    
    if all_present:
        log.info("✅ All required secrets are present in the environment.")
    else:
        log.error("❌ Some required secrets are missing. Cannot proceed.")
    
    return all_present

def test_session_initialization():
    """Test Angel One session initialization"""
    log.info("--- Step 2: Attempting to initialize Angel One session... ---")
    
    if initialize_angelone_session():
        log.info("✅✅✅ LOGIN SUCCESSFUL!")
        return True
    else:
        log.error("❌❌❌ LOGIN FAILED!")
        return False

def test_index_quotes():
    """Test fetching quotes for major indices"""
    log.info("\n" + "="*80)
    log.info("--- Step 3: Testing Index Quotes ---")
    log.info("="*80)
    
    indices_to_test = [
        ("Nifty 50", "get_full_quote"),
        ("NIFTY", "get_index_quote"),
        ("BANKNIFTY", "get_index_quote"),
        ("FINNIFTY", "get_index_quote"),
    ]
    
    results = {}
    
    for index_name, method in indices_to_test:
        try:
            log.info(f"\n📊 Testing {index_name} using {method}()...")
            
            if method == "get_full_quote":
                quote = get_full_quote(index_name)
            else:
                quote = get_index_quote(index_name)
            
            if quote:
                ltp = quote.get('ltp', 'N/A')
                log.info(f"✅ SUCCESS! {index_name} LTP: {ltp}")
                results[index_name] = {"status": "SUCCESS", "ltp": ltp, "data": quote}
            else:
                log.error(f"❌ FAILED to fetch {index_name}")
                results[index_name] = {"status": "FAILED", "ltp": None, "data": None}
                
        except Exception as e:
            log.error(f"❌ ERROR fetching {index_name}: {e}")
            results[index_name] = {"status": "ERROR", "ltp": None, "error": str(e)}
    
    # Summary
    log.info("\n" + "="*80)
    log.info("INDEX QUOTES SUMMARY:")
    log.info("="*80)
    successful = sum(1 for r in results.values() if r["status"] == "SUCCESS")
    log.info(f"✅ Successful: {successful}/{len(indices_to_test)}")
    
    for index_name, result in results.items():
        if result["status"] == "SUCCESS":
            log.info(f"  ✅ {index_name}: {result['ltp']}")
        else:
            log.info(f"  ❌ {index_name}: {result['status']}")
    
    return results

def test_stock_quotes():
    """Test fetching quotes for regular stocks"""
    log.info("\n" + "="*80)
    log.info("--- Step 4: Testing Stock Quotes ---")
    log.info("="*80)
    
    stocks_to_test = [
        "RELIANCE",
        "TCS",
        "INFY",
        "HDFCBANK",
        "ICICIBANK",
    ]
    
    results = {}
    
    for stock in stocks_to_test:
        try:
            log.info(f"\n📈 Testing {stock}...")
            quote = get_full_quote(stock)
            
            if quote:
                ltp = quote.get('ltp', 'N/A')
                log.info(f"✅ SUCCESS! {stock} LTP: {ltp}")
                results[stock] = {"status": "SUCCESS", "ltp": ltp, "data": quote}
            else:
                log.error(f"❌ FAILED to fetch {stock}")
                results[stock] = {"status": "FAILED", "ltp": None, "data": None}
                
        except Exception as e:
            log.error(f"❌ ERROR fetching {stock}: {e}")
            results[stock] = {"status": "ERROR", "ltp": None, "error": str(e)}
    
    # Summary
    log.info("\n" + "="*80)
    log.info("STOCK QUOTES SUMMARY:")
    log.info("="*80)
    successful = sum(1 for r in results.values() if r["status"] == "SUCCESS")
    log.info(f"✅ Successful: {successful}/{len(stocks_to_test)}")
    
    for stock, result in results.items():
        if result["status"] == "SUCCESS":
            log.info(f"  ✅ {stock}: {result['ltp']}")
        else:
            log.info(f"  ❌ {stock}: {result['status']}")
    
    return results

def test_historical_data():
    """Test fetching historical data"""
    log.info("\n" + "="*80)
    log.info("--- Step 5: Testing Historical Data ---")
    log.info("="*80)
    
    # Test with Nifty 50 - last 5 days of daily data
    today = datetime.now()
    from_date = (today - timedelta(days=7)).strftime("%Y-%m-%d %H:%M")
    to_date = today.strftime("%Y-%m-%d %H:%M")
    
    log.info(f"\n📅 Fetching historical data for NIFTY")
    log.info(f"   From: {from_date}")
    log.info(f"   To: {to_date}")
    
    try:
        df = get_historical_data("NIFTY", from_date, to_date, interval="ONE_DAY")
        
        if df is not None and not df.empty:
            log.info(f"✅ SUCCESS! Retrieved {len(df)} candles")
            log.info(f"\nSample data (last 3 days):")
            log.info(df.tail(3).to_string())
            return {"status": "SUCCESS", "rows": len(df)}
        else:
            log.error("❌ FAILED to fetch historical data (empty result)")
            return {"status": "FAILED", "rows": 0}
            
    except Exception as e:
        log.error(f"❌ ERROR fetching historical data: {e}")
        return {"status": "ERROR", "error": str(e)}

def print_final_summary(env_check, login, indices, stocks, historical):
    """Print final test summary"""
    log.info("\n" + "="*80)
    log.info("="*80)
    log.info("           FINAL TEST SUMMARY")
    log.info("="*80)
    log.info("="*80)
    
    tests = [
        ("Environment Variables", env_check),
        ("Session Login", login),
        ("Index Quotes", sum(1 for r in indices.values() if r["status"] == "SUCCESS") if isinstance(indices, dict) else 0),
        ("Stock Quotes", sum(1 for r in stocks.values() if r["status"] == "SUCCESS") if isinstance(stocks, dict) else 0),
        ("Historical Data", historical.get("status") == "SUCCESS" if isinstance(historical, dict) else False),
    ]
    
    all_passed = True
    for test_name, result in tests:
        if isinstance(result, bool):
            status = "✅ PASS" if result else "❌ FAIL"
            if not result:
                all_passed = False
        elif isinstance(result, int):
            status = f"✅ {result} passed"
        else:
            status = "❓ UNKNOWN"
        
        log.info(f"{test_name:.<40} {status}")
    
    log.info("="*80)
    
    if all_passed and isinstance(indices, dict) and isinstance(stocks, dict):
        log.info("🎉 ALL TESTS PASSED! Angel One integration is working perfectly.")
    else:
        log.warning("⚠️  Some tests failed. Check logs above for details.")
    
    log.info("="*80)

def main():
    """Main test execution"""
    log.info("="*80)
    log.info("--- Starting Enhanced Angel One Connection Test ---")
    log.info("="*80)
    
    # Step 1: Check environment variables
    env_check = test_environment_variables()
    if not env_check:
        log.error("Cannot proceed without required environment variables.")
        return
    
    # Step 2: Initialize session
    login_success = test_session_initialization()
    if not login_success:
        log.error("Cannot proceed without successful login.")
        return
    
    # Step 3: Test index quotes
    index_results = test_index_quotes()
    
    # Step 4: Test stock quotes
    stock_results = test_stock_quotes()
    
    # Step 5: Test historical data
    historical_results = test_historical_data()
    
    # Final summary
    print_final_summary(env_check, login_success, index_results, stock_results, historical_results)
    
    log.info("\n--- Test Complete ---")

if __name__ == "__main__":
    main()