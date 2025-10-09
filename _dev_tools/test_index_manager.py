# test_index_manager.py
import pandas as pd
import index_manager
from logger_config import log

def verify_point_in_time_logic():
    """
    A simple test to verify that the get_point_in_time_nifty50_tickers
    function is correctly fetching historical constituents.
    """
    log.info("--- 🕵️‍♂️ Verifying Historical Index Manager ---")

    # --- Define some test dates from the past ---
    test_dates = [
        pd.to_datetime("2024-04-01"),
        pd.to_datetime("2024-10-15"),
        pd.to_datetime("2025-03-01")
    ]

    all_passed = True
    for date in test_dates:
        log.info(f"\n--- Testing for date: {date.date()} ---")
        try:
            tickers = index_manager.get_point_in_time_nifty50_tickers(date)
            
            if tickers and len(tickers) > 40:
                log.info(f"✅ SUCCESS: Found {len(tickers)} tickers.")
                # Optional: Print a few tickers to see the list
                # log.info(f"Sample: {tickers[:3]}")
            else:
                log.error(f"❌ FAILED: Function returned an empty or incomplete list for {date.date()}.")
                all_passed = False

        except Exception as e:
            log.error(f"❌ FAILED: An error occurred for {date.date()}: {e}", exc_info=True)
            all_passed = False
    
    print("-" * 50)
    if all_passed:
        print("✅✅✅ All tests passed! The index manager is working correctly.")
    else:
        print("❌❌❌ One or more tests failed. Please check the logs.")
    print("-" * 50)


if __name__ == "__main__":
    # This check assumes your nifty50_historical_constituents.csv is in the same directory.
    # Make sure it contains data for the dates you are testing (e.g., 2024, 2025).
    verify_point_in_time_logic()