# debug_indicators.py
import data_retriever
import pandas_ta as ta
from logger_config import log

def run_diagnostic():
    """
    A simple script to diagnose the pandas_ta column name issue.
    """
    log.info("--- Starting Indicator Diagnostic Test ---")
    
    # We'll use a single, reliable ticker for this test
    ticker = "RELIANCE.NS"
    
    log.info(f"Fetching data for {ticker} using the project's data_retriever...")
    data = data_retriever.get_historical_stock_data(ticker, end_date="2024-06-30")
    
    if data is None or data.empty:
        log.error("Failed to fetch data. Cannot run diagnostic.")
        return

    # --- DIAGNOSTIC STEP 1 ---
    # Print the column names EXACTLY as they are after fetching
    print("\n[DIAGNOSTIC] Columns BEFORE pandas_ta calculation:")
    print(list(data.columns))
    print("-" * 50)
    
    # --- DIAGNOSTIC STEP 2 ---
    # Run the ATR calculation
    log.info("Attempting to calculate ATR with pandas_ta...")
    data.ta.atr(length=14, append=True)
    
    # --- DIAGNOSTIC STEP 3 ---
    # Print the column names EXACTLY as they are after the calculation
    print("\n[DIAGNOSTIC] Columns AFTER pandas_ta calculation:")
    print(list(data.columns))
    print("-" * 50)

    # --- DIAGNOSTIC STEP 4 ---
    # Check for the expected column
    atr_col_name = None
    for col in data.columns:
        if "ATR" in col.upper(): # Use a broad search
            atr_col_name = col
            break
            
    if atr_col_name:
        log.info(f"SUCCESS: Found an ATR column named: '{atr_col_name}'")
    else:
        log.error("FAILURE: No column containing 'ATR' was created by the library.")

if __name__ == "__main__":
    run_diagnostic()