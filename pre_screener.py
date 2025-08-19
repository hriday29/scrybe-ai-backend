# pre_screener.py (Updated and Final)
import pandas as pd
import config
import data_retriever
from logger_config import log
import pandas_ta as ta

def run_pre_screening():
    """
    Analyzes all Nifty 50 stocks to find the best candidates for the
    Apex strategy based on their volatility and trend-following characteristics.
    """
    log.info("--- Starting Intelligent Shortlisting Pre-Screener ---")
    tickers = config.NIFTY_50_TICKERS
    
    # --- FIX: Use the end_date from the config file ---
    # This makes the script dynamic and configurable without changing code.
    # If PRE_SCREENER_END_DATE in config.py is None, it fetches the latest data.
    screener_end_date = config.PRE_SCREENER_END_DATE
    if screener_end_date:
        log.warning(f"Using historical end date for pre-screening: {screener_end_date}")
    # --- END FIX ---
    
    results = []

    for ticker in tickers:
        log.info(f"Analyzing {ticker}...")
        
        # --- FIX: Pass the configured end_date to the data retriever ---
        data = data_retriever.get_historical_stock_data(ticker, end_date=screener_end_date)
        # --- END FIX ---
        
        if data is None or len(data) < 252:
            log.warning(f"Skipping {ticker}, not enough historical data.")
            continue

        # Stage 1: Volatility & Character Analysis
        data.ta.atr(length=14, append=True)
        
        # Robustly find the ATR column, regardless of its exact name (e.g., 'ATRr_14')
        atr_col_name = None
        for col in data.columns:
            if col.upper().startswith('ATR'):
                atr_col_name = col
                break
        
        if not atr_col_name:
            log.warning(f"Could not find ATR column for {ticker}. Skipping.")
            continue

        data['atr_pct'] = (data[atr_col_name] / data['close']) * 100
        avg_atr_pct = data['atr_pct'].mean()

        # Stage 2: Trend Profile Analysis
        data.ta.adx(length=14, append=True)
        if 'ADX_14' not in data.columns:
            log.warning(f"Could not calculate ADX for {ticker}. Skipping.")
            continue

        trending_days = data[data['ADX_14'] > 25]
        trend_pct = (len(trending_days) / len(data)) * 100
        
        results.append({
            "Ticker": ticker,
            "Avg_ATR_Pct": f"{avg_atr_pct:.2f}%",
            "Trend_Pct": f"{trend_pct:.2f}%"
        })

    if not results:
        log.error("Failed to analyze any tickers.")
        return

    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.sort_values(by="Trend_Pct", ascending=False)
    
    print("\n" + "="*50)
    log.info("--- Pre-Screening Analysis Complete ---")
    print("="*50)
    print("Stocks ranked by 'Trendiness' (ADX > 25):")
    print(results_df_sorted.to_string(index=False))
    print("="*50)
    log.info("RECOMMENDATION: Select the Top 10-15 stocks from this list that have a reasonable 'Avg_ATR_Pct' (e.g., between 1.5% and 4.0%) for the final Apex Confirmation Run.")

if __name__ == "__main__":
    run_pre_screening()