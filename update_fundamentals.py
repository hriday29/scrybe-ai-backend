# update_fundamentals.py (Final UTC-Aware Version)
import yfinance as yf
from logger_config import log
import database_manager
import index_manager
import time
import pandas as pd

def get_curated_fundamentals(ticker_info: dict) -> dict:
    return {
        "marketCap": ticker_info.get("marketCap"), "enterpriseValue": ticker_info.get("enterpriseValue"),
        "trailingPE": ticker_info.get("trailingPE"), "forwardPE": ticker_info.get("forwardPE"),
        "priceToBook": ticker_info.get("priceToBook"), "enterpriseToRevenue": ticker_info.get("enterpriseToRevenue"),
        "enterpriseToEbitda": ticker_info.get("enterpriseToEbitda"), "profitMargins": ticker_info.get("profitMargins"),
        "returnOnEquity": ticker_info.get("returnOnEquity"), "debtToEquity": ticker_info.get("debtToEquity"),
        "revenueGrowth": ticker_info.get("revenueGrowth"), "grossProfits": ticker_info.get("grossProfits"),
        "ebitda": ticker_info.get("ebitda"), "earningsGrowth": ticker_info.get("earningsGrowth"),
        "heldPercentInsiders": ticker_info.get("heldPercentInsiders"), "heldPercentInstitutions": ticker_info.get("heldPercentInstitutions"),
    }

def run_fundamentals_update():
    log.info("--- Starting Comprehensive Historical Fundamentals Update ---")
    try:
        database_manager.init_db(purpose='analysis')
        fundamentals_collection = database_manager.db.fundamentals

        # --- FIX 1: Fetch the CURRENT stock universe ---
        stock_universe = index_manager.get_nifty50_tickers()
        if not stock_universe:
            log.error("Could not fetch stock universe. Aborting fundamentals update.")
            return

        log.info(f"Updating comprehensive fundamental data for {len(stock_universe)} tickers...")

        for i, ticker_symbol in enumerate(stock_universe):
            try:
                log.info(f"({i+1}/{len(stock_universe)}) Processing {ticker_symbol}...")
                ticker_obj = yf.Ticker(ticker_symbol)
                latest_info = ticker_obj.info
                
                if not latest_info or not latest_info.get('marketCap'):
                    log.warning(f"No valid summary info found for {ticker_symbol}. Skipping.")
                    continue

                # --- STRATEGY: Save the LATEST full data as a distinct document ---
                latest_doc = get_curated_fundamentals(latest_info)
                latest_doc['ticker'] = ticker_symbol
                latest_doc['asOfDate'] = pd.to_datetime('today', utc=True).to_pydatetime()
                fundamentals_collection.update_one(
                    {'ticker': ticker_symbol, 'asOfDate': latest_doc['asOfDate']},
                    {'$set': latest_doc},
                    upsert=True
                )
                log.info(f"Saved CURRENT snapshot for {ticker_symbol}.")

                # --- FIX 2: Create PURE historical documents, free of lookahead bias ---
                quarterly_financials = ticker_obj.quarterly_financials
                if quarterly_financials.empty:
                    log.warning(f"No quarterly financial data found for {ticker_symbol}.")
                    continue

                # Process the last 3 years (12 quarters) of historical data
                for date_col in quarterly_financials.columns[:12]:
                    as_of_date = pd.to_datetime(date_col, utc=True).to_pydatetime()
                    income_statement = quarterly_financials[date_col]
                    
                    # Create a NEW, clean document for each historical quarter
                    historical_doc = {
                        'ticker': ticker_symbol,
                        'asOfDate': as_of_date,
                        'totalRevenue': income_statement.get('Total Revenue'),
                        'netIncome': income_statement.get('Net Income'),
                        'grossProfit': income_statement.get('Gross Profit'),
                        'ebitda': income_statement.get('EBITDA'),
                        # Note: We cannot get historical PE, D/E etc. from this source.
                        # We are prioritizing correctness over completeness to avoid bias.
                    }
                    
                    # Calculate profit margin for this specific quarter
                    if historical_doc['totalRevenue'] and historical_doc['totalRevenue'] > 0:
                        historical_doc['profitMargins'] = historical_doc['netIncome'] / historical_doc['totalRevenue']
                    
                    # Save the pure historical document
                    fundamentals_collection.update_one(
                        {'ticker': ticker_symbol, 'asOfDate': as_of_date},
                        {'$set': {k: v for k, v in historical_doc.items() if v is not None}},
                        upsert=True
                    )
                
                log.info(f"✅ Saved {len(quarterly_financials.columns[:12])} quarters of PURE historical fundamentals for {ticker_symbol}.")
                time.sleep(2) # Rate limit yfinance calls

            except Exception as e:
                log.error(f"❌ Failed to process {ticker_symbol}: {e}")

    except Exception as e:
        log.critical(f"A critical error occurred during the fundamentals update process: {e}", exc_info=True)
    finally:
        database_manager.close_db_connection()
        log.info("--- Comprehensive Historical Fundamentals Update Complete ---")

if __name__ == "__main__":
    run_fundamentals_update()