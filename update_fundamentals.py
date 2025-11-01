"""
update_fundamentals.py

Purpose
- Populate and refresh fundamentals for the live analysis database using yfinance,
    saving a current snapshot and point-in-time quarterly history for the Smallcap 250 universe.

How it fits
- Supports the AI and UI with richer fundamentals beyond price/technicals, stored under
    the analysis DB for consumption during inference and display.

Main role
- Batch job that iterates the target universe, curates fields, and upserts documents with
    UTC-aware timestamps, handling rate limits and partial data gracefully.
"""
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

        # --- MODIFIED: Fetch the Nifty Smallcap 250 universe ---
        log.info("Fetching Nifty Smallcap 250 ticker list...")
        stock_universe = index_manager.get_nifty_smallcap_250_tickers()  # <-- CHANGED FUNCTION CALL
        # --- END MODIFICATION ---

        if not stock_universe:
            log.error("Could not fetch Smallcap 250 universe. Aborting fundamentals update.")
            return

        log.info(f"Updating comprehensive fundamental data for {len(stock_universe)} Smallcap 250 tickers...")

        for i, ticker_symbol in enumerate(stock_universe):
            try:
                log.info(f"({i+1}/{len(stock_universe)}) Processing {ticker_symbol}...")
                ticker_obj = yf.Ticker(ticker_symbol)
                latest_info = ticker_obj.info

                if not latest_info or not latest_info.get('marketCap'):
                    log.warning(f"No valid summary info found for {ticker_symbol}. Skipping.")
                    continue

                # --- STRATEGY: Save the LATEST full data ---
                latest_doc = get_curated_fundamentals(latest_info)
                latest_doc['ticker'] = ticker_symbol
                latest_doc['asOfDate'] = pd.to_datetime('today', utc=True).to_pydatetime()

                fundamentals_collection.update_one(
                    {'ticker': ticker_symbol, 'asOfDate': latest_doc['asOfDate']},
                    {'$set': latest_doc},
                    upsert=True
                )

                # --- Create PURE historical documents ---
                quarterly_financials = ticker_obj.quarterly_financials
                if quarterly_financials.empty:
                    log.warning(f"No quarterly financial data found for {ticker_symbol}.")
                    time.sleep(1)  # Shorter delay when financials fail
                    continue

                quarters_saved_count = 0
                for date_col in quarterly_financials.columns[:12]:  # Last 3 years
                    as_of_date = pd.to_datetime(date_col, utc=True).to_pydatetime()
                    income_statement = quarterly_financials[date_col]

                    historical_doc = {
                        'ticker': ticker_symbol,
                        'asOfDate': as_of_date,
                        'totalRevenue': income_statement.get('Total Revenue'),
                        'netIncome': income_statement.get('Net Income'),
                        'grossProfit': income_statement.get('Gross Profit'),
                        'ebitda': income_statement.get('EBITDA'),
                    }

                    # Derived field: Profit margins
                    if historical_doc.get('totalRevenue') and historical_doc['totalRevenue'] > 0 and historical_doc.get('netIncome') is not None:
                        historical_doc['profitMargins'] = historical_doc['netIncome'] / historical_doc['totalRevenue']

                    update_result = fundamentals_collection.update_one(
                        {'ticker': ticker_symbol, 'asOfDate': as_of_date},
                        {'$set': {k: v for k, v in historical_doc.items() if pd.notna(v)}},
                        upsert=True
                    )

                    if update_result.upserted_id or update_result.modified_count > 0:
                        quarters_saved_count += 1

                log.info(f"✅ Processed {ticker_symbol}: Saved CURRENT snapshot + {quarters_saved_count} historical quarters.")
                time.sleep(2)  # Respect rate limits

            except Exception as e:
                log.error(f"❌ Failed to process {ticker_symbol}: {e}")
                time.sleep(1)  # Prevent hammering API

    except Exception as e:
        log.critical(f"A critical error occurred during the fundamentals update process: {e}", exc_info=True)
    finally:
        database_manager.close_db_connection()
        log.info("--- Comprehensive Historical Fundamentals Update Complete ---")

if __name__ == "__main__":
    run_fundamentals_update()