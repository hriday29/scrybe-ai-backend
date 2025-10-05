# update_fundamentals.py
import yfinance as yf
from logger_config import log
import database_manager
import index_manager
import time

def get_curated_fundamentals(ticker_info: dict) -> dict:
    """Extracts a curated list of key fundamental metrics from a yfinance info object."""
    return {
        "marketCap": ticker_info.get("marketCap"),
        "enterpriseValue": ticker_info.get("enterpriseValue"),
        "trailingPE": ticker_info.get("trailingPE"),
        "forwardPE": ticker_info.get("forwardPE"),
        "priceToBook": ticker_info.get("priceToBook"),
        "enterpriseToRevenue": ticker_info.get("enterpriseToRevenue"),
        "enterpriseToEbitda": ticker_info.get("enterpriseToEbitda"),
        "profitMargins": ticker_info.get("profitMargins"),
        "returnOnEquity": ticker_info.get("returnOnEquity"),
        "debtToEquity": ticker_info.get("debtToEquity"),
        "totalRevenue": ticker_info.get("totalRevenue"),
        "revenueGrowth": ticker_info.get("revenueGrowth"),
        "grossProfits": ticker_info.get("grossProfits"),
        "ebitda": ticker_info.get("ebitda"),
        "earningsGrowth": ticker_info.get("earningsGrowth"),
        "heldPercentInsiders": ticker_info.get("heldPercentInsiders"),
        "heldPercentInstitutions": ticker_info.get("heldPercentInstitutions"),
    }

def run_fundamentals_update():
    """Fetches and stores fundamental data for the Nifty 50 universe."""
    log.info("---  फंडामेंटल डेटा अपडेट शुरू हो रहा है (Starting Fundamentals Data Update) ---")
    
    try:
        # We use the 'analysis' DB as that's where the live app will read from.
        database_manager.init_db(purpose='analysis')
        fundamentals_collection = database_manager.db.fundamentals
        
        stock_universe = index_manager.get_nifty50_tickers()
        if not stock_universe:
            log.error("Could not fetch stock universe. Aborting fundamentals update.")
            return

        log.info(f"Updating fundamental data for {len(stock_universe)} tickers...")

        for i, ticker_symbol in enumerate(stock_universe):
            try:
                log.info(f"({i+1}/{len(stock_universe)}) Fetching data for {ticker_symbol}...")
                ticker_obj = yf.Ticker(ticker_symbol)
                info = ticker_obj.info

                if info and info.get('marketCap'): # Basic check for valid data
                    curated_data = get_curated_fundamentals(info)
                    
                    # Using update_one with upsert=True is the standard way to insert-or-update.
                    fundamentals_collection.update_one(
                        {'ticker': ticker_symbol},
                        {'$set': curated_data},
                        upsert=True
                    )
                    log.info(f"✅ Successfully updated fundamentals for {ticker_symbol}.")
                else:
                    log.warning(f"⚠️ No valid info found for {ticker_symbol}. Skipping.")
                
                time.sleep(1) # Be respectful of the API rate limits

            except Exception as e:
                log.error(f"❌ Failed to process {ticker_symbol}: {e}")
        
    except Exception as e:
        log.critical(f"A critical error occurred during the fundamentals update process: {e}", exc_info=True)
    finally:
        database_manager.close_db_connection()
        log.info("--- फंडामेंटल डेटा अपडेट पूरा हुआ (Fundamentals Data Update Complete) ---")

if __name__ == "__main__":
    run_fundamentals_update()