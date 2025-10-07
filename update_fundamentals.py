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
        stock_universe = index_manager.get_point_in_time_nifty50_tickers(current_day)
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
                rich_latest_fundamentals = get_curated_fundamentals(latest_info)
                quarterly_financials = ticker_obj.quarterly_financials
                if quarterly_financials.empty:
                    log.warning(f"No quarterly financial data found for {ticker_symbol}. Using latest data only.")
                    rich_latest_fundamentals['ticker'] = ticker_symbol
                    rich_latest_fundamentals['asOfDate'] = pd.to_datetime('today', utc=True).to_pydatetime() # UTC Fix
                    fundamentals_collection.update_one(
                        {'ticker': ticker_symbol, 'asOfDate': rich_latest_fundamentals['asOfDate']},
                        {'$set': rich_latest_fundamentals}, upsert=True)
                    continue
                for date_col in quarterly_financials.columns[:12]:
                    # --- TIMEZONE FIX: Make the date UTC-aware before saving ---
                    as_of_date = pd.to_datetime(date_col, utc=True).to_pydatetime()
                    
                    income_statement = quarterly_financials[date_col]
                    total_revenue_quarter = income_statement.get('Total Revenue')
                    net_income_quarter = income_statement.get('Net Income')
                    profit_margin_quarterly = (net_income_quarter / total_revenue_quarter) if total_revenue_quarter and total_revenue_quarter > 0 else None
                    final_doc = rich_latest_fundamentals.copy()
                    final_doc['ticker'] = ticker_symbol
                    final_doc['asOfDate'] = as_of_date
                    final_doc['totalRevenue'] = total_revenue_quarter
                    final_doc['netIncome'] = net_income_quarter
                    if profit_margin_quarterly is not None:
                        final_doc['profitMargins'] = profit_margin_quarterly
                    fundamentals_collection.update_one(
                        {'ticker': ticker_symbol, 'asOfDate': as_of_date},
                        {'$set': final_doc}, upsert=True)
                log.info(f"✅ Successfully updated {len(quarterly_financials.columns[:12])} quarters of rich fundamentals for {ticker_symbol}.")
                time.sleep(2)
            except Exception as e:
                log.error(f"❌ Failed to process {ticker_symbol}: {e}")
    except Exception as e:
        log.critical(f"A critical error occurred during the fundamentals update process: {e}", exc_info=True)
    finally:
        database_manager.close_db_connection()
        log.info("--- Comprehensive Historical Fundamentals Update Complete ---")

if __name__ == "__main__":
    run_fundamentals_update()