# _dev_tools/analyze_predictions.py

import argparse
import pandas as pd
from pandas.tseries.offsets import BDay # For business day calculations
import yfinance as yf
from logger_config import log
import sys
import os
import time
from datetime import timedelta

# Ensure the script can find project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    import database_manager
    import config # Needed for DB URI potentially, though init_db uses it
except ImportError as e:
    log.fatal(f"Failed to import project modules. Error: {e}")
    sys.exit(1)

# --- Configuration ---
ANALYSIS_HORIZON_DAYS = 5 # How many trading days ahead to check the price move
YFINANCE_DELAY = 0.5 # Delay between yfinance calls if not using bulk

def get_price_on_or_after(df: pd.DataFrame, target_date: pd.Timestamp):
    """Gets the closing price on the target_date or the next available day."""
    if df is None or df.empty:
        return None
    try:
        # Find the first date in the index that is >= target_date
        actual_date = df.index[df.index >= target_date].min()
        if pd.notna(actual_date):
            return df.loc[actual_date, 'close']
        else:
            # If no date >= target_date, maybe return last known price? Or None?
            log.warning(f"No data found on or after {target_date.date()} in DataFrame.")
            return None # Indicate data missing for that future date
    except Exception as e:
        log.error(f"Error getting price for date {target_date.date()}: {e}")
        return None


def run_prediction_analysis(batch_id: str):
    """
    Analyzes the directional accuracy of BUY/SHORT predictions over a short horizon.
    """
    log.info(f"--- Analyzing Prediction Directional Accuracy for Batch: {batch_id} ---")
    log.info(f"Analysis Horizon: {ANALYSIS_HORIZON_DAYS} trading days.")

    predictions_data = []
    unique_tickers = set()
    min_date = pd.Timestamp.max.tz_localize('UTC')
    max_date = pd.Timestamp.min.tz_localize('UTC')

    try:
        # 1. Connect and Fetch Predictions
        database_manager.init_db(purpose='scheduler')
        if database_manager.db is None or database_manager.predictions_collection is None:
            log.error("Failed to initialize scheduler database connection.")
            return

        log.info(f"Fetching BUY/SHORT predictions for batch_id='{batch_id}'...")
        # Ensure prediction_date is timezone-aware if stored that way
        query = {
            "batch_id": batch_id,
            "signal": {"$in": ["BUY", "SHORT"]}
        }
        predictions_cursor = database_manager.predictions_collection.find(query)
        predictions_data = list(predictions_cursor)
        log.info(f"Found {len(predictions_data)} BUY/SHORT predictions.")

        if not predictions_data:
            log.warning("No relevant predictions found for this batch_id.")
            return

        # Determine date range and unique tickers needed
        for pred in predictions_data:
            unique_tickers.add(pred['ticker'])
            # Ensure prediction_date is a Timestamp and timezone-aware (assume UTC if naive)
            pred_date = pd.to_datetime(pred['prediction_date'])
            if pred_date.tzinfo is None:
                 pred_date = pred_date.tz_localize('UTC')
            pred['prediction_date_ts'] = pred_date # Store Timestamp version

            if pred_date < min_date: min_date = pred_date
            if pred_date > max_date: max_date = pred_date

        if not unique_tickers:
             log.warning("No tickers found in predictions.")
             return

        # 2. Fetch Historical Data (Bulk Preferred)
        log.info(f"Fetching historical data for {len(unique_tickers)} tickers...")
        # Calculate overall date range needed: from min prediction date to max prediction date + horizon + buffer
        fetch_start_date = (min_date - BDay(1)).strftime('%Y-%m-%d') # Start slightly before first prediction
        fetch_end_date = (max_date + BDay(ANALYSIS_HORIZON_DAYS + 5)).strftime('%Y-%m-%d') # End after last required date + buffer
        log.info(f"Required data range: {fetch_start_date} to {fetch_end_date}")

        historical_data_dict = {}
        try:
            # Attempt bulk download
            bulk_data = yf.download(
                tickers=list(unique_tickers),
                start=fetch_start_date,
                end=fetch_end_date,
                progress=False,
                ignore_tz=True # yfinance often works better ignoring timezone on download
            )
            if not bulk_data.empty:
                 # Convert multi-index to dict of DataFrames
                 for ticker in unique_tickers:
                     try:
                        df = bulk_data.xs(ticker, level=1, axis=1).dropna(how='all')
                        df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'}, inplace=True)
                        # Ensure index is DatetimeIndex and UTC for comparisons later
                        df.index = pd.to_datetime(df.index, utc=True)
                        if not df.empty:
                            historical_data_dict[ticker] = df
                     except KeyError:
                          log.warning(f"No data returned for {ticker} in bulk download.")
                     except Exception as ex_proc:
                          log.warning(f"Error processing bulk data for {ticker}: {ex_proc}")
            else:
                 log.warning("Bulk download returned empty data.")

            log.info(f"Successfully fetched/processed data for {len(historical_data_dict)} tickers via bulk download.")

        except Exception as bulk_e:
            log.warning(f"Bulk download failed: {bulk_e}. Falling back to individual fetches (will be slow)...")
            # Fallback to individual fetches (slower, respects rate limit)
            for i, ticker in enumerate(list(unique_tickers)):
                log.info(f"Fetching individually for {ticker} ({i+1}/{len(unique_tickers)})...")
                try:
                    df = yf.download(ticker, start=fetch_start_date, end=fetch_end_date, progress=False, ignore_tz=True)
                    if not df.empty:
                        df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'}, inplace=True)
                        df.index = pd.to_datetime(df.index, utc=True) # Ensure UTC
                        historical_data_dict[ticker] = df
                    time.sleep(YFINANCE_DELAY)
                except Exception as indv_e:
                    log.error(f"Failed to fetch data for {ticker}: {indv_e}")

        # 3. Analyze Each Prediction
        log.info(f"Analyzing {len(predictions_data)} predictions...")
        results = []
        for pred in predictions_data:
            ticker = pred['ticker']
            pred_date_ts = pred['prediction_date_ts']
            signal = pred['signal']
            # Price at prediction might be slightly off if prediction date wasn't trading day
            # Let's use the price on the *next* trading day as the effective entry price
            entry_date_target = pred_date_ts + BDay(1)
            exit_date_target = entry_date_target + BDay(ANALYSIS_HORIZON_DAYS)

            hist_df = historical_data_dict.get(ticker)
            if hist_df is None:
                log.warning(f"Skipping prediction for {ticker} on {pred_date_ts.date()}: No historical data found.")
                continue

            entry_price = get_price_on_or_after(hist_df, entry_date_target)
            exit_price = get_price_on_or_after(hist_df, exit_date_target)

            if entry_price is None or exit_price is None:
                log.warning(f"Skipping prediction for {ticker} on {pred_date_ts.date()}: Missing entry or exit price.")
                continue

            pct_change = ((exit_price - entry_price) / entry_price) * 100 if entry_price != 0 else 0

            is_correct = False
            if signal == 'BUY' and pct_change > 0:
                is_correct = True
            elif signal == 'SHORT' and pct_change < 0:
                is_correct = True

            results.append({
                "ticker": ticker,
                "prediction_date": pred_date_ts.date(),
                "signal": signal,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pct_change": pct_change,
                "is_correct": is_correct
            })

        # 4. Calculate and Print Summary Statistics
        if not results:
            log.warning("No valid results after analysis. Cannot calculate statistics.")
            return

        results_df = pd.DataFrame(results)
        total_signals_analyzed = len(results_df)
        accuracy = results_df['is_correct'].mean() * 100

        avg_gain_correct = results_df[results_df['is_correct']]['pct_change'].abs().mean()
        avg_loss_incorrect = results_df[~results_df['is_correct']]['pct_change'].abs().mean()

        # Calculate average gain for BUYs and SHORTs separately
        avg_gain_buy = results_df[(results_df['signal'] == 'BUY') & results_df['is_correct']]['pct_change'].mean()
        avg_loss_buy = results_df[(results_df['signal'] == 'BUY') & ~results_df['is_correct']]['pct_change'].mean()
        avg_gain_short = results_df[(results_df['signal'] == 'SHORT') & results_df['is_correct']]['pct_change'].mean() # Should be negative
        avg_loss_short = results_df[(results_df['signal'] == 'SHORT') & ~results_df['is_correct']]['pct_change'].mean() # Should be positive

        print("\n" + "="*70)
        print(f" Prediction Directional Analysis Summary (Batch: {batch_id})")
        print(f" Horizon: {ANALYSIS_HORIZON_DAYS} Trading Days")
        print("="*70)
        print(f"{'Total Signals Analyzed:':<30} {total_signals_analyzed}")
        print(f"{'Overall Directional Accuracy:':<30} {accuracy:.2f}%")
        print("-" * 70)
        print(f"{'Avg % Change (Correct Signals):':<30} {avg_gain_correct:.2f}%")
        print(f"{'Avg % Change (Incorrect Signals):':<30} {-avg_loss_incorrect:.2f}%") # Show as negative loss
        print("-" * 70)
        print("Breakdown by Signal Type:")
        print(f"  BUY Signals:")
        print(f"    Avg Correct % Gain: {avg_gain_buy:.2f}%")
        print(f"    Avg Incorrect % Loss: {avg_loss_buy:.2f}%")
        print(f"  SHORT Signals:")
        print(f"    Avg Correct % Gain (Price Drop): {avg_gain_short:.2f}%") # Actual gain % for shorts
        print(f"    Avg Incorrect % Loss (Price Rise): {avg_loss_short:.2f}%") # Actual loss % for shorts
        print("="*70)
        log.info("--- Prediction Analysis Complete ---")

        # Optional: Save results_df to CSV for deeper analysis
        # results_df.to_csv(f"_dev_tools/prediction_analysis_{batch_id}.csv", index=False)
        # log.info(f"Saved detailed analysis results to _dev_tools/prediction_analysis_{batch_id}.csv")


    except Exception as e:
        log.critical(f"An error occurred during prediction analysis: {e}", exc_info=True)
    finally:
        database_manager.close_db_connection()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze directional accuracy of backtest predictions.")
    parser.add_argument("--batch_id", required=True, help="The batch_id of the predictions to analyze.")
    args = parser.parse_args()

    run_prediction_analysis(batch_id=args.batch_id)