# check_market_open.py
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import pytz

# Get the NSE calendar
nse = mcal.get_calendar('NSE')

# Get today's date in India's timezone and determine the previous day
india_tz = pytz.timezone('Asia/Kolkata')
today = datetime.now(india_tz).date()
previous_day = today - timedelta(days=1)

# Check if the PREVIOUS day was a valid trading day
if nse.valid_days(start_date=previous_day, end_date=previous_day).empty:
    print(f"Market was closed on the previous day ({previous_day}). Skipping analysis.")
    # Exit with a non-zero code to indicate failure, stopping the workflow
    exit(1) 
else:
    print(f"Market was open on the previous day ({previous_day}). Proceeding with analysis.")
    # Exit with a zero code to indicate success, allowing the workflow to continue
    exit(0)