# check_market_open.py
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import pytz

# Get the NSE calendar
nse = mcal.get_calendar('NSE')

# Get today's date in India's timezone
india_tz = pytz.timezone('Asia/Kolkata')
today = datetime.now(india_tz)

# Determine the day to check based on the current day of the week.
# If it's Monday (weekday() == 0), the previous market day we care about is Friday.
# So, we check 3 calendar days ago.
if today.weekday() == 0:
    day_to_check = today.date() - timedelta(days=3)
# For any other day (Tuesday-Friday), we just check the previous calendar day.
else:
    day_to_check = today.date() - timedelta(days=1)

# Check if the determined day (e.g., last Friday or yesterday) was a valid trading day.
if nse.valid_days(start_date=day_to_check, end_date=day_to_check).empty:
    print(f"Market was closed on the last relevant day ({day_to_check}). Skipping analysis.")
    # Exit with a non-zero code to indicate failure, stopping the workflow correctly
    exit(1) 
else:
    print(f"Market was open on the last relevant day ({day_to_check}). Proceeding with analysis.")
    # Exit with a zero code to indicate success, allowing the workflow to continue
    exit(0)