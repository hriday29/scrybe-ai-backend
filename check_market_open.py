# check_market_open.py
import pandas_market_calendars as mcal
from datetime import datetime
import pytz

# Get the NSE calendar
nse = mcal.get_calendar('NSE')

# Get today's date in India's timezone
india_tz = pytz.timezone('Asia/Kolkata')
today = datetime.now(india_tz).date()

# Check if today is a trading day
if nse.valid_days(start_date=today, end_date=today).empty:
    print(f"Market is closed today ({today}). Skipping analysis.")
    exit(1) # Exit with a non-zero code to stop the workflow
else:
    print(f"Market is open today ({today}). Proceeding with analysis.")
    exit(0) # Exit with a zero code to allow the workflow to continue