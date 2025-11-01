# check_market_open.py
"""
check_market_open.py

Purpose
- CI/cron guard that checks whether the last relevant NSE trading day was open; exits with
    code 1 to skip downstream jobs on holidays/weekends, or 0 to proceed.

How it fits
- Used in automated workflows before running the daily analysis pipeline to avoid wasting
    compute on non-trading days.

Main role
- Compute the appropriate prior market day (handles Monday -> Friday) in IST and probe the
    NSE calendar via pandas_market_calendars.
"""
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import pytz
import sys

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
else:
    day_to_check = today.date() - timedelta(days=1)

# Format the date for nicer logs (e.g., "Friday, 13 Sep 2025")
day_to_check_str = day_to_check.strftime("%A, %d %b %Y")

# Always log what date we are checking
print(f"Checking last relevant trading day: {day_to_check_str}")

# Check if the determined day (e.g., last Friday or yesterday) was a valid trading day.
if nse.valid_days(start_date=day_to_check, end_date=day_to_check).empty:
    print(f"[SKIPPED] Market was closed on {day_to_check_str}. No analysis will be run.")
    # Exit with non-zero → workflow step is marked as skipped (not failed, thanks to continue-on-error)
    sys.exit(1)
else:
    print(f"[OK] Market was open on {day_to_check_str}. Proceeding with analysis.")
    # Exit with zero → workflow continues
    sys.exit(0)
