# send_daily_report.py
from datetime import datetime, timedelta, timezone
import database_manager
from notifier import send_daily_briefing
from logger_config import log

def get_todays_activity():
    """Queries the database for new signals and closed trades from the last 24 hours."""
    log.info("Fetching today's activity for daily briefing...")
    now_utc = datetime.now(timezone.utc)
    start_of_day_utc = now_utc - timedelta(hours=24)

    new_signals = list(database_manager.live_predictions_collection.find({"prediction_date": {"$gte": start_of_day_utc}}))
    log.info(f"Found {len(new_signals)} new signals from today.")

    closed_trades = list(database_manager.live_performance_collection.find({"close_date": {"$gte": start_of_day_utc}}))
    log.info(f"Found {len(closed_trades)} closed trades from today.")
    
    return {"new_signals": new_signals, "closed_trades": closed_trades}

if __name__ == "__main__":
    database_manager.init_db('analysis')
    try:
        activity_data = get_todays_activity()
        if activity_data['new_signals'] or activity_data['closed_trades']:
            send_daily_briefing(
                new_signals=activity_data['new_signals'],
                closed_trades=activity_data['closed_trades']
            )
        else:
            log.info("No new activity today. Skipping daily briefing email.")
    except Exception as e:
        log.error(f"Failed to generate daily report: {e}", exc_info=True)
    finally:
        database_manager.close_db_connection()