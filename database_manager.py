# database_manager.py
import pymongo
import certifi
from logger_config import log
import config
from datetime import datetime, timezone, timedelta
import data_retriever
from datetime import datetime, timezone

# --- Global Connection Variables ---
client = None
db = None
# For Historical Backtesting
predictions_collection = None
performance_collection = None
# For Live App Data & Reports
analysis_results_collection = None
backtest_reports_collection = None
# NEW: For Live Performance Tracking
live_predictions_collection = None
live_performance_collection = None

def init_db(purpose: str):
    """Initializes a connection to the correct database based on the purpose."""
    global client, db, predictions_collection, performance_collection, analysis_results_collection, backtest_reports_collection, live_predictions_collection, live_performance_collection

    uri = None
    if purpose == 'scheduler':
        uri = config.SCHEDULER_DB_URI
        log.info("Initializing DB connection for: SCHEDULER / BACKTESTER")
    elif purpose == 'analysis':
        uri = config.ANALYSIS_DB_URI
        log.info("Initializing DB connection for: LIVE APPLICATION / API")
    else:
        raise ValueError(f"Invalid database purpose specified: {purpose}")

    try:
        client = pymongo.MongoClient(uri, tlsCAFile=certifi.where())
        db = client.get_default_database()

        if purpose == 'scheduler':
            predictions_collection = db.predictions
            performance_collection = db.performance
            log.info("SCHEDULER collections are ready.")
        
        elif purpose == 'analysis':
            analysis_results_collection = db.analysis_results
            backtest_reports_collection = db.backtest_reports
            live_predictions_collection = db.live_predictions
            live_performance_collection = db.live_performance
            log.info("ANALYSIS collections are ready.")

    except Exception as e:
        log.error(f"Connection to MongoDB for '{purpose}' failed. Error: {e}")
        client = db = predictions_collection = performance_collection = analysis_results_collection = backtest_reports_collection = live_predictions_collection = live_performance_collection = None

# --- Functions for LIVE Application ---

def save_vst_analysis(ticker: str, analysis_doc: dict):
    """Saves the complete VST analysis document for a ticker."""
    if analysis_results_collection is None:
        log.error("Cannot save analysis, 'analysis' db not initialized.")
        return
    try:
        # The new structure is simpler: the analysis doc becomes the root
        payload = analysis_doc.copy()
        payload.pop('_id', None)
        payload['last_updated'] = datetime.now(timezone.utc)
        
        # We perform an update, but preserve the existing active_trade fields
        analysis_results_collection.update_one(
            {"ticker": ticker},
            {
                "$set": payload,
            },
            upsert=True
        )
        log.info(f"Successfully saved VST analysis state for {ticker}.")
    except Exception as e:
        log.error(f"Failed to save VST analysis for {ticker}. Error: {e}")

def set_active_trade(ticker: str, trade_object: dict | None):
    """Sets or clears the 'active_trade' field for a given ticker."""
    if analysis_results_collection is None:
        log.error(f"Cannot set active trade for {ticker}, 'analysis' db not initialized.")
        return
    try:
        expiry_date = trade_object.get('expiry_date') if trade_object else None
        analysis_results_collection.update_one(
            {"ticker": ticker},
            {"$set": {"active_trade": trade_object, "active_trade_expiry": expiry_date}},
            upsert=True
        )
        if trade_object:
            log.info(f"✅ Successfully set ACTIVE TRADE for {ticker}. Signal: {trade_object['signal']}")
        else:
            log.info(f"Cleared active trade for {ticker}.")
    except Exception as e:
        log.error(f"Failed to set active trade for {ticker}. Error: {e}")

def close_live_trade(ticker: str, trade_object: dict, close_reason: str, close_price: float):
    """Closes a live trade, logs performance, and clears the active_trade state."""
    log.info(f"Closing trade for {ticker} and logging performance.")
    entry_price = trade_object['entry_price']
    return_pct = ((close_price - entry_price) / entry_price) * 100 if trade_object['signal'] == 'BUY' else ((entry_price - close_price) / entry_price) * 100
    
    performance_doc = {
        "ticker": ticker, "strategy": trade_object.get('strategy', 'unknown'),
        "signal": trade_object['signal'], "status": "Closed",
        "open_date": trade_object['entry_date'], "close_date": datetime.now(timezone.utc),
        "closing_reason": close_reason, "return_pct": round(return_pct, 2),
        "entry_price": entry_price, "close_price": close_price
    }
    live_performance_collection.insert_one(performance_doc)

    analysis_results_collection.update_one(
        {"ticker": ticker},
        {"$set": {"active_trade": None, "active_trade_expiry": None}, "$push": {"trade_history": trade_object}}
    )
    log.info(f"✅ Performance logged and active trade cleared for {ticker}.")
    return performance_doc

def save_live_prediction(prediction_doc: dict):
    """Saves a pre-formatted prediction document to the live predictions collection."""
    if live_predictions_collection is None:
        log.error("Cannot save live prediction, 'analysis' db not initialized.")
        return
    try:
        live_predictions_collection.insert_one(prediction_doc)
        log.info(f"Saved new LIVE prediction for {prediction_doc['ticker']}.")
    except Exception as e:
        log.error(f"Failed to save live prediction for {prediction_doc.get('ticker')}. Error: {e}")

def get_live_track_record() -> list:
    """Fetches all closed trades from the live performance collection for the UI."""
    if live_performance_collection is None:
        log.error("Cannot get track record, 'analysis' db not initialized.")
        return []
    try:
        return list(live_performance_collection.find({}).sort("close_date", pymongo.DESCENDING))
    except Exception as e:
        log.error(f"Failed to fetch live track record: {e}")
        return []

def get_precomputed_analysis(ticker: str) -> dict | None:
    """Fetches the complete, pre-computed analysis document for a given stock."""
    if analysis_results_collection is None:
        log.error(f"Cannot get analysis for {ticker}, 'analysis' db not initialized.")
        return None
    try:
        log.info(f"Fetching pre-computed analysis for {ticker} from database.")
        result = analysis_results_collection.find_one({"ticker": ticker})
        return result
    except Exception as e:
        log.error(f"Failed to fetch pre-computed analysis for {ticker}: {e}")
        return None

# --- Functions for HISTORICAL Backtesting ---

def save_prediction_for_backtesting(prediction_doc: dict, batch_id: str):
    """Saves a pre-formatted prediction document to the scheduler database."""
    if predictions_collection is None:
        log.error("Cannot save prediction, 'scheduler' database not initialized.")
        return
    try:
        prediction_doc['batch_id'] = batch_id
        predictions_collection.insert_one(prediction_doc)
        log.info(f"Saved new backtest prediction for {prediction_doc['ticker']} on {prediction_doc['prediction_date'].strftime('%Y-%m-%d')}.")
    except Exception as e:
        log.error(f"Failed to save prediction doc for {prediction_doc.get('ticker')}. Error: {e}")

def get_all_closed_trades(strategy_name: str) -> list:
    """Fetches all closed trades from the performance collection for a specific strategy."""
    if performance_collection is None:
        log.error("Cannot get trades, 'scheduler' database not initialized.")
        return []
    try:
        return list(performance_collection.find({"status": "Closed", "strategy": strategy_name}))
    except Exception as e:
        log.error(f"Failed to fetch closed trades from database: {e}")
        return []

def save_backtest_report(strategy_name: str, report_data: dict):
    """Saves the generated backtest report to the 'backtest_reports' collection."""
    if backtest_reports_collection is None:
        log.error("Cannot save report, 'analysis' database not initialized.")
        return
    try:
        report_data_with_timestamp = report_data.copy()
        report_data_with_timestamp['last_updated'] = datetime.now(timezone.utc)
        backtest_reports_collection.update_one(
            {'strategy_name': strategy_name},
            {'$set': report_data_with_timestamp},
            upsert=True
        )
        log.info("✅ Successfully saved backtest report to the analysis database.")
    except Exception as e:
        log.error(f"Failed to save backtest report to analysis database: {e}")

def clear_scheduler_data():
    """Deletes all documents from the predictions and performance collections."""
    if db is None:
        log.error("Cannot clear, database not initialized correctly.")
        return
    try:
        p_del = db.predictions.delete_many({}).deleted_count
        f_del = db.performance.delete_many({}).deleted_count
        log.warning(f"Cleared scheduler DB: Deleted {p_del} predictions and {f_del} performance records.")
    except Exception as e:
        log.error(f"Failed to clear scheduler database. Error: {e}")

def get_open_trades() -> list:
    """
    Fetches all active trades and enriches them with the latest market price, P&L, days held, and R/R.
    """
    if analysis_results_collection is None:
        log.error("Cannot get open trades, 'analysis' db not initialized.")
        return []

    try:
        open_trades_list = []
        trade_docs = list(analysis_results_collection.find({"active_trade": {"$ne": None}}))

        for doc in trade_docs:
            trade = doc['active_trade']
            ticker = doc['ticker']

            live_data = data_retriever.get_live_financial_data(ticker)
            current_price = live_data['curatedData'].get('currentPrice') if live_data and live_data.get('curatedData') else None

            if current_price is None:
                log.warning(f"Could not fetch current price for open trade {ticker}. Skipping.")
                continue

            # Calculate P&L
            entry_price = trade['entry_price']
            if trade['signal'] == 'BUY':
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
            else: # SELL
                pnl_percent = ((entry_price - current_price) / entry_price) * 100
            
            # --- DEFINITIVE FIX FOR DATETIME ERROR ---
            entry_date = trade['entry_date']
            # Ensure the date from the DB is timezone-aware before comparison
            if entry_date.tzinfo is None:
                entry_date = entry_date.replace(tzinfo=timezone.utc)
            
            days_held = (datetime.now(timezone.utc).date() - entry_date.date()).days
            # --- END FIX ---

            # Assemble the final object for the UI
            open_trades_list.append({
                "ticker": ticker,
                "companyName": doc.get('companyName', ticker),
                "signal": trade['signal'],
                "entry_date": trade['entry_date'],
                "expiry_date": trade['expiry_date'],
                "entry_price": entry_price,
                "current_price": current_price,
                "target": trade['target'],
                "stop_loss": trade['stop_loss'],
                "risk_reward_ratio": trade.get('risk_reward_ratio', 'N/A'),
                "pnl_percent": round(pnl_percent, 2),
                "days_held": days_held,
                "strategy": trade.get('strategy', 'VST')
            })
            
        return open_trades_list
    except Exception as e:
        log.error(f"Failed to fetch and process open trades: {e}", exc_info=True)
        return []
    
def save_user_trade(trade_data: dict, user_id: str): # Accept the user_id
    """Saves a user-submitted trade log to the user_trades collection."""
    if db is None:
        log.error("Cannot save user trade, 'analysis' db not initialized.")
        return None
    try:
        user_trades_collection = db.user_trades
        
        trade_data['logged_at'] = datetime.now(timezone.utc)
        trade_data['user_id'] = user_id # Use the real user_id
        
        result = user_trades_collection.insert_one(trade_data)
        log.info(f"Successfully saved user trade with ID: {result.inserted_id} for user {user_id}")
        return result.inserted_id
    except Exception as e:
        log.error(f"Failed to save user trade. Error: {e}")
        return None

def add_analysis_vote(analysis_id: str, vote_type: str):
    """Finds an analysis by its unique ID and increments its vote counter."""
    if db is None:
        log.error("Cannot add vote, 'analysis' db not initialized.")
        return False
        
    if vote_type not in ['agree', 'unsure', 'disagree']:
        log.warning(f"Invalid vote type received: {vote_type}")
        return False
        
    try:
        feedback_collection = db.analysis_feedback
        update_field = f"{vote_type}_votes"
        
        # --- DEFINITIVE FIX: Two-step update/insert logic ---
        
        # Step 1: Try to increment the vote on an existing document.
        result = feedback_collection.update_one(
            {"analysis_id": analysis_id},
            {"$inc": {update_field: 1}}
        )

        # Step 2: If no document was found and updated, create a new one.
        if result.matched_count == 0:
            log.info(f"No existing feedback doc found for {analysis_id}. Creating new one.")
            feedback_doc = {
                "analysis_id": analysis_id,
                "agree_votes": 1 if vote_type == 'agree' else 0,
                "unsure_votes": 1 if vote_type == 'unsure' else 0,
                "disagree_votes": 1 if vote_type == 'disagree' else 0
            }
            feedback_collection.insert_one(feedback_doc)
        
        # ---------------------------------------------------

        log.info(f"Successfully logged '{vote_type}' vote for analysis_id: {analysis_id}.")
        return True
            
    except Exception as e:
        log.error(f"Failed to add analysis vote. Error: {e}", exc_info=True)
        return False
    
def save_feedback(feedback_data: dict, user_id: str): # <-- 1. Add user_id here
    """Saves user-submitted feedback to the feedback collection."""
    if db is None:
        log.error("Cannot save feedback, 'analysis' db not initialized.")
        return None
    try:
        feedback_collection = db.feedback
        
        feedback_data['submitted_at'] = datetime.now(timezone.utc)
        feedback_data['user_id'] = user_id # <-- 2. Use the real user_id
        
        result = feedback_collection.insert_one(feedback_data)
        log.info(f"Successfully saved user feedback with ID: {result.inserted_id}")
        return result.inserted_id
    except Exception as e:
        log.error(f"Failed to save user feedback. Error: {e}")
        return None
    
def save_faq_submission(submission_data: dict, user_id: str): # <-- 1. Add user_id here
    """Saves a user-submitted question to the faq_submissions collection."""
    if db is None:
        log.error("Cannot save FAQ submission, 'analysis' db not initialized.")
        return None
    try:
        faq_submissions_collection = db.faq_submissions
        
        submission_data['submitted_at'] = datetime.now(timezone.utc)
        submission_data['user_id'] = user_id # <-- 2. Use the real user_id
        submission_data['status'] = "new" 
        
        result = faq_submissions_collection.insert_one(submission_data)
        log.info(f"Successfully saved FAQ submission with ID: {result.inserted_id}")
        return result.inserted_id
    except Exception as e:
        log.error(f"Failed to save FAQ submission. Error: {e}")
        return None

def close_db_connection():
    """Closes the global MongoDB client connection if it's open."""
    global client
    if client:
        client.close()
        client = None # Reset the client
        log.info("--- Database connection successfully closed. ---")