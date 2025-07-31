# api.py
from flask import Flask, jsonify, Response
from flask_cors import CORS
import json
from bson import ObjectId
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import pandas as pd
import yfinance as yf
import os

# Import all project modules
import database_manager
import data_retriever
import config
from ai_analyzer import AIAnalyzer # Import the new class
from logger_config import log

from flask import request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

stock_list_cache = None
stock_list_cache_timestamp = None
all_analysis_cache = None
all_analysis_cache_timestamp = None

app = Flask(__name__)
# Get the frontend URL from an environment variable
frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000') 
# Initialize CORS with the frontend URL
CORS(app, resources={r"/api/*": {"origins": frontend_url}})

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# In-memory cache for index analysis with a 60-minute TTL
INDEX_CACHE = {}
CACHE_DURATION_MINUTES = 60

# Initialize the AI Analyzer once on startup for efficiency
try:
    ai_analyzer_instance = AIAnalyzer(config.GEMINI_API_KEY)
except ValueError as e:
    log.fatal(f"Could not initialize AI Analyzer: {e}")
    ai_analyzer_instance = None

log.info("Initializing default database connection for the application...")
database_manager.init_db(purpose='analysis')

@app.route('/api/news/analyze-one', methods=['POST'])
@limiter.limit("10 per day") # Apply the specific rate limit for this endpoint
def analyze_single_news_article():
    log.info("API call received for /api/news/analyze-one")
    
    if not ai_analyzer_instance:
        return jsonify({"error": "AI Analyzer is not available."}), 503

    article_data = request.json
    if not article_data or 'title' not in article_data:
        return jsonify({"error": "Invalid article data provided."}), 400

    try:
        analysis_result = ai_analyzer_instance.get_single_news_impact_analysis(article_data)
        if analysis_result:
            return jsonify(analysis_result)
        else:
            return jsonify({"error": "AI failed to generate an analysis for this article."}), 500
    except Exception as e:
        log.error(f"Error during single news analysis: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

# Add an error handler for rate limit events
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify(error=f"Rate limit exceeded: {e.description}"), 429

# Custom JSON serializer for handling MongoDB's ObjectId and datetime
def custom_json_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, ObjectId):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

# --- Stock Analysis Endpoints ---

@app.route('/api/open-trades', methods=['GET'])
def get_open_trades_endpoint():
    log.info("API call received for /api/open-trades")
    try:
        database_manager.init_db(purpose='analysis')
        open_trades = database_manager.get_open_trades()
        return Response(json.dumps(open_trades, default=custom_json_serializer), mimetype='application/json')
    except Exception as e:
        log.error(f"Failed to fetch open trades: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred while fetching open trades."}), 500
    
@app.route('/api/stock-list', methods=['GET'])
def get_stock_list():
    log.info("API call received for /api/stock-list")

    global stock_list_cache, stock_list_cache_timestamp

    # Cache is valid for 1 hour
    if stock_list_cache and stock_list_cache_timestamp and (datetime.now() - stock_list_cache_timestamp).total_seconds() < 3600:
        log.info("Serving stock list from cache.")
        return jsonify(stock_list_cache)

    log.info("Generating new stock list from DB.")
    try:
        database_manager.init_db(purpose='analysis')
        projection = {"ticker": 1, "companyName": 1, "_id": 0}
        stock_list = list(database_manager.analysis_results_collection.find({}, projection))
        if not stock_list:
            return jsonify([])

        stock_list.sort(key=lambda x: x.get('companyName', ''))

        stock_list_cache = stock_list
        stock_list_cache_timestamp = datetime.now()

        return jsonify(stock_list)
    except Exception as e:
        log.error(f"Failed to fetch stock list: {e}", exc_info=True)
        return jsonify({"error": "Could not retrieve stock list."}), 500

@app.route('/api/analyze/<string:ticker>', methods=['GET'])
def get_analysis(ticker):
    log.info(f"API call received for /api/analyze/{ticker}")
    try:
        database_manager.init_db(purpose='analysis')
        analysis_data = database_manager.get_precomputed_analysis(ticker.upper())
        if analysis_data is None:
            return jsonify({"error": "Analysis not found for the requested stock."}), 404
        return Response(json.dumps(analysis_data, default=custom_json_serializer), mimetype='application/json')
    except Exception as e:
        log.error(f"Failed to fetch analysis for {ticker}: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred."}), 500

@app.route('/api/indices', methods=['GET'])
def get_indices_list():
    log.info("API call received for /api/indices")
    try:
        indices = [{"name": name, "ticker": ticker} for name, ticker in config.INDEX_LIST.items()]
        return jsonify(indices)
    except Exception as e:
        log.error(f"Failed to create indices list from config: {e}", exc_info=True)
        return jsonify({"error": "Could not retrieve indices list."}), 500

@app.route('/api/index-analysis/<path:index_ticker>', methods=['GET'])
def get_index_analysis_data(index_ticker):
    log.info(f"API call received for /api/index-analysis/{index_ticker}")
    
    # --- CACHE CHECK LOGIC ---
    now = datetime.now()
    if index_ticker in INDEX_CACHE:
        cached_item = INDEX_CACHE[index_ticker]
        if (now - cached_item['timestamp']) < timedelta(minutes=CACHE_DURATION_MINUTES):
            log.info(f"Serving fresh analysis for {index_ticker} from CACHE.")
            return Response(json.dumps(cached_item['data'], default=custom_json_serializer), mimetype='application/json')
    # -------------------------
    
    if not ai_analyzer_instance:
        return jsonify({"error": "AI Analyzer is not available."}), 503
        
    index_name = next((name for name, ticker in config.INDEX_LIST.items() if ticker == index_ticker), None)
    if not index_name:
        return jsonify({"error": "Invalid index ticker provided."}), 404

    try:
        log.info(f"No valid cache found. Running LIVE analysis for {index_ticker}...")
        analysis_data = ai_analyzer_instance.get_index_analysis(index_name, index_ticker)
        
        if analysis_data is None or "error" in analysis_data:
            return jsonify({"error": analysis_data.get("error", "Failed to generate index analysis.")}), 500

        # --- STORE IN CACHE ---
        INDEX_CACHE[index_ticker] = {'data': analysis_data, 'timestamp': now}
        log.info(f"Stored new analysis for {index_ticker} in cache.")
        # --------------------

        return Response(json.dumps(analysis_data, default=custom_json_serializer), mimetype='application/json')
    except Exception as e:
        log.error(f"Failed to fetch index analysis for {index_name}: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred."}), 500

# --- Utility Endpoints ---

@app.route('/api/market-pulse', methods=['GET'])
def get_market_pulse():
    log.info("API call received for /api/market-pulse")
    try:
        performance_data = data_retriever.get_nifty50_performance()
        if not performance_data:
             raise ValueError("Live market data source returned no data.")
        df = pd.DataFrame.from_dict(performance_data['stock_performance'], orient='index')
        nifty_history = yf.Ticker("^NSEI").history(period="35d")
        p_1d = (nifty_history['Close'].iloc[-1] / nifty_history['Close'].iloc[-2] - 1) * 100 if len(nifty_history) >= 2 else 0
        p_5d = (nifty_history['Close'].iloc[-1] / nifty_history['Close'].iloc[-6] - 1) * 100 if len(nifty_history) >= 6 else 0
        p_1m = (nifty_history['Close'].iloc[-1] / nifty_history['Close'].iloc[-22] - 1) * 100 if len(nifty_history) >= 22 else 0
        pulse_data = { "overall_performance": df['pctChange'].mean(), "sector_performance": df.groupby('sector')['pctChange'].mean().to_dict(), "advancing_stocks": int((df['pctChange'] > 0).sum()), "declining_stocks": int((df['pctChange'] < 0).sum()), "last_updated": datetime.now().isoformat(), "performance_1_day": p_1d, "performance_5_day": p_5d, "performance_1_month": p_1m }
        return jsonify(pulse_data)
    except Exception as e:
        log.error(f"Failed to get live market pulse: {e}")
        return jsonify({"error": "Market Pulse data is currently unavailable."}), 503

@app.route('/api/track-record', methods=['GET'])
def get_track_record():
    log.info("API call received for /api/track-record")
    try:
        database_manager.init_db(purpose='analysis')
        
        # This now correctly fetches the results from your live daily runs
        results = database_manager.get_live_track_record()
        
        return Response(json.dumps(results, default=custom_json_serializer), mimetype='application/json')
    except Exception as e:
        log.error(f"Failed to fetch track record: {e}", exc_info=True)
        return jsonify({"error": "Could not retrieve AI track record."}), 500

@app.route('/api/news/<string:ticker>', methods=['GET'])
def get_news_for_ticker_endpoint(ticker):
    log.info(f"API call received for /api/news/{ticker}")
    try:
        # No need to init_db if it's already done, but safe to keep
        database_manager.init_db(purpose='analysis') 
        
        articles = data_retriever.get_news_articles_for_ticker(ticker.upper())
        
        return Response(json.dumps(articles, default=custom_json_serializer), mimetype='application/json')
    except Exception as e:
        log.error(f"Failed to fetch news for {ticker}: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred while fetching news."}), 500

@app.route('/api/analysis/ask', methods=['POST'])
@limiter.limit("25 per day") # Allow more queries per day than heavy analysis
def ask_conversational_question():
    log.info("API call received for /api/analysis/ask")
    
    if not ai_analyzer_instance:
        return jsonify({"error": "AI Analyzer is not available."}), 503

    request_data = request.json
    question = request_data.get('question')
    analysis_context = request_data.get('context')

    if not question or not analysis_context:
        return jsonify({"error": "A question and analysis context are required."}), 400

    try:
        result = ai_analyzer_instance.get_conversational_answer(question, analysis_context)
        if result:
            return jsonify(result)
        else:
            return jsonify({"error": "The AI failed to generate an answer."}), 500
    except Exception as e:
        log.error(f"Error during conversational Q&A: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500
    
@app.route('/api/trades/log', methods=['POST'])
def log_user_trade():
    log.info("API call received for /api/trades/log")
    
    trade_log_data = request.json
    if not trade_log_data or 'ticker' not in trade_log_data:
        return jsonify({"error": "Invalid trade log data provided."}), 400

    try:
        database_manager.init_db(purpose='analysis')
        insert_id = database_manager.save_user_trade(trade_log_data)
        
        if insert_id:
            return jsonify({"status": "success", "trade_id": str(insert_id)}), 201
        else:
            return jsonify({"error": "Failed to save the trade log."}), 500
    except Exception as e:
        log.error(f"Error logging user trade: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/api/analysis/vote', methods=['POST'])
@limiter.limit("50 per day")
def record_analysis_vote():
    log.info("API call received for /api/analysis/vote")
    
    data = request.json
    analysis_id = data.get('analysis_id')
    vote_type = data.get('vote_type')

    if not analysis_id or not vote_type:
        return jsonify({"error": "analysis_id and vote_type are required."}), 400

    try:
        database_manager.init_db(purpose='analysis')
        success = database_manager.add_analysis_vote(analysis_id, vote_type)
        
        if success:
            return jsonify({"status": "success", "message": f"Vote '{vote_type}' recorded."})
        else:
            return jsonify({"error": "Failed to record vote."}), 500
    except Exception as e:
        log.error(f"Error recording vote: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500
    
@app.route('/api/analysis/votes/<string:analysis_id>', methods=['GET'])
def get_analysis_votes(analysis_id):
    log.info(f"API call received for /api/analysis/votes/{analysis_id}")
    try:
        database_manager.init_db(purpose='analysis')
        feedback_collection = database_manager.db.analysis_feedback
        
        votes = feedback_collection.find_one({"analysis_id": analysis_id})
        
        if votes:
            # Prepare a clean response, ensuring all vote keys exist
            response_data = {
                "agree_votes": votes.get("agree_votes", 0),
                "unsure_votes": votes.get("unsure_votes", 0),
                "disagree_votes": votes.get("disagree_votes", 0)
            }
            return jsonify(response_data)
        else:
            # If no votes yet, return all zeros
            return jsonify({"agree_votes": 0, "unsure_votes": 0, "disagree_votes": 0})
            
    except Exception as e:
        log.error(f"Error getting votes for {analysis_id}: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500
    
@app.route('/api/feedback/submit', methods=['POST'])
@limiter.limit("10 per hour") # Prevent spam
def submit_feedback():
    log.info("API call received for /api/feedback/submit")
    
    data = request.json
    
    # Automatically capture context
    data['context'] = {
        'page_url': request.headers.get('Referer'),
        'user_agent': request.headers.get('User-Agent')
    }

    if not data.get('feedback_text'):
        return jsonify({"error": "Feedback text is required."}), 400

    try:
        database_manager.init_db(purpose='analysis')
        insert_id = database_manager.save_feedback(data)
        
        if insert_id:
            return jsonify({"status": "success", "feedback_id": str(insert_id)}), 201
        else:
            return jsonify({"error": "Failed to save feedback."}), 500
    except Exception as e:
        log.error(f"Error submitting feedback: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500
    
@app.route('/api/faq/submit', methods=['POST'])
@limiter.limit("5 per hour") # Prevent spam
def submit_faq_question():
    log.info("API call received for /api/faq/submit")
    
    data = request.json
    if not data.get('question_text'):
        return jsonify({"error": "Question text is required."}), 400

    try:
        database_manager.init_db(purpose='analysis')
        insert_id = database_manager.save_faq_submission(data)
        
        if insert_id:
            return jsonify({"status": "success", "submission_id": str(insert_id)}), 201
        else:
            return jsonify({"error": "Failed to save submission."}), 500
    except Exception as e:
        log.error(f"Error submitting FAQ question: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/api/all-analysis', methods=['GET'])
def get_all_analysis():
    log.info("API call received for /api/all-analysis")

    global all_analysis_cache, all_analysis_cache_timestamp

    # Cache is valid for 15 minutes (900 seconds)
    if all_analysis_cache and all_analysis_cache_timestamp and (datetime.now() - all_analysis_cache_timestamp).total_seconds() < 900:
        log.info("Serving all analysis data from cache.")
        return Response(json.dumps(all_analysis_cache, default=custom_json_serializer), mimetype='application/json')

    log.info("Generating new all analysis data from DB.")
    try:
        database_manager.init_db(purpose='analysis')
        results = list(database_manager.analysis_results_collection.find({}))

        # Update cache
        all_analysis_cache = results
        all_analysis_cache_timestamp = datetime.now()

        return Response(json.dumps(results, default=custom_json_serializer), mimetype='application/json')
    except Exception as e:
        log.error(f"Failed to fetch all analysis: {e}", exc_info=True)
        return jsonify({"error": "Could not retrieve all analysis data."}), 500
    
if __name__ == '__main__':
    log.info("Initializing default database connection...")
    database_manager.init_db(purpose='analysis')
    log.info("Starting Flask server on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)