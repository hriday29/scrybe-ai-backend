# index.py
"""
api/index.py

Purpose
- Flask API for Scrybe AI: exposes health probes, cached analysis retrieval, index analysis,
    market pulse, track record, news, conversational Q&A, voting/feedback/FAQ, and user trade logs.

How it fits
- Serves the frontend via CORS with rate limiting and caching, reads from analysis and live
    collections, and delegates AI tasks to AIAnalyzer configured with provider abstraction.

Main role
- Public HTTP layer that validates/authenticates requests (Firebase), orchestrates DB reads/writes,
    and shapes JSON responses for the UI and integrations.
"""
from flask import Flask, jsonify, Response, request
from flask_cors import CORS
import json
from bson import ObjectId
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import os
import time
import database_manager
import data_retriever
import config
from ai_analyzer import AIAnalyzer
from logger_config import log
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
import zoneinfo
from functools import wraps
import firebase_admin
from firebase_admin import credentials, auth
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration


def convert_utc_to_ist_string(utc_dt: datetime) -> str:
    """Converts a timezone-aware UTC datetime object to a user-friendly IST string."""
    if not isinstance(utc_dt, datetime):
        return "Invalid Date"
    
    ist_zone = zoneinfo.ZoneInfo("Asia/Kolkata")
    ist_dt = utc_dt.astimezone(ist_zone)
    # This format is more descriptive for a main title
    return ist_dt.strftime("%I:%M:%S %p %Z on %b %d, %Y") 

app = Flask(__name__)

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    integrations=[
        FlaskIntegration(),
    ],
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=1.0,
)

try:
    # IMPORTANT: This assumes your `firebase-service-account.json` is in the root
    # directory and your script is in a subdirectory like /api.
    # Adjust the path if your structure is different.
    cred_path = os.path.join(os.path.dirname(__file__), '..', 'firebase-service-account.json')
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    log.info("✅ Firebase Admin SDK initialized successfully.")
except Exception as e:
    log.fatal(f"🔥 Firebase Admin SDK initialization failed. Error: {e}")

def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authorization token is missing or invalid."}), 401

        id_token = auth_header.split('Bearer ')[1]
        
        try:
            # Verify the token
            decoded_token = auth.verify_id_token(id_token)
            # Pass the decoded token to the route
            return f(decoded_token, *args, **kwargs)
        except firebase_admin.auth.ExpiredIdTokenError:
            return jsonify({"error": "Token has expired. Please log in again."}), 401
        except firebase_admin.auth.InvalidIdTokenError:
            return jsonify({"error": "Token is invalid."}), 401
        except Exception as e:
            log.error(f"An unexpected error occurred during token verification: {e}")
            return jsonify({"error": "Could not verify authorization."}), 500
            
    return decorated_function

# Explicitly allow your deployed frontend and localhost for development
frontend_urls = [
    os.getenv('FRONTEND_URL', 'http://localhost:3000'),
    'https://scrybe-ai-frontend.onrender.com'
]
# Use list(set(...)) to avoid duplicates if the env var is the same
CORS(app, resources={r"/api/*": {"origins": list(set(frontend_urls))}})
log.info(f"CORS configured for origins: {list(set(frontend_urls))}")

# Allow default API limits to be configured via env (comma-separated)
_default_limits_env = os.getenv("API_RATE_LIMITS_DEFAULT", "").strip()
if _default_limits_env:
    _default_limits = [s.strip() for s in _default_limits_env.split(",") if s.strip()]
else:
    _default_limits = ["200 per day", "50 per hour"]

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=_default_limits
)

# Per-endpoint limit env overrides
LIMIT_ANALYZE_ONE = os.getenv("API_LIMIT_ANALYZE_ONE", "10 per day")
LIMIT_VOTE = os.getenv("API_LIMIT_VOTE", "50 per day")
LIMIT_FEEDBACK = os.getenv("API_LIMIT_FEEDBACK", "10 per hour")
LIMIT_FAQ = os.getenv("API_LIMIT_FAQ", "5 per hour")

# Cache configuration
cache_config = {
    "CACHE_TYPE": "SimpleCache",  # use SimpleCache for development
    "CACHE_DEFAULT_TIMEOUT": 300 # default timeout in seconds
}
app.config.from_mapping(cache_config)
cache = Cache(app)

# Initialize the AI Analyzer once on startup for efficiency
try:
    # The AIAnalyzer now uses the provider factory pattern and reads credentials from config
    ai_analyzer_instance = AIAnalyzer()
    log.info("AI Analyzer initialized successfully with dynamic provider selection.")
except Exception as e:
    log.fatal(f"Could not initialize AI Analyzer: {e}")
    ai_analyzer_instance = None

log.info("Initializing default database connection for the application...")
database_manager.init_db(purpose='analysis')

@app.route('/', methods=['GET'])
def health_check():
    """Deep health check with DB ping, AI readiness (shallow by default), and cache test.

    Query params:
      - deep=true to perform an actual tiny AI call (may take longer and consume tokens)
    """
    checks = {}
    overall_ok = True

    # 1) DB ping
    t0 = time.perf_counter()
    try:
        if database_manager.client is None:
            # Attempt to initialize if not done
            database_manager.init_db(purpose='analysis')
        database_manager.client.admin.command('ping')
        checks['db'] = {"ok": True, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    except Exception as e:
        overall_ok = False
        checks['db'] = {"ok": False, "error": str(e)}

    # 2) Cache test (SimpleCache)
    try:
        cache.set("__health_probe__", "1", timeout=5)
        checks['cache'] = {"ok": cache.get("__health_probe__") == "1"}
        if not checks['cache']['ok']:
            overall_ok = False
    except Exception as e:
        overall_ok = False
        checks['cache'] = {"ok": False, "error": str(e)}

    # 3) AI provider readiness (shallow by default)
    deep = request.args.get('deep', 'false').lower() in ('1', 'true', 'yes') or os.getenv('AI_HEALTHCHECK_DEEP', 'false').lower() in ('1', 'true', 'yes')
    try:
        if ai_analyzer_instance is None:
            raise RuntimeError("AI Analyzer not initialized")
        # Shallow: verify model names present
        ai_ok = bool(ai_analyzer_instance.primary_model and ai_analyzer_instance.secondary_model)
        ai_detail = {"ok": ai_ok, "deep": False}
        if deep and ai_ok:
            # Perform a minimal completion request
            messages = [
                {"role": "system", "content": "You are a health-check probe."},
                {"role": "user", "content": "Reply with the single word: ok"}
            ]
            try:
                txt = ai_analyzer_instance.provider.chat_completions(
                    messages=messages,
                    model=ai_analyzer_instance.secondary_model,
                    temperature=0,
                    timeout=15,
                    max_tokens=4,
                )
                ai_detail = {"ok": isinstance(txt, str) and ("ok" in txt.lower()), "deep": True}
            except Exception as aie:
                ai_detail = {"ok": False, "deep": True, "error": str(aie)}
        checks['ai_provider'] = ai_detail
        if not ai_detail.get('ok', False):
            overall_ok = False
    except Exception as e:
        overall_ok = False
        checks['ai_provider'] = {"ok": False, "error": str(e)}

    status = "ok" if overall_ok else "degraded"
    return jsonify({
        "status": status,
        "checks": checks,
        "timestamp": datetime.now().isoformat()
    }), 200 if overall_ok else 503

@app.route('/ready', methods=['GET'])
def readiness_probe():
    """Shallow readiness probe suitable for load balancers.
    Checks that DB and AI analyzer are initialized and config is present.
    """
    ok = True
    details = {}

    if database_manager.db is None:
        ok = False
        details['db'] = False
    else:
        details['db'] = True

    if ai_analyzer_instance is None or not (config.PRIMARY_MODEL and config.SECONDARY_MODEL):
        ok = False
        details['ai'] = False
    else:
        details['ai'] = True

    return jsonify({"ready": ok, **details}), 200 if ok else 503

@app.route('/api/news/analyze-one', methods=['POST'])
@token_required
@limiter.limit(LIMIT_ANALYZE_ONE) # Apply the specific rate limit for this endpoint
def analyze_single_news_article(current_user):
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
@token_required
def get_open_trades_endpoint(current_user):
    log.info("API call received for /api/open-trades")
    try:
        open_trades = database_manager.get_open_trades()
        return Response(json.dumps(open_trades, default=custom_json_serializer), mimetype='application/json')
    except Exception as e:
        log.error(f"Failed to fetch open trades: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred while fetching open trades."}), 500

@app.route('/api/analysis/<path:ticker>', methods=['GET'])
@cache.cached(timeout=3600) # Cache each ticker's analysis for 1 hour
def get_analysis(ticker):
    log.info(f"API call received for /api/analyze/{ticker} - Running DB query.")
    ticker = ticker.upper()
    try:
        analysis_data = database_manager.get_precomputed_analysis(ticker)
        if analysis_data is None:
            return jsonify({"error": "Analysis not found for the requested stock."}), 404
        
        utc_timestamp = analysis_data.get('timestamp')
        if utc_timestamp:
            analysis_data['display_timestamp'] = convert_utc_to_ist_string(utc_timestamp)
        
        return Response(json.dumps(analysis_data, default=custom_json_serializer), mimetype='application/json')
    except Exception as e:
        log.error(f"Failed to fetch analysis for {ticker}: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/api/indices', methods=['GET'])
def get_indices_list():
    log.info("API call received for /api/indices")
    try:
        indices = [{"name": name, "ticker": ticker} for name, ticker in config.INDEX_LIST.items()]
        return jsonify(indices)
    except Exception as e:
        log.error(f"Failed to create indices list from config: {e}", exc_info=True)
        return jsonify({"error": "Could not retrieve indices list."}), 500
    
@app.route('/api/config/a-list', methods=['GET'])
@cache.cached(timeout=86400) # Cache for a full day
def get_a_list_config():
    log.info("API call received for /api/config/a-list")
    try:
        # Directly return the list from your config file
        a_list = config.LIVE_TRADING_UNIVERSE
        return jsonify({"a_list": a_list})
    except Exception as e:
        log.error(f"Failed to retrieve A-List from config: {e}", exc_info=True)
        return jsonify({"error": "Could not retrieve A-List configuration."}), 500

@app.route('/api/index-analysis/<path:index_ticker>', methods=['GET'])
@cache.cached(timeout=3600) # Cache each index's analysis for 1 hour
def get_index_analysis_data(index_ticker):
    log.info(f"API call for /api/index-analysis/{index_ticker} - Running LIVE AI analysis.")
        
    if not ai_analyzer_instance:
        return jsonify({"error": "AI Analyzer is not available."}), 503
        
    index_name = next((name for name, ticker in config.INDEX_LIST.items() if ticker == index_ticker), None)
    if not index_name:
        return jsonify({"error": "Invalid index ticker provided."}), 404

    try:
        current_macro_context = config.LIVE_MACRO_CONTEXT

        # Build latest technicals for the index (price vs MAs, RSI)
        latest_technicals = None
        try:
            hist = yf.Ticker(index_ticker).history(period="200d", interval="1d")
            if not hist.empty:
                df = hist.copy()
                df['MA_20'] = df['Close'].rolling(window=20).mean()
                df['MA_50'] = df['Close'].rolling(window=50).mean()
                df['MA_200'] = df['Close'].rolling(window=200).mean()
                # Simple RSI implementation
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (loss.replace(0, 1e-9))
                df['RSI_14'] = 100 - (100 / (1 + rs))
                last = df.iloc[-1]
                latest_technicals = {
                    "lastClose": float(last['Close']),
                    "MA_20": float(last['MA_20']) if pd.notna(last['MA_20']) else None,
                    "MA_50": float(last['MA_50']) if pd.notna(last['MA_50']) else None,
                    "MA_200": float(last['MA_200']) if pd.notna(last['MA_200']) else None,
                    "RSI_14": float(last['RSI_14']) if pd.notna(last['RSI_14']) else None,
                    "above_50dma": bool(last['Close'] > last['MA_50']) if pd.notna(last['MA_50']) else None,
                    "above_200dma": bool(last['Close'] > last['MA_200']) if pd.notna(last['MA_200']) else None,
                }
        except Exception as te:
            log.warning(f"Failed to compute latest technicals for {index_ticker}: {te}")

        # Get latest VIX value (India VIX)
        vix_value = None
        try:
            vix_hist = yf.Ticker("^INDIAVIX").history(period="5d", interval="1d")
            if not vix_hist.empty:
                vix_value = str(float(vix_hist['Close'].iloc[-1]))
        except Exception as ve:
            log.warning(f"Failed to fetch India VIX: {ve}")

        # Options chain data for the index (primary via Angel One with yfinance fallback)
        options_data = data_retriever.get_index_option_data(index_ticker)

        analysis_data = ai_analyzer_instance.get_index_analysis(
            index_name=index_name,
            macro_context=current_macro_context,
            latest_technicals=latest_technicals,
            vix_value=vix_value,
            options_data=options_data
        )

        if analysis_data is None or "error" in analysis_data:
            return jsonify({"error": analysis_data.get("error", "Failed to generate index analysis.")}), 500

        return Response(json.dumps(analysis_data, default=custom_json_serializer), mimetype='application/json')
    except Exception as e:
        log.error(f"Failed to fetch index analysis for {index_name}: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

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
        
        articles = data_retriever.get_news_articles_for_ticker(ticker.upper())
        
        return Response(json.dumps(articles, default=custom_json_serializer), mimetype='application/json')
    except Exception as e:
        log.error(f"Failed to fetch news for {ticker}: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred while fetching news."}), 500

@app.route('/api/ask', methods=['POST']) # Standardized Route
@token_required
def ask_conversational_question(current_user):
    log.info("API call: /api/ask")
    if not ai_analyzer_instance:
        return jsonify({"error": "AI Analyzer is not available."}), 503

    data = request.json
    question = data.get('question')
    ticker = data.get('ticker')

    if not question or not ticker:
        return jsonify({"error": "A question and ticker are required."}), 400

    analysis_context = database_manager.get_precomputed_analysis(ticker)
    if not analysis_context:
        return jsonify({"error": "Analysis context not found for the given ticker."}), 404

    # ============================ NEW ROBUST FIX ============================
    # REMOVE the previous fix for just the '_id' field.
    # This new approach cleans the entire object of all non-serializable types
    # (like ObjectId and datetime) by using your existing custom serializer.
    
    json_string_context = json.dumps(analysis_context, default=custom_json_serializer)
    cleaned_analysis_context = json.loads(json_string_context)
    
    # ============================ END OF FIX ================================

    # Now, pass the fully cleaned context to the AI analyzer
    result = ai_analyzer_instance.get_conversational_answer(question, cleaned_analysis_context)
    
    if result:
        return jsonify(result)
    else:
        return jsonify({"error": "The AI failed to generate an answer."}), 500
 
@app.route('/api/trades/log', methods=['POST'])
@token_required
def log_user_trade(current_user): # Add the decorator and current_user
    log.info(f"API call to log trade for user: {current_user['uid']}")
    
    trade_log_data = request.json
    if not trade_log_data or 'ticker' not in trade_log_data:
        return jsonify({"error": "Invalid trade log data provided."}), 400

    try:
        # Pass the real user ID to your database function
        insert_id = database_manager.save_user_trade(trade_log_data, current_user['uid'])
        
        if insert_id:
            return jsonify({"status": "success", "trade_id": str(insert_id)}), 201
        else:
            return jsonify({"error": "Failed to save the trade log."}), 500
    except Exception as e:
        log.error(f"Error logging user trade: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/api/vote', methods=['POST'])
@token_required
@limiter.limit(LIMIT_VOTE)
def record_analysis_vote(current_user):
    log.info("API call received for /api/analysis/vote")
    
    data = request.json
    analysis_id = data.get('analysis_id')
    vote_type = data.get('vote_type')

    if not analysis_id or not vote_type:
        return jsonify({"error": "analysis_id and vote_type are required."}), 400

    try:
        result = database_manager.add_analysis_vote(analysis_id, vote_type, current_user['uid'])

        if result.get("success"):
            return jsonify({"status": "success", "message": f"Vote '{vote_type}' recorded."})
        else:
            return jsonify({"error": result.get("error", "Failed to record vote.")}), 500

    except Exception as e:
        log.error(f"Error recording vote: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500
    
@app.route('/api/analysis/votes/<string:analysis_id>', methods=['GET'])
def get_analysis_votes(analysis_id):
    log.info(f"API call received for /api/analysis/votes/{analysis_id}")
    try:
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
@token_required
@limiter.limit(LIMIT_FEEDBACK) # Prevent spam
def submit_feedback(current_user):
    log.info(f"API call received for /api/feedback/submit from user {current_user['uid']}")
    
    data = request.json
    
    # Automatically capture context
    data['context'] = {
        'page_url': request.headers.get('Referer'),
        'user_agent': request.headers.get('User-Agent')
    }

    if not data.get('feedback_text'):
        return jsonify({"error": "Feedback text is required."}), 400

    try:
        # Pass the user's ID to the database function
        insert_id = database_manager.save_feedback(data, current_user['uid'])
        
        if insert_id:
            return jsonify({"status": "success", "feedback_id": str(insert_id)}), 201
        else:
            return jsonify({"error": "Failed to save feedback."}), 500
    except Exception as e:
        log.error(f"Error submitting feedback: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500
    
@app.route('/api/faq/submit', methods=['POST'])
@token_required
@limiter.limit(LIMIT_FAQ) # Prevent spam
def submit_faq_question(current_user):
    log.info(f"API call received for /api/faq/submit from user {current_user['uid']}")
    
    data = request.json
    if not data.get('question_text'):
        return jsonify({"error": "Question text is required."}), 400

    try:
        # Pass the user's ID to the database function
        insert_id = database_manager.save_faq_submission(data, current_user['uid'])
        
        if insert_id:
            return jsonify({"status": "success", "submission_id": str(insert_id)}), 201
        else:
            return jsonify({"error": "Failed to save submission."}), 500
    except Exception as e:
        log.error(f"Error submitting FAQ question: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/api/all-analysis', methods=['GET'])
@cache.cached(timeout=900) # Cache the result of this function for 900 seconds (15 mins)
def get_all_analysis():
    log.info("API call received for /api/all-analysis - Running DB query.")
    try:
        results = list(database_manager.analysis_results_collection.find({}))
        return Response(json.dumps(results, default=custom_json_serializer), mimetype='application/json')
    except Exception as e:
        log.error(f"Failed to fetch all analysis: {e}", exc_info=True)
        return jsonify({"error": "Could not retrieve all analysis data."}), 500

@app.route('/api/my-trades', methods=['GET'])
@token_required
def get_my_trades_endpoint(current_user):
    log.info(f"API call to get trades for user: {current_user['uid']}")
    try:
        user_trades_collection = database_manager.db.user_trades
        # Find all trades that match the logged-in user's ID
        my_trades = list(user_trades_collection.find({'user_id': current_user['uid']}))
        
        # Sort trades by the date they were logged, with the newest first
        my_trades.sort(key=lambda x: x.get('logged_at'), reverse=True)
        
        return Response(json.dumps(my_trades, default=custom_json_serializer), mimetype='application/json')
    except Exception as e:
        log.error(f"Error fetching trades for user {current_user['uid']}: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred while fetching your trades."}), 500
    
if __name__ == '__main__':
    log.info("Initializing default database connection...")
    database_manager.init_db(purpose='analysis')
    
    # Determine if we're in development or production
    is_development = os.getenv('FLASK_ENV') == 'development'
    
    if is_development:
        log.warning("⚠️  RUNNING IN DEVELOPMENT MODE - Debug is enabled")
        app.run(host='0.0.0.0', port=5001, debug=True)
    else:
        log.info("Starting Flask server in PRODUCTION mode on http://localhost:5001")
        app.run(host='0.0.0.0', port=5001, debug=False)