import yfinance as yf
import pandas as pd
import requests
import index_manager
import json
import os
import config
from logger_config import log
from datetime import datetime
from datetime import datetime, timezone
import pandas_ta as ta

# --- FIX: Ensure the cache directory exists on startup ---
CACHE_DIR = 'cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

CACHE_FILE = os.path.join(CACHE_DIR, 'sector_cache.json')
# --------------------------------------------------------

# In data_retriever.py

def load_sector_cache():
    """
    Safely loads the sector cache. If the file is corrupt or missing,
    it returns an empty dictionary.
    """
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)
                # Ensure it returns a dictionary even if file is empty
                return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            log.warning("Could not decode sector_cache.json. It might be corrupt. Starting fresh.")
            return {}
    return {}

# In data_retriever.py

def save_sector_cache(cache):
    """
    Saves the sector cache, but only if it contains a reasonable number
    of tickers, preventing incomplete saves.
    """
    # Sanity check: Only write to the file if the cache is substantial.
    if isinstance(cache, dict) and len(cache) > 40:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=4)
        log.info(f"Successfully saved a valid sector cache with {len(cache)} entries.")
    else:
        log.warning(f"Skipped saving sector cache because it was incomplete (had {len(cache)} entries). The existing valid cache file is preserved.")

def get_historical_stock_data(ticker_symbol: str, end_date=None):
    """
    Fetches historical daily OHLC stock data with a 1-year fallback.
    Accepts an optional 'end_date' for historical point-in-time analysis.
    """
    log.info(f"Fetching historical data for {ticker_symbol} from Yahoo Finance...")
    if end_date:
        log.info(f"--- Using historical end_date: {end_date} ---")
        
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Attempt to get 5 years of data, using the end_date if provided.
        df = ticker.history(period="5y", end=end_date)

        if df.empty:
            log.warning(f"No 5-year data found for {ticker_symbol}. Trying 1-year period as a fallback...")
            df = ticker.history(period="1y", end=end_date)

        if df.empty:
            log.warning(f"No historical data returned for {ticker_symbol} after fallback.")
            return None
            
        df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        df.index = df.index.tz_localize(None) # Remove timezone for consistency
        log.info(f"Successfully fetched {len(df)} historical data points for {ticker_symbol}.")
        return df.sort_index()

    except Exception as e:
        log.error(f"Error fetching historical data for {ticker_symbol}: {e}")
        return None

def get_live_financial_data(ticker_symbol: str):
    """Fetches curated live financial data."""
    log.info(f"Fetching all live financial data for {ticker_symbol}...")
    try:
        ticker = yf.Ticker(ticker_symbol)
        raw_info = ticker.info
        curated_data = {
            "currentPrice": raw_info.get("currentPrice", raw_info.get("previousClose")),
            "trailingPE": raw_info.get("trailingPE"),
            "priceToBook": raw_info.get("priceToBook"),
            "profitMargins": raw_info.get("profitMargins"),
            "heldPercentInstitutions": raw_info.get("heldPercentInstitutions")
        }
        if curated_data["currentPrice"] is None:
            log.error(f"No price data available for {ticker_symbol}.")
            return None
        log.info(f"Successfully fetched all live financial data for {ticker_symbol}.")
        return {"curatedData": curated_data, "rawDataSheet": raw_info}
    except Exception as e:
        log.error(f"Error fetching live financial data for {ticker_symbol}: {e}")
        return None

def get_nifty50_performance():
    """
    Fetches Nifty 50 performance using your original resilient "Hybrid Fetch" strategy.
    """
    log.info("Fetching Nifty 50 performance with Hybrid Fetch strategy...")
    nifty50_tickers = index_manager.get_nifty50_tickers()
    if not nifty50_tickers:
        return None

    try:
        log.info("Pass 1: Attempting fast batch download...")
        batch_data = yf.download(tickers=nifty50_tickers, period="5d", interval="1d", progress=False)
        
        if batch_data.empty:
            log.warning("The initial batch download returned an empty dataframe. This may happen on non-trading days.")
            return None

        successful_tickers = batch_data['Close'].dropna(how='all', axis=1).columns.tolist()
        missing_tickers = list(set(nifty50_tickers) - set(successful_tickers))
        all_data_frames = [batch_data]

        if missing_tickers:
            log.warning(f"Pass 2: {len(missing_tickers)} tickers failed in batch. Attempting individual fallback...")
            for ticker in missing_tickers:
                try:
                    individual_data = yf.download(tickers=ticker, period="2d", interval="1d", progress=False)
                    if not individual_data.empty:
                        all_data_frames.append(individual_data)
                        log.info(f"Rescued data for {ticker}")
                except Exception:
                    log.error(f"Failed to rescue data for {ticker}")
        
        if not all_data_frames:
            log.error("Failed to download any Nifty 50 price data.")
            return None

        combined_data = pd.concat(all_data_frames, axis=1)
        closing_prices = combined_data['Close'].dropna(how='all', axis=1).tail(2)

        if len(closing_prices) < 2:
            log.warning("Not enough historical data (less than 2 days) to calculate performance.")
            return None

        valid_tickers = closing_prices.columns[closing_prices.notna().all()]
        if not valid_tickers.any():
            log.warning("No stocks with valid 2-day price data found.")
            return None
        
        performance_df = closing_prices[valid_tickers].iloc[-1].to_frame('lastPrice').copy()
        performance_df['pctChange'] = ((closing_prices[valid_tickers].iloc[-1] - closing_prices[valid_tickers].iloc[0]) / closing_prices[valid_tickers].iloc[0]) * 100
        
        sector_cache = load_sector_cache()
        sectors = {ticker: sector_cache[ticker] for ticker in performance_df.index if ticker in sector_cache}
        tickers_to_fetch = [ticker for ticker in performance_df.index if ticker not in sector_cache]
        
        if tickers_to_fetch:
            log.info(f"Found {len(tickers_to_fetch)} new tickers. Fetching their sectors...")
            for ticker in tickers_to_fetch:
                try:
                    info = yf.Ticker(ticker).info
                    sector = info.get('sector', 'Other')
                    sectors[ticker] = sector
                    sector_cache[ticker] = sector
                except Exception:
                    sectors[ticker] = 'Other'
            save_sector_cache(sector_cache)
        
        # Ensure all tickers have a sector
        for ticker in performance_df.index:
            if ticker not in sectors:
                sectors[ticker] = sector_cache.get(ticker, 'Other')

        performance_df['sector'] = performance_df.index.map(sectors).fillna('Other')
        performance_df.dropna(inplace=True)
        sector_performance = performance_df.groupby('sector')['pctChange'].mean().to_dict()
        log.info(f"Successfully processed performance for {len(performance_df)} stocks.")
        return{"stock_performance": performance_df.to_dict('index'), "sector_performance": sector_performance}

    except Exception as e:
        log.error(f"An error occurred while fetching Nifty 50 performance: {e}")
        return None

def get_upcoming_events(ticker_symbol: str):
    """
    Fetches upcoming events from a more robust, manually curated list of
    recurring market-wide events.
    """
    log.info(f"Compiling upcoming events for {ticker_symbol}...")
    all_events = []
    today = pd.Timestamp.now()

    # This is a more comprehensive list with future dates
    major_economic_events = [
        {"date": "2025-07-12", "event": "CPI Inflation Data Release"},
        {"date": "2025-07-15", "event": "Start of Quarterly Results Season"},
        {"date": "2025-07-31", "event": "Monthly F&O Series Expiry"},
        {"date": "2025-08-01", "event": "Auto Sales Figures Release (Monthly)"},
        {"date": "2025-08-08", "event": "RBI Monetary Policy Meeting"},
        {"date": "2025-08-12", "event": "IIP Data Release (Monthly)"},
        {"date": "2025-08-28", "event": "Monthly F&O Series Expiry"},
        {"date": "2025-09-01", "event": "Auto Sales Figures Release (Monthly)"},
        {"date": "2025-09-12", "event": "CPI Inflation Data Release (Monthly)"},
    ]

    for evt in major_economic_events:
        event_date = pd.to_datetime(evt['date'])
        # The logic to check if the event is in the future remains the same
        if today <= event_date < today + pd.DateOffset(months=3):
            all_events.append(evt)

    # The yfinance part for company-specific events remains
    try:
        ticker = yf.Ticker(ticker_symbol)
        calendar_data = ticker.calendar
        if isinstance(calendar_data, pd.DataFrame) and not calendar_data.empty:
            if 'Earnings Date' in calendar_data.index:
                earnings_date_start = calendar_data.loc['Earnings Date', 0]
                if pd.notna(earnings_date_start):
                    event_text = f"Next Earnings Announcement around {earnings_date_start.strftime('%b %d, %Y')}"
                    all_events.append({"date": earnings_date_start.strftime('%Y-%m-%d'), "event": event_text})
    except Exception as e:
        log.warning(f"Could not fetch company-specific calendar events for {ticker_symbol}. Error: {e}")

    if all_events:
        all_events.sort(key=lambda x: x['date'])
        log.info(f"Successfully compiled a total of {len(all_events)} upcoming events.")
    else:
        log.warning(f"No upcoming events found for {ticker_symbol}.")
    
    return all_events

def get_benchmarks_data(period: str = "1y", end_date=None):
    """
    Fetches historical closing prices for all benchmark assets.
    Accepts an optional 'end_date' for historical point-in-time analysis.
    """
    log.info("Fetching historical data for all benchmarks...")
    if end_date:
        log.info(f"--- Using historical end_date: {end_date} ---")

    benchmark_tickers = {
        "Nifty50": "^NSEI", "USD-INR": "INR=X", "S&P 500": "^GSPC",
        "Dow Jones": "^DJI", "Nikkei 225": "^N225", "Crude Oil": "CL=F", "Gold": "GC=F"
    }
    try:
        data = yf.download(
            tickers=list(benchmark_tickers.values()), 
            period=period, 
            interval="1d", 
            progress=False,
            end=end_date
        )
        
        if data.empty:
            log.warning("Failed to download any benchmark data.")
            return None
            
        closing_prices = data['Close']
        closing_prices.rename(columns={v: k for k, v in benchmark_tickers.items()}, inplace=True)
        log.info("Successfully fetched benchmark data.")
        return closing_prices
        
    except Exception as e:
        log.error(f"An error occurred while fetching benchmark data: {e}")
        return None

def get_index_option_data(index_ticker: str) -> dict:
    """
    Fetches key option chain data for a given index.
    """
    log.info(f"Fetching option chain data for {index_ticker}...")
    try:
        ticker = yf.Ticker(index_ticker)
        
        # --- THIS IS THE FIX ---
        # First, check if there are any option expiry dates available
        if not ticker.options:
            log.warning(f"No options chain data found for {index_ticker}.")
            return None
        # -------------------

        # Get the option chain for the nearest expiry date
        opt_chain = ticker.option_chain(ticker.options[0])
        
        calls = opt_chain.calls
        puts = opt_chain.puts

        pcr = puts['openInterest'].sum() / calls['openInterest'].sum()
        max_oi_call_strike = calls.loc[calls['openInterest'].idxmax()]['strike']
        max_oi_put_strike = puts.loc[puts['openInterest'].idxmax()]['strike']

        log.info(f"Successfully calculated option metrics for {index_ticker}.")
        return {
            "pcr": round(pcr, 2),
            "max_oi_call": max_oi_call_strike,
            "max_oi_put": max_oi_put_strike
        }
    except Exception as e:
        log.warning(f"Could not fetch or process option chain data for {index_ticker}. Error: {e}")
        return None

# In data_retriever.py

def get_intraday_data(ticker_symbol: str):
    """Fetches recent 15-minute intraday data for a stock."""
    try:
        log.info(f"[FETCH] Getting 15-min intraday data for {ticker_symbol}...")
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="5d", interval="15m")
        if df.empty:
            log.warning(f"No intraday data returned for {ticker_symbol}.")
            return None
            
        # --- THIS IS THE FIX ---
        # Normalize column names to lowercase for consistency
        df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        # ----------------------

        return df
    except Exception as e:
        log.error(f"Error fetching intraday data for {ticker_symbol}: {e}")
        return None

# In data_retriever.py
# Make sure you have 'import requests' at the top of the file

def get_news_sentiment(ticker_symbol: str):
    """
    Fetches recent news for a stock from NewsAPI.org and performs
    a simple sentiment analysis to find negative catalysts.
    """
    log.info(f"[FETCH] Getting live news sentiment for {ticker_symbol}...")
    if not config.NEWS_API_KEY:
        log.warning("News API key not configured. Skipping news analysis.")
        return {"has_negative_catalyst": False, "latest_headline": "News API not configured."}

    # Define keywords that indicate negative sentiment for a stock
    negative_keywords = [
        'downgrade', 'scandal', 'probe', 'investigation', 'misses estimates', 
        'plummets', 'tumbles', 'falls', 'lawsuit', 'fraud', 'fine'
    ]
    
    # Clean the ticker for the search query (e.g., "RELIANCE.NS" -> "Reliance")
    query = ticker_symbol.split('.')[0]
    
    # Construct the API URL
    url = (f"https://newsapi.org/v2/everything?"
           f"q={query}"
           f"&language=en"
           f"&sortBy=publishedAt"
           f"&pageSize=5"
           f"&apiKey={config.NEWS_API_KEY}")

    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get('articles', [])

        if not articles:
            return {"has_negative_catalyst": False, "latest_headline": "No recent news found."}

        # Check titles and descriptions for negative keywords
        for article in articles:
            title = article.get('title') or ""
            description = article.get('description') or ""
            text_to_check = (title + description).lower()
            if any(keyword in text_to_check for keyword in negative_keywords):
                log.warning(f"Negative catalyst found for {ticker_symbol}: {article['title']}")
                return {
                    "has_negative_catalyst": True,
                    "latest_headline": article['title']
                }
        
        # If no negative keywords found in top articles
        return {
            "has_negative_catalyst": False,
            "latest_headline": articles[0]['title'] # Return the latest headline for context
        }

    except requests.exceptions.RequestException as e:
        log.error(f"Failed to fetch news for {ticker_symbol}: {e}")
        return {"has_negative_catalyst": False, "latest_headline": "Error fetching news."}

def get_options_data(ticker_symbol: str):
    """
    Fetches key options chain data for a stock, including PCR by OI and Volume,
    Max OI levels, and average Implied Volatility.
    """
    log.info(f"[FETCH] Getting immersive options chain data for {ticker_symbol}...")
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Get the option chain for the nearest expiration date
        # First, check if there are any option expiry dates available
        if not ticker.options:
            log.warning(f"No options chain data found for {ticker_symbol}.")
            return None
        
        opt = ticker.option_chain(ticker.options[0])
        calls = opt.calls
        puts = opt.puts

        # --- Calculate Key Metrics ---

        # 1. Put-Call Ratios (both are important)
        pcr_volume = puts['volume'].sum() / calls['volume'].sum() if calls['volume'].sum() > 0 else 0
        pcr_oi = puts['openInterest'].sum() / calls['openInterest'].sum() if calls['openInterest'].sum() > 0 else 0

        # 2. Max OI Strikes (Key Support & Resistance)
        max_oi_call_strike = calls.loc[calls['openInterest'].idxmax()]['strike']
        max_oi_put_strike = puts.loc[puts['openInterest'].idxmax()]['strike']
        
        # 3. Implied Volatility (Market's Fear/Expectation Gauge)
        # We calculate a weighted average IV for at-the-money options
        atm_calls = calls.iloc[(calls['strike'] - calls['lastPrice'].iloc[-1]).abs().argsort()[:5]]
        atm_puts = puts.iloc[(puts['strike'] - puts['lastPrice'].iloc[-1]).abs().argsort()[:5]]
        avg_iv = pd.concat([atm_calls['impliedVolatility'], atm_puts['impliedVolatility']]).mean()

        return {
            "put_call_ratio_volume": round(pcr_volume, 2),
            "put_call_ratio_oi": round(pcr_oi, 2),
            "max_oi_call_strike": max_oi_call_strike,
            "max_oi_put_strike": max_oi_put_strike,
            "average_iv": f"{avg_iv:.2%}" # Format as percentage
        }
        
    except Exception as e:
        log.warning(f"Could not retrieve options data for {ticker_symbol}. It may not be an F&O stock. Error: {e}")
        return None
    
def get_social_sentiment(ticker_symbol: str):
    """
    !! PLACEHOLDER FUNCTION !!
    Simulates fetching and analyzing sentiment from social media (e.g., X/Twitter).
    In a real system, this would use a library like snscrape or a dedicated API.
    """
    log.info(f"[FETCH] Getting social media sentiment for {ticker_symbol}...")
    # This mock response assumes neutral sentiment.
    # A real function would analyze recent posts for bearish/bullish keywords.
    return {
        "sentiment": "Neutral", 
        "source": "X/Twitter (Simulated)"
    }

def get_news_articles_for_ticker(ticker_symbol: str) -> dict:
    """
    Fetches news using a 2-step fallback: yfinance and then MarketAux.
    """
    log.info(f"[FETCH] Running news fetch for {ticker_symbol}...")

    # --- Attempt 1: yfinance (Highest Relevance) ---
    try:
        log.info("--> News Fetch Attempt 1: yfinance...")
        ticker = yf.Ticker(ticker_symbol)
        news_list = ticker.news
        if news_list:
            formatted_articles = [{
                "title": article.get('title'), "url": article.get('link'),
                "source_name": article.get('publisher'),
                "published_at": datetime.fromtimestamp(article['providerPublishTime'], tz=timezone.utc).isoformat(),
                "description": ""
            } for article in news_list]
            log.info("--> Success on Attempt 1: Found news via yfinance.")
            return {"type": "Ticker-Specific News", "articles": formatted_articles[:8]}
    except Exception:
        log.warning(f"--> yfinance news fetch failed for {ticker_symbol}. Trying fallback.")

    # --- Attempt 2: MarketAux (Reliable Fallback) ---
    api_key = config.MARKETAUX_API_KEY # Make sure to add MARKETAUX_API_KEY to your config.py
    if api_key:
        try:
            log.info("--> News Fetch Attempt 2: MarketAux...")
            # Clean the ticker for the API call (e.g., "RELIANCE.NS" -> "RELIANCE")
            query = ticker_symbol.split('.')[0]
            url = f"https://api.marketaux.com/v1/news/all?symbols={query}&language=en&api_token={api_key}"

            response = requests.get(url, timeout=10)
            response.raise_for_status()
            articles = response.json().get('data', [])

            if articles:
                # Reformat the articles to match what the frontend expects
                formatted_articles = [{
                    "title": article.get('title'),
                    "url": article.get('url'),
                    "source_name": article.get('source'),
                    "published_at": article.get('published_at'),
                    "description": article.get('description')
                } for article in articles]
                log.info("--> Success on Attempt 2: Found news via MarketAux.")
                return {"type": "Related Market News", "articles": formatted_articles[:5]} # Limit to 5 articles
        except Exception as e:
            log.error(f"--> MarketAux news fetch failed for {ticker_symbol}. Error: {e}")

    # Final fallback if all else fails
    return {"type": "No News Found", "articles": []}

def get_market_regime() -> str:
    """
    Determines the current market regime based on NIFTY 50's EMAs.
    Returns 'Bullish', 'Bearish', or 'Neutral'.
    """
    log.info("Determining current market regime...")
    try:
        # Fetch about a year's worth of data for the NIFTY 50 index
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="1y")

        if len(hist) < 100:
            log.warning("Not enough NIFTY 50 data to determine market regime.")
            return "Neutral"

        # Calculate 20, 50, and 100-day Exponential Moving Averages
        hist.ta.ema(length=20, append=True)
        hist.ta.ema(length=50, append=True)
        hist.ta.ema(length=100, append=True)
        hist.dropna(inplace=True)

        # Get the latest values
        latest_emas = hist.iloc[-1]
        ema_20 = latest_emas['EMA_20']
        ema_50 = latest_emas['EMA_50']
        ema_100 = latest_emas['EMA_100']

        # Determine the regime
        if ema_20 > ema_50 > ema_100:
            regime = "Bullish"
        elif ema_20 < ema_50 < ema_100:
            regime = "Bearish"
        else:
            regime = "Neutral"
        
        log.info(f"Current Market Regime determined as: {regime}")
        return regime

    except Exception as e:
        log.error(f"Failed to determine market regime: {e}")
        # Default to Neutral in case of any error
        return "Neutral"