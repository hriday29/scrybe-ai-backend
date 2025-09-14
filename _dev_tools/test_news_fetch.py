import os
import sys
import pprint
from dotenv import load_dotenv

# This is a crucial step to ensure the script can find your other modules
# like 'data_retriever' and 'config'.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Now that the path is set, we can import your modules
import data_retriever
import config  # This will load API keys from the environment

def test_single_ticker_news(ticker: str):
    """
    Runs an isolated test of the news fetching function for one ticker.
    """
    print("="*50)
    print(f"FETCHING LIVE NEWS FOR: {ticker}")
    print("="*50)

    # Load environment variables from your .env file
    load_dotenv()
    
    # Make sure the API key is loaded
    if not config.NEWSAPI_API_KEY:
        print("🔥 ERROR: NEWSAPI_API_KEY not found in environment.")
        print("Make sure it's in your .env file.")
        return

    # Call the function directly
    news_data = data_retriever.get_news_articles_for_ticker(ticker)

    print(f"\nResult Type: {news_data.get('type')}")
    print(f"Found {len(news_data.get('articles', []))} articles.")
    print("\n--- ARTICLES ---")
    
    # Pretty-print the results
    pprint.pprint(news_data.get('articles'))
    print("\nTest complete.")

if __name__ == "__main__":
    # You can change this ticker to test any stock you want
    ticker_to_test = "HINDALCO.NS"
    test_single_ticker_news(ticker_to_test)