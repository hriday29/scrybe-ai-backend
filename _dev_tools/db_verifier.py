# db_verifier.py
import os
import argparse
import pandas as pd
import pymongo
import certifi
from dotenv import load_dotenv

def verify_batch_signals(batch_id: str, db_uri: str):
    """
    Connects to the database and verifies the signals stored for a specific batch_id.
    """
    print("--- 🔍 Starting Database Verification Script ---")

    if not db_uri:
        print("❌ ERROR: Database URI not found. Make sure your .env file is set up correctly.")
        return

    try:
        print(f"Connecting to the database...")
        client = pymongo.MongoClient(db_uri, tlsCAFile=certifi.where())
        db = client.get_default_database()
        predictions_collection = db.predictions
        print("✅ Database connection successful.")
    except Exception as e:
        print(f"❌ ERROR: Could not connect to the database. Error: {e}")
        return

    # --- Investigation 1: Get ALL signals for the batch to see the distribution ---
    print(f"\n--- [Query 1] Checking all document signals for batch_id: '{batch_id}' ---")
    
    try:
        all_docs = list(predictions_collection.find({"batch_id": batch_id}))
        if not all_docs:
            print(f"‼️ Result: No documents found for batch_id '{batch_id}'.")
            print("   Possible issues: Is the batch_id spelled correctly? Did the orchestrator save any data?")
            client.close()
            return
            
        df = pd.DataFrame(all_docs)
        print(f"✅ Found a total of {len(df)} documents for this batch.")
        
        print("\nSignal distribution:")
        print(df['signal'].value_counts().to_string())

    except Exception as e:
        print(f"❌ ERROR during Query 1: {e}")
        client.close()
        return

    # --- Investigation 2: Run the EXACT query the backtester uses ---
    print(f"\n--- [Query 2] Running the backtester's query for tradeable signals ('BUY' or 'SELL') ---")
    
    try:
        tradeable_signals_query = {
            "batch_id": batch_id, 
            "signal": {"$in": ["BUY", "SELL"]}
        }
        tradeable_signals_count = predictions_collection.count_documents(tradeable_signals_query)

        if tradeable_signals_count > 0:
            print(f"✅ SUCCESS: Found {tradeable_signals_count} tradeable ('BUY' or 'SELL') signal(s) in the database.")
            print("   This confirms the orchestrator saved trades correctly and there is an issue in `stateful_backtester.py`.")
        else:
            print(f"‼️ RESULT: Found 0 tradeable ('BUY' or 'SELL') signals.")
            print("   This suggests all generated BUY/SELL signals were vetoed and saved as 'HOLD'.")

    except Exception as e:
        print(f"❌ ERROR during Query 2: {e}")
    finally:
        client.close()
        print("\n--- ✅ Verification complete. Database connection closed. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify signals in the database for a given batch ID.")
    parser.add_argument('--batch_id', required=True, help='The batch_id to verify.')
    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()
    scheduler_db_uri = os.getenv("SCHEDULER_DB_URI")
    
    verify_batch_signals(args.batch_id, scheduler_db_uri)