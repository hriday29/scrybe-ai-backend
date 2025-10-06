# test_angelone_connection.py
import os
from dotenv import load_dotenv
from logger_config import log
import angelone_retriever
import config

log.info("--- Starting Angel One Connection Test ---")

# Load environment variables from .env file
load_dotenv()

# Check if necessary variables are loaded
if not all([config.ANGELONE_API_KEY, config.ANGELONE_CLIENT_ID, config.ANGELONE_PASSWORD, config.ANGELONE_TOTP]):
    log.error("One or more Angel One credentials are missing in your .env file. Aborting.")
else:
    log.info("All Angel One credentials found in environment.")

    # Attempt to initialize the session
    success = angelone_retriever.initialize_angelone_session()

    if success:
        log.info("✅✅✅ LOGIN SUCCESSFUL!")
        log.info("--- Attempting to fetch Nifty 50 quote as a final test... ---")
        
        # As a final proof, try to fetch a simple piece of data
        nifty_quote = angelone_retriever.get_full_quote("NIFTY 50", exchange="NSE")
        
        if nifty_quote:
            log.info(f"✅✅✅ DATA FETCH SUCCESSFUL! Nifty 50 LTP: {nifty_quote.get('ltp')}")
            log.info("--- Angel One Connection is working perfectly. ---")
        else:
            log.error("❌❌❌ DATA FETCH FAILED! Login was successful, but could not fetch data.")
    else:
        log.error("❌❌❌ LOGIN FAILED! Please double-check your credentials in the .env file and the TOTP secret.")