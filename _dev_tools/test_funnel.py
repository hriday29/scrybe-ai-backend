# test_funnel.py
from logger_config import log
import market_regime_analyzer
import sector_analyzer
import quantitative_screener
import config

def run_funnel_test():
    """
    A simple script to test the entire data-gathering funnel without
    making any expensive AI calls.
    """
    log.info("--- STARTING LOCAL FUNNEL TEST ---")

    # Step 1: Get Market Regime
    regime = market_regime_analyzer.get_current_market_regime()
    if not regime:
        log.error("Market Regime step failed.")
        return

    # Step 2: Get Strong Sectors
    sectors = sector_analyzer.get_top_performing_sectors()
    if sectors is None: # Check for None in case of error
        log.error("Sector Analysis step failed.")
        return
    log.info(f"Top performing sectors identified: {sectors}")

    # Step 3: Generate Watchlist
    watchlist = quantitative_screener.generate_dynamic_watchlist(sectors)
    if watchlist is None: # Check for None in case of error
        log.error("Quantitative Screener step failed.")
        return

    log.info("--- ✅ LOCAL FUNNEL TEST COMPLETE ---")
    log.info(f"Final Dynamic Watchlist for today would be: {watchlist}")
    log.info(f"A total of {len(watchlist)} stocks would be sent for Apex AI Analysis.")

if __name__ == "__main__":
    run_funnel_test()