# simple_seed.py
from datetime import datetime, timezone
import database_manager
from logger_config import log

# --- Sample Data 1: A "BUY" Signal ---
sample_data_1 = {
    "ticker": "RELIANCE.NS",
    "companyName": "Reliance Industries Ltd",
    "scrybeScore": 78,
    "signal": "BUY",
    "confidence": "High",
    "predicted_gain_pct": 8.5,
    "gain_prediction_rationale": "Strong technical breakout from a consolidation phase, supported by positive sector momentum.",
    "keyInsight": "Reliance is showing exceptional relative strength and is poised for a short-term up-move.",
    "analystVerdict": "The convergence of strong technicals, positive sector outlook, and supportive market regime creates a high-probability long setup. The stock has cleared a key resistance level and shows significant upside potential.",
    "keyRisks_and_Mitigants": {
        "risk_1": "A sudden broad-market downturn could negate the setup.",
        "risk_2": "Failure to hold the breakout level would indicate a false move.",
        "mitigant": "A strict stop-loss can be placed just below the breakout support level."
    },
    "thesisInvalidationPoint": "A daily close below 2800 would invalidate the bullish thesis.",
    "keyObservations": {
        "confluencePoints": ["Strong technical breakout", "Sector outperformance", "Bullish market regime"],
        "contradictionPoints": ["Valuation is slightly stretched, suggesting caution."]
    },
    "technical_indicators": {
        "RSI": 68.5,
        "ADX": 32.1,
        "Price vs EMA20": "Above",
    }
}

# --- Sample Data 2: A "HOLD" Signal ---
sample_data_2 = {
    "ticker": "TCS.NS",
    "companyName": "Tata Consultancy Services",
    "scrybeScore": 22,
    "signal": "HOLD",
    "confidence": "Low",
    "predicted_gain_pct": 2.1,
    "gain_prediction_rationale": "The stock is currently range-bound with conflicting technical signals.",
    "keyInsight": "TCS is in a neutral phase; waiting for a clearer directional signal is the prudent approach.",
    "analystVerdict": "While the company's fundamentals are strong, the technical picture is ambiguous. The stock is caught between its 20 and 50-day moving averages, and options data suggests a lack of immediate directional bias. No clear trading edge exists at this moment.",
     "keyRisks_and_Mitigants": {
        "risk_1": "A breakdown below key support could initiate a new downtrend.",
        "risk_2": "Choppy, directionless price action can lead to small, frustrating losses if a trade is forced.",
        "mitigant": "The best action is to remain on the sidelines until a clear trend emerges."
    },
    "thesisInvalidationPoint": "A breakout above 4100 or a breakdown below 3850 would establish a new trend.",
    "keyObservations": {
        "confluencePoints": ["Strong fundamental profile"],
        "contradictionPoints": ["Neutral technical indicators", "Lack of sector momentum", "Ambiguous options sentiment"]
    },
    "technical_indicators": {
        "RSI": 51.2,
        "ADX": 18.9,
        "Price vs EMA20": "Below",
    }
}

def seed_database():
    log.info("--- Starting Simple Database Seed ---")
    try:
        # Connect to the live application database
        database_manager.init_db('analysis')

        # Save the first sample document
        log.info(f"Seeding data for {sample_data_1['ticker']}...")
        database_manager.save_vst_analysis(sample_data_1['ticker'], sample_data_1)
        
        # Save the second sample document
        log.info(f"Seeding data for {sample_data_2['ticker']}...")
        database_manager.save_vst_analysis(sample_data_2['ticker'], sample_data_2)

        log.info("✅ Database seeding successful. Two sample documents were inserted.")

    except Exception as e:
        log.error(f"An error occurred during seeding: {e}", exc_info=True)
    finally:
        # Close the database connection
        database_manager.close_db_connection()

if __name__ == "__main__":
    seed_database()