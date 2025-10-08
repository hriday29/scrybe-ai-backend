#ai_analyzer.py
import time
import google.generativeai as genai
import config
import json
import base64
import pandas as pd
import data_retriever
from logger_config import log
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import pandas_ta as ta

class AIAnalyzer:
    """A class to handle all interactions with the Generative AI model."""

    def __init__(self, api_key: str, model_name: str = None):
        if not api_key or "PASTE_YOUR" in api_key:
            raise ValueError("Gemini API key not configured or is invalid.")
        self.api_key = api_key
        self.model_name = model_name or config.PRO_MODEL  # default to your main model
        genai.configure(api_key=self.api_key)
        log.info("AIAnalyzer initialized and Gemini API key configured.")

    def update_api_key(self, new_key: str):
        """Updates the API key for the generative model client."""
        if not new_key or "PASTE_YOUR" in new_key:
            raise ValueError("Invalid or missing API key provided.")
        self.api_key = new_key
        genai.configure(api_key=self.api_key)
        log.info("AIAnalyzer API key has been updated.")

    def get_apex_analysis(self, ticker: str, full_context: dict, strategic_review: str, tactical_lookback: str, per_stock_history: str, model_name: str, screener_reason: str) -> dict:
    """
    Generates a definitive, institutional-grade analysis using the "Apex" multi-layered model.
    This version has the simplified error handling (fallback removed).
    """
    log.info(f"Generating APEX analysis for {ticker}...")

    # --- All your prompt and schema setup remains the same ---
    system_instruction = """
    You are "Scrybe," an expert-level quantitative AI analyst at a top-tier hedge fund. Your function is to conduct a rigorous, unbiased, multi-layered analysis of a stock to find a high-probability swing trade opportunity (5-10 day holding period).

    **ANALYTICAL MANDATE:**
    Your analysis must be a masterclass in synthesis. You will receive a comprehensive data packet. You must weigh all evidence, identify points of confluence and contradiction, and generate a definitive "BUY", "SELL", or "HOLD" signal with a clear, justifiable thesis.

    **CORE ANALYTICAL PROTOCOL:**
    1.  **Synthesize the Primary Layers:** Form a core thesis by integrating the Macro Context, Sector Strength, Fundamental Analysis, and Technical Analysis.
    2.  **Incorporate Market Internals:** Use the `market_internals` data (Options Sentiment and Order Book) to refine your thesis. Does the micro-structure of the market support your view?
    3.  **Consider Catalysts:** Evaluate the `news_and_events` data for any near-term catalysts that could accelerate or invalidate your thesis.
    4.  **Apply Self-Correction (Backtest Only):** If `performance_feedback` is provided, use it as a self-correction mechanism. If recent trades of a similar thesis type have failed, you must state this and justify why this new trade is different, or adopt a more cautious stance.

    **ADVANCED DATA SYNTHESIS INSTRUCTIONS:**
    -   **Macro Context:** Do not just state the regime. Explain **how** the performance of global indices, commodities, and market breadth influences your risk appetite for this specific stock. A strong Nifty 50 trend with weak market breadth is a low-quality signal.
    -   **Market Internals:** Explicitly analyze the **Order Book Imbalance (OBI)** for immediate supply/demand pressure. In your options analysis, discuss what the **Greeks and IV** imply about market maker positioning and future volatility expectations.
    -   **Fundamentals:** Correlate your thesis with the company's **quarterly growth metrics and shareholder changes**. A bullish technical pattern is stronger if backed by accelerating earnings and increased institutional holding.

    **FINAL OUTPUT REQUIREMENTS:**
    1.  **scrybeScore:** Your final conviction score from **-100 (max conviction SELL)** to **+100 (max conviction BUY)**. A score between -25 and +25 is neutral (HOLD).
    2.  **signal:** "BUY", "SELL", or "HOLD". This must be a direct consequence of your score.
    3.  **thesisType:** Your classification of the setup (e.g., "Momentum Breakout in Strong Sector", "Bearish Breakdown on Weak Fundamentals", "Mean Reversion Against Frothy Sentiment").
    4.  **analystVerdict:** Your master narrative. This must be a concise, powerful summary that starts with your conclusion and then justifies it by synthesizing the most critical data points from your analysis above.
    5.  **keyInsight:** The single most important takeaway for a human fund manager to read.
    """
    output_schema = {
        "type": "OBJECT",
        "properties": {
            "scrybeScore": {"type": "NUMBER"},
            "signal": {"type": "STRING", "enum": ["BUY", "SELL", "HOLD"]},
            "thesisType": {"type": "STRING"},
            "confidence": {"type": "STRING", "enum": ["Low", "Medium", "High", "Very High"]},
            "predicted_gain_pct": {"type": "NUMBER"},
            "gain_prediction_rationale": {"type": "STRING"},
            "keyInsight": {"type": "STRING"},
            "analystVerdict": {"type": "STRING"},
            "keyRisks_and_Mitigants": {
                "type": "OBJECT",
                "properties": {
                    "risk_1": {"type": "STRING"},
                    "risk_2": {"type": "STRING"},
                    "mitigant": {"type": "STRING"}
                },
                "required": ["risk_1", "risk_2", "mitigant"]
            },
            "thesisInvalidationPoint": {"type": "STRING"},
            "keyObservations": {
                "type": "OBJECT",
                "properties": {
                    "confluencePoints": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "contradictionPoints": {"type": "ARRAY", "items": {"type": "STRING"}}
                },
                "required": ["confluencePoints", "contradictionPoints"]
            }
        },
        "required": [
            "scrybeScore", "signal", "confidence", "predicted_gain_pct", "gain_prediction_rationale",
            "keyInsight", "analystVerdict", "keyRisks_and_Mitigants", "thesisInvalidationPoint", "keyObservations", "thesisType"
        ]
    }

    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json",
        response_schema=output_schema,
        max_output_tokens=8192
    )

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    formatted_context = "\n## Today's Full Data Packet ##\n" + json.dumps(full_context, indent=2)
    prompt_parts = [
        f"Performance Feedback (Strategy): {strategic_review or 'Data Not Available'}",
        f"Performance Feedback ({ticker}): {per_stock_history or 'Data Not Available'}",
        f"Tactical Lookback Notes: {tactical_lookback or 'Data Not Available'}",
        f"Screener Reason for selecting {ticker}: **{screener_reason}**",
        formatted_context
    ]

    # --- Simplified Main Logic (No fallback) ---
    log.info(f"Primary attempt with {model_name}...")
    for attempt in range(3):
        try:
            model = genai.GenerativeModel(
                model_name,
                system_instruction=system_instruction,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            response = model.generate_content(prompt_parts, request_options={"timeout": 300})

            if not response.parts:
                raise ValueError(f"Empty or blocked response. Feedback: {response.prompt_feedback}")

            analysis_result = json.loads(response.text)
            analysis_result["model_used"] = model_name
            log.info(f"✅ Success on attempt {attempt + 1} with {model_name}.")
            return analysis_result

        except Exception as e:
            log.warning(f"Attempt {attempt + 1} with {model_name} failed with a transient error: {e}")
            if attempt < 2:
                time.sleep(5)

    # --- End of Function (no fallback) ---
    raise Exception(f"Primary model ({model_name}) failed after all 3 attempts with transient errors.")

    def get_single_news_impact_analysis(self, article: dict) -> dict:
        """
        Analyzes a single news article to determine its likely impact on a stock's price.
        """
        log.info(f"[AI] Getting impact analysis for single news article...")

        system_instruction = """
        You are a senior investment analyst at a hedge fund. Your task is to analyze a single news headline and its description to determine its immediate, short-term impact on the associated stock's price.

        **Your Three-Step Protocol:**
        1.  **Summarize the Core Event:** In one brief sentence, what is the key takeaway from the article?
        2.  **Determine the Impact:** Classify the likely impact as 'Positive', 'Negative', or 'Neutral'.
        3.  **Provide a Concise Rationale:** In one or two sentences, explain your reasoning. If the news is market-wide, explain how it affects this specific stock. If it's company-specific, explain its significance.

        Your final output must be a JSON object.
        """

        output_schema = {
            "type": "OBJECT", "properties": {
                "impact": {"type": "STRING", "enum": ["Positive", "Negative", "Neutral"]},
                "summary": {"type": "STRING"},
                "rationale": {"type": "STRING"}
            }, "required": ["impact", "summary", "rationale"]
        }

        prompt = f"""
        Analyze the following news article and provide your impact assessment in the required JSON format.
        
        **Title:** {article.get('title', 'N/A')}
        **Description:** {article.get('description', 'N/A')}
        """

        generation_config = genai.types.GenerationConfig(response_mime_type="application/json", response_schema=output_schema)
        model = genai.GenerativeModel(config.FLASH_MODEL, system_instruction=system_instruction, generation_config=generation_config)

        try:
            response = model.generate_content(prompt, request_options={"timeout": 120})
            return json.loads(response.text)
        except Exception as e:
            log.error(f"Single news impact analysis call failed. Error: {e}")
            return None

    def get_simple_momentum_signal(self, ticker: str, technical_indicators: dict) -> dict:
        """
        Analyzes a stock's technical indicators to generate a simple, rules-based momentum signal.
        """
        log.info(f"Generating simple momentum signal for {ticker}...")

        system_instruction = """
        You are a technical analyst specializing in identifying high-probability momentum swing trades. Your task is to analyze a stock's daily technical indicators and determine if it is in a strong, established trend suitable for a trade.

        **YOUR CRITERIA:**
        1.  **Trend Confirmation (ADX):** A strong trend is indicated by an ADX value above 25.
        2.  **Momentum (RSI):** In an uptrend, the RSI should be above 50, showing bullish momentum. In a downtrend, it should be below 50.
        3.  **Trend Alignment (Moving Averages):** The closing price must be above the 20-day and 50-day moving averages for a BUY signal, and below for a SELL signal.

        **YOUR RESPONSE:**
        You must respond with a JSON object. Based ONLY on the criteria above, determine the signal. If all criteria for a BUY or SELL are met, provide a high `convictionScore`. If some but not all are met, provide a lower score. If criteria are contradictory, the signal must be HOLD.
        """

        output_schema = {
            "type": "OBJECT",
            "properties": {
                "signal": {"type": "STRING", "enum": ["BUY", "SELL", "HOLD"]},
                "convictionScore": {"type": "NUMBER", "description": "A score from 0-100 based on how well the criteria are met."},
                "rationale": {"type": "STRING", "description": "A brief, one-sentence explanation of your decision based on the criteria."}
            },
            "required": ["signal", "convictionScore", "rationale"]
        }

        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json", 
            response_schema=output_schema
        )
        
        # Using FLASH model for efficiency on this simple task
        model = genai.GenerativeModel(
            config.FLASH_MODEL, 
            system_instruction=system_instruction, 
            generation_config=generation_config
        )

        prompt_parts = [
            f"Analyze the following technical data for {ticker} based ONLY on the rules provided in the system instruction.",
            json.dumps(technical_indicators)
        ]

        max_retries = 4
        delay = 2
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt_parts, request_options={"timeout": 120})
                
                if not response.parts:
                    finish_reason = "Unavailable"
                    if hasattr(response, "candidates") and response.candidates:
                        finish_reason = getattr(response.candidates[0], "finish_reason", "Unknown")
                    
                    log.warning(
                        f"[AI] Attempt {attempt + 1} for {ticker} returned an empty response "
                        f"(finish_reason: {finish_reason}). Retrying..."
                    )
                    raise ValueError("Empty response from API")
                
                return json.loads(response.text)
            except Exception as e:
                log.warning(f"[AI] Attempt {attempt + 1} for {ticker} failed. Error: {e}")
                if "429" in str(e) and "quota" in str(e).lower():
                    log.error("Quota exceeded. Raising exception to trigger key rotation.")
                    raise e
                if attempt < max_retries - 1:
                    log.info(f"Waiting for {delay} seconds before retrying...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    log.error(f"[AI] Final attempt failed for {ticker}. Skipping analysis.")
                    return None

    def get_mean_reversion_signal(self, ticker: str, technical_indicators: dict) -> dict:
        """
        Analyzes technicals to find mean-reversion trades in non-trending markets.
        """
        log.info(f"Generating mean-reversion signal for {ticker}...")

        system_instruction = """
        You are a technical analyst specializing in identifying high-probability mean-reversion swing trades in choppy or range-bound markets. Your task is to identify when a stock is oversold or overbought and likely to revert to its average price.

        **YOUR CRITERIA:**
        1.  **Non-Trending Confirmation (ADX):** The market must be choppy. This is indicated by an ADX value BELOW 22. This is the most important rule. If ADX is 22 or higher, there is a trend, and you must HOLD.
        2.  **Oversold/Overbought (RSI):** For a BUY signal, the RSI should be in an 'oversold' territory (below 40). For a SELL signal, the RSI must be in an 'overbought' territory (above 60).
        3.  **Price Location (Bollinger Bands):** For a BUY signal, the closing price should be NEAR or below the Lower Bollinger Band. For a SELL signal, the closing price should be NEAR or above the Upper Bollinger Band.

        **YOUR RESPONSE:**
        You must respond with a JSON object. Based ONLY on the criteria above, determine the signal. A signal is only valid if the ADX is LOW. Provide a high `convictionScore` if all criteria are met.
        """

        output_schema = {
            "type": "OBJECT",
            "properties": {
                "signal": {"type": "STRING", "enum": ["BUY", "SELL", "HOLD"]},
                "convictionScore": {"type": "NUMBER", "description": "A score from 0-100 based on how well the criteria are met."},
                "rationale": {"type": "STRING", "description": "A brief, one-sentence explanation of your decision based on the criteria."}
            },
            "required": ["signal", "convictionScore", "rationale"]
        }

        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json", 
            response_schema=output_schema
        )
        
        model = genai.GenerativeModel(
            config.FLASH_MODEL, 
            system_instruction=system_instruction, 
            generation_config=generation_config
        )

        prompt_parts = [
            f"Analyze the following technical data for {ticker} for a mean-reversion opportunity based ONLY on the rules provided in the system instruction.",
            json.dumps(technical_indicators)
        ]

        try:
            response = model.generate_content(prompt_parts, request_options={"timeout": 120})
            return json.loads(response.text)
        except Exception as e:
            # --- THIS IS THE FIX ---
            # If it's a quota error, we MUST raise it so the runner can catch it and rotate the key.
            if "429" in str(e):
                raise e
            # For any other error, we log it and return None so the script can continue.
            else:
                log.error(f"[AI] Mean reversion analysis for {ticker} failed with a non-quota error: {e}")
                return None
            # --- END OF FIX ---
    
    def get_intraday_short_signal(self, prompt_data: dict) -> dict:
        """
        Analyzes holistic data to find high-probability intraday short candidates
        using the powerful Pro model.
        """
        ticker = prompt_data.get("ticker", "N/A")
        log.info(f"[AI] Getting intraday short signal for {ticker} with FLASH model...")

        system_instruction = """
        You are an elite HFT analyst specializing in identifying high-probability intraday short-selling opportunities for the NEXT trading session. Your analysis must be swift, data-driven, and decisive.

        **PRIME SHORT CHECKLIST:**
        A stock only qualifies as a 'PRIME SHORT' if it meets at least TWO of the following four criteria, indicating overwhelming bearish confluence:

        1.  **SECTOR WEAKNESS:** Is the stock's broader sector (e.g., NIFTY IT) also showing weakness? A stock fighting its sector's trend is a low-probability trade.
        2.  **NEGATIVE SENTIMENT:** Is there a specific negative news catalyst or extremely bearish options market data (e.g., high Put-Call Ratio)?
        3.  **BEARISH INTRADAY TECHNICALS:** Is the stock showing clear weakness on its 15-minute chart? Specifically, is it trading **below its VWAP** and has it recently broken a key intraday support level?
        4.  **DAILY CHART CONTEXT:** Is the daily chart setup consistent with a downward move? (e.g., is it already in a downtrend or rejecting a major resistance level?).

        **Final Output:**
        Based on this multi-factor analysis, respond with a JSON object containing your signal ('PRIME SHORT', 'NEUTRAL', or 'AVOID SHORT'), your confidence level, and a concise rationale explaining which criteria were met to justify your final, decisive call.
        """

        output_schema = {
            "type": "OBJECT", "properties": {
                "signal": {"type": "STRING", "enum": ["PRIME SHORT", "NEUTRAL", "AVOID SHORT"]},
                "confidence": {"type": "STRING", "enum": ["Low", "Medium", "High", "Very High"]},
                "finalRationale": {"type": "STRING"},
                "checklist": {
                    "type": "OBJECT", "properties": {
                        "sectorWeakness": {"type": "OBJECT", "properties": {"met": {"type": "BOOLEAN"}, "details": {"type": "STRING"}}},
                        "newsSentiment": {"type": "OBJECT", "properties": {"met": {"type": "BOOLEAN"}, "details": {"type": "STRING"}}},
                        "intradayTechnicals": {"type": "OBJECT", "properties": {"met": {"type": "BOOLEAN"}, "details": {"type": "STRING"}}},
                        "optionsPressure": {"type": "OBJECT", "properties": {"met": {"type": "BOOLEAN"}, "details": {"type": "STRING"}}}
                    }
                }
            }, "required": ["signal", "confidence", "finalRationale", "checklist"]
        }
        
        prompt = f"Analyze the following data for {ticker} and provide your signal: {json.dumps(prompt_data)}"

        generation_config = genai.types.GenerationConfig(response_mime_type="application/json", response_schema=output_schema)
        
        # --- Using the FLASH_MODEL as requested ---
        model = genai.GenerativeModel(config.FLASH_MODEL, system_instruction=system_instruction, generation_config=generation_config)

        try:
            response = model.generate_content(prompt, request_options={"timeout": 120})
            return json.loads(response.text)
        except Exception as e:
            log.error(f"Intraday short analysis call failed for {ticker}. Error: {e}")
            return None

    def get_index_analysis(self, index_name: str, index_ticker: str, macro_context: dict) -> dict:
            """
            Generates a CIO-grade, in-depth analysis for a market index, including a fallback for deriving key levels from technicals if options data is unavailable.
            """
            log.info(f"Generating definitive CIO-grade index analysis for {index_name}...")
            
            system_instruction = """
            You are the Chief Investment Officer (CIO) of a global macro fund. Your task is to synthesize technical, macroeconomic, and (when available) options market data into a clear, institutional-grade strategic report on a major stock market index.

            **Multi-Factor Synthesis Protocol:**
            1.  **Technical Health Assessment:** Analyze the provided technical indicator data (Price vs. MAs, RSI) to determine the current trend strength and momentum.
            2.  **Key Levels Analysis (CRITICAL):**
                - If options data (Max OI) is provided, you MUST use the high OI strike prices as the primary psychological support/resistance levels.
                - **If options data is "Not Available", you MUST derive the Key Support and Resistance levels from the technical data provided (e.g., recent swing highs/lows, significant moving averages).** The report must contain key levels.
            **3. Correlation & Context Check:**
            - Analyze the stock's correlation to the NIFTY 50. Is it moving with the market (high positive correlation) or against it?
            - A 'BUY' signal in a stock that is strongly correlated with a bearish NIFTY 50 requires extra caution and a pristine setup. Acknowledge this context in your verdict.
            4.  **Sentiment Analysis:** Use the Put-Call Ratio (PCR) and Volatility Index (VIX) to gauge current market sentiment.
            5.  **Macroeconomic Overlay:** Interpret how the macro data influences the index's trajectory.
            6.  **Final Synthesis:** Combine all factors to produce a cohesive report, completing all fields in the required JSON format with detailed, well-reasoned insights.
            """

            output_schema = {
                "type": "OBJECT", "properties": {
                    "marketPulse": {"type": "OBJECT", "properties": {"overallBias": {"type": "STRING"}, "volatilityIndexStatus": {"type": "STRING"}}, "required": ["overallBias", "volatilityIndexStatus"]},
                    "trendAnalysis": {"type": "OBJECT", "properties": {"shortTermTrend": {"type": "STRING"}, "mediumTermTrend": {"type": "STRING"}, "keyTrendIndicators": {"type": "STRING"}}, "required": ["shortTermTrend", "mediumTermTrend", "keyTrendIndicators"]},
                    "keyLevels": {"type": "OBJECT", "properties": {"resistance": {"type": "ARRAY", "items": {"type": "STRING"}}, "support": {"type": "ARRAY", "items": {"type": "STRING"}}}, "required": ["resistance", "support"]},
                    "optionsMatrix": {"type": "OBJECT", "properties": {"maxOpenInterestCall": {"type": "STRING"}, "maxOpenInterestPut": {"type": "STRING"}, "putCallRatioAnalysis": {"type": "STRING"}}, "required": ["maxOpenInterestCall", "maxOpenInterestPut", "putCallRatioAnalysis"]},
                    "forwardOutlook": {"type": "OBJECT", "properties": {"next7Days": {"type": "STRING"}, "primaryRisk": {"type": "STRING"}}, "required": ["next7Days", "primaryRisk"]}
                }, "required": ["marketPulse", "trendAnalysis", "keyLevels", "optionsMatrix", "forwardOutlook"]
            }
            
            historical_data = data_retriever.get_historical_stock_data(index_ticker)
            vix_data = data_retriever.get_historical_stock_data("^INDIAVIX")
            options_data = data_retriever.get_index_option_data(index_ticker)
            
            if historical_data is None or len(historical_data) < 50:
                return {"error": "Not enough historical data to analyze the index."}
                
            historical_data.ta.rsi(length=14, append=True)
            historical_data.ta.ema(length=20, append=True)
            historical_data.ta.ema(length=50, append=True)
            historical_data.dropna(inplace=True)
            latest_data = historical_data.iloc[-1]
            
            vix_value_str = f"{vix_data.iloc[-1]['close']:.2f}" if vix_data is not None and not vix_data.empty else "Not Available"
            options_data_str = json.dumps(options_data) if options_data else "Not Available"

            
            prompt = f"""
            Generate a CIO-level strategic report for the {index_name}. Synthesize all of the following data:
            - Latest Technicals: Price({latest_data['close']:.2f}), 20-EMA({latest_data['EMA_20']:.2f}), 50-EMA({latest_data['EMA_50']:.2f}), RSI({latest_data['RSI_14']:.2f})
            - Latest Volatility Index (India VIX): {vix_value_str}
            - Latest Options Data: {options_data_str}
            - Macroeconomic Context: {json.dumps(macro_context)}
            Provide your full analysis in the required JSON format. If options data is not available, you must still provide key support and resistance levels based on technicals.
            """
            
            generation_config = genai.types.GenerationConfig(response_mime_type="application/json", response_schema=output_schema)
            model = genai.GenerativeModel(config.FLASH_MODEL, system_instruction=system_instruction, generation_config=generation_config)
            
            try:
                response = model.generate_content(prompt, request_options={"timeout": 120})
                return json.loads(response.text)
            except Exception as e:
                log.error(f"Index analysis AI call failed for {index_name}. Error: {e}")
                return None

    def get_volatility_qualifier(self, technical_indicators: dict) -> str:
        """
        A simple, fast, and deterministic function to classify market volatility.
        This version uses direct Python logic instead of an AI call for reliability.
        """
        log.info("Getting volatility qualifier...")
        try:
            bbw_value = float(technical_indicators.get("Bollinger Band Width Percent", 0))

            if bbw_value < 6:
                qualifier = "Quiet Consolidation"
            elif bbw_value > 15:
                qualifier = "Volatile Move"
            else:
                qualifier = "Steady Climb"
            
            log.info(f"Successfully got volatility qualifier: '{qualifier}'")
            return qualifier

        except Exception as e:
            log.error(f"[Volatility Qualifier] An unexpected error occurred: {e}")
            return ""
            
    def get_dvm_scores(self, live_financial_data: dict, technical_indicators: dict) -> dict:
        """A method to generate Durability, Valuation, and Momentum scores."""
        log.info("Generating DVM scores...")
        max_retries = 4
        delay = 2 # Initial delay of 2 seconds
        for attempt in range(max_retries):
            try:
                system_instruction = """
                You are a stringent quantitative financial analyst. Your job is to generate three scores (Durability, Valuation, Momentum) and a corresponding descriptive phrase for each.

                **CRITICAL RULES:**
                1.  The `score` MUST be on a scale of 0 to 100, where 100 is the best possible outcome and 0 is the worst.
                2.  The `phrase` MUST be logically consistent with the numeric `score`. A low score requires a cautious or negative phrase. A high score requires a positive phrase. There can be no contradictions.
                3.  The `status` must also align. Scores below 40 are 'Poor', 40-60 are 'Neutral', and above 60 are 'Good'.

                **Example of a GOOD, LOGICAL output:**
                {
                "durability": {
                    "score": 85,
                    "status": "Good",
                    "phrase": "The company shows excellent financial health with low debt and strong cash flow, indicating superior durability."
                }
                }

                **Example of a BAD, CONTRADICTORY output (DO NOT DO THIS):**
                {
                "durability": {
                    "score": 15,
                    "status": "Poor",
                    "phrase": "The company shows excellent financial health."
                }
                }

                Your analysis must be strict and the scores must directly reflect the data provided.
                """
                output_schema = {
                    "type": "OBJECT", "properties": {
                        "durability": {"type": "OBJECT", "properties": {"score": {"type": "NUMBER"}, "status": {"type": "STRING"}, "phrase": {"type": "STRING"}}, "required": ["score", "status", "phrase"]},
                        "valuation": {"type": "OBJECT", "properties": {"score": {"type": "NUMBER"}, "status": {"type": "STRING"}, "phrase": {"type": "STRING"}}, "required": ["score", "status", "phrase"]},
                        "momentum": {"type": "OBJECT", "properties": {"score": {"type": "NUMBER"}, "status": {"type": "STRING"}, "phrase": {"type": "STRING"}}, "required": ["score", "status", "phrase"]},
                    }, "required": ["durability", "valuation", "momentum"]
                }
                prompt_parts = [
                    "Generate the DVM scores based on this data:",
                    f"Financial Data: {json.dumps(live_financial_data['curatedData'])}",
                    f"Technical Indicators: {json.dumps(technical_indicators)}"
                ]
                generation_config = genai.types.GenerationConfig(response_mime_type="application/json", response_schema=output_schema)
                model = genai.GenerativeModel(config.FLASH_MODEL, system_instruction=system_instruction, generation_config=generation_config)
                
                response = model.generate_content(prompt_parts)
                return json.loads(response.text) # Success: return and exit loop
            except Exception as e:
                if "429" in str(e) and "quota" in str(e).lower():
                    log.error("Quota exceeded for DVM Scoring. Raising exception to trigger key rotation.")
                    raise e # CRITICAL: Alert the main runner
                log.warning(f"[DVM Scoring] Attempt {attempt + 1} failed. Error: {e}")
                if attempt < max_retries - 1:
                    log.info(f"Waiting for {delay} seconds before retrying...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    log.error(f"[DVM Scoring] Failed after {max_retries} attempts.")
                    return None
    
    def get_conversational_answer(self, question: str, analysis_context: dict) -> dict:
        """
        Answers a user's question based on the context of a specific stock analysis.
        """
        log.info(f"[AI] Answering a conversational question...")

        system_instruction = """
        You are an expert financial analyst AI. A user is asking a question about a stock analysis report you have already generated.
        Your task is to answer the user's question CONCISELY and STRICTLY based on the JSON data provided in the 'Analysis Context'.

        **CRITICAL RULES:**
        1.  **DO NOT** use any external knowledge or real-time information. Your entire universe of information is the provided JSON context.
        2.  If the question cannot be answered from the provided data, you MUST respond with: "I'm sorry, but I cannot answer that question with the available analysis data."
        3.  Keep your answers short and to the point (2-3 sentences maximum).
        """

        prompt = f"""
        **Analysis Context:**
        ```json
        {json.dumps(analysis_context, indent=2)}
        ```

        **User's Question:** "{question}"

        Please provide your answer based on the rules.
        """
        
        # For conversational answers, we don't need a strict JSON output schema from the AI
        model = genai.GenerativeModel(config.FLASH_MODEL, system_instruction=system_instruction)

        try:
            response = model.generate_content(prompt, request_options={"timeout": 120})
            # We wrap the text response in our own JSON object for the frontend
            return {"answer": response.text}
        except Exception as e:
            log.error(f"Conversational Q&A call failed. Error: {e}")
            return None
