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

    def __init__(self, api_key: str):
        if not api_key or "PASTE_YOUR" in api_key:
            raise ValueError("Gemini API key not configured or is invalid.")
        genai.configure(api_key=api_key)
        log.info("AIAnalyzer initialized and Gemini API key configured.")

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

    def get_apex_analysis(self, ticker: str, full_context: dict, strategic_review: str, tactical_lookback: str, per_stock_history: str) -> dict:
        """
        Generates a definitive, institutional-grade analysis using the "Apex" multi-layered model.
        """
        log.info(f"Generating APEX analysis for {ticker}...")

        system_instruction = """
        You are "Apex," the Chief Investment Officer of a high-conviction quantitative fund. Your analysis is legendary for its depth, clarity, and synthesis of disparate data points. You will produce an institutional-grade report for a given stock, culminating in a decisive **Scrybe Score** and a **Predicted Gain Percentage**.

        **PROTOCOL - STEP 1: REVIEW YOUR OWN PERFORMANCE.**
        Before you begin your analysis, you MUST first review the "Omni-Context" data provided: your "30-Day Performance Review" and your "Previous Day's Note." In your final verdict, you must explicitly state how this context has influenced your confidence and final score for today's decision.

        **PROTOCOL - STEP 2: MULTI-LAYERED THESIS FORMATION.**
        You will now synthesize the following six layers of intelligence into a single, cohesive thesis. You must weigh each layer according to the specified importance.

        - **Layer 1: Macro & Inter-market Context (Weight: 15%):** What is the risk environment? Are global markets, oil, and currencies providing tailwinds or headwinds?
        - **Layer 2: Sector & Relative Strength (Weight: 20%):** Is the stock's sector in favor? Crucially, is the stock stronger or weaker than the Nifty 50 itself? Relative strength is a key indicator of alpha.
        - **Layer 3: The Fundamental "Moat" (Weight: 15%):** Assess the company's financial health. Is this a durable business with strong profitability and manageable debt, or is it fundamentally weak?
        - **Layer 4: Multi-Timeframe Technicals (Weight: 30%):** This is your most heavily weighted factor. Analyze and state the trend on the **Weekly Chart (long-term context)**, the setup on the **Daily Chart (our trade timeframe)**, and the immediate momentum on the **15-Minute Chart (entry timing)**.
        - **Layer 5: Options Sentiment (Weight: 10%):** Where is the derivatives market placing its bets? Analyze the Put-Call Ratio and key Open Interest levels to gauge market positioning.
        - **Layer 6: The News Catalyst (Weight: 10%):** Is there a recent earnings report, news event, or story driving the price?

        **PROTOCOL - STEP 3: THE FINAL SYNTHESIS & PREDICTION.**
        1.  **Synthesize:** In your `analystVerdict`, provide a master narrative combining all six layers, starting with how your Omni-Context review shaped your thinking.
        2.  **Score:** Provide a `scrybeScore` from -100 to +100 based on your conviction.
        3.  **Predict Gain:** Provide a `predicted_gain_pct`. This is your realistic estimate of the potential profit for this trade setup over the strategy's holding period if the thesis plays out.
        4.  **Assess Risk:** You MUST identify the `keyRisks_and_Mitigants` to your thesis.
        5.  **Invalidate:** You MUST provide a specific `thesisInvalidationPoint` (a price or event) that would prove your analysis wrong.
        """

        output_schema = {
            "type": "OBJECT",
            "properties": {
                # --- Core Decision Metrics ---
                "scrybeScore": {
                    "type": "NUMBER",
                    "description": "The final conviction score from -100 (high conviction sell) to +100 (high conviction buy)."
                },
                "signal": {
                    "type": "STRING",
                    "enum": ["BUY", "SELL", "HOLD"],
                    "description": "The definitive trading signal derived from the Scrybe Score."
                },
                "confidence": {
                    "type": "STRING",
                    "enum": ["Low", "Medium", "High", "Very High"],
                    "description": "The qualitative confidence level, logically derived from the Scrybe Score."
                },
                "predicted_gain_pct": {
                    "type": "NUMBER",
                    "description": "The realistic estimated profit percentage for this trade setup over the holding period."
                },
                "gain_prediction_rationale": {
                    "type": "STRING",
                    "description": "A concise justification for the predicted gain percentage, based on volatility and technical targets."
                },
                "keyInsight": {
                    "type": "STRING",
                    "description": "The single most important, actionable takeaway from the entire analysis."
                },
                "analystVerdict": {
                    "type": "STRING",
                    "description": "The master narrative synthesizing all six layers, starting with the influence of the Omni-Context review."
                },

                # --- Risk Assessment ---
                "keyRisks_and_Mitigants": {
                    "type": "OBJECT",
                    "description": "The top two factors that could cause this trade to fail, and any mitigating factors.",
                    "properties": {
                        "risk_1": {"type": "STRING"},
                        "risk_2": {"type": "STRING"},
                        "mitigant": {"type": "STRING"}
                    },
                    "required": ["risk_1", "risk_2", "mitigant"]
                },
                "thesisInvalidationPoint": {
                    "type": "STRING",
                    "description": "A specific price level or event that, if breached, would definitively invalidate the entire thesis."
                },

                # --- Detailed Analysis Layers ---
                "keyObservations": {
                    "type": "OBJECT",
                    "description": "The key points of agreement and disagreement across the analytical layers.",
                    "properties": {
                        "confluencePoints": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "contradictionPoints": {"type": "ARRAY", "items": {"type": "STRING"}}
                    },
                    "required": ["confluencePoints", "contradictionPoints"]
                },
                "technicalBreakdown": {
                    "type": "OBJECT",
                    "description": "A detailed breakdown of the multi-timeframe technical analysis.",
                    "properties": {
                        "weekly_view": {"type": "OBJECT", "properties": {"status": {"type": "STRING"}, "interpretation": {"type": "STRING"}}},
                        "daily_view": {"type": "OBJECT", "properties": {"status": {"type": "STRING"}, "interpretation": {"type": "STRING"}}},
                        "intraday_view": {"type": "OBJECT", "properties": {"status": {"type": "STRING"}, "interpretation": {"type": "STRING"}}},
                        "ADX": {"type": "OBJECT", "properties": {"value": {"type": "STRING"}, "status": {"type": "STRING"}, "interpretation": {"type": "STRING"}}},
                        "RSI": {"type": "OBJECT", "properties": {"value": {"type": "STRING"}, "status": {"type": "STRING"}, "interpretation": {"type": "STRING"}}},
                        "MACD": {"type": "OBJECT", "properties": {"value": {"type": "STRING"}, "status": {"type": "STRING"}, "interpretation": {"type": "STRING"}}}
                    },
                    "required": ["weekly_view", "daily_view", "intraday_view", "ADX", "RSI", "MACD"]
                },
                "fundamentalBreakdown": {
                    "type": "OBJECT",
                    "description": "A summary of the company's financial health.",
                    "properties": {
                        "Valuation": {"type": "OBJECT", "properties": {"status": {"type": "STRING"}, "interpretation": {"type": "STRING"}}},
                        "Profitability": {"type": "OBJECT", "properties": {"status": {"type": "STRING"}, "interpretation": {"type": "STRING"}}},
                        "Ownership": {"type": "OBJECT", "properties": {"status": {"type": "STRING"}, "interpretation": {"type": "STRING"}}}
                    },
                    "required": ["Valuation", "Profitability", "Ownership"]
                },
                 "optionsBreakdown": {
                    "type": "OBJECT",
                    "description": "A summary of the options market sentiment.",
                    "properties": {
                        "sentiment": {"type": "STRING", "enum": ["Bullish", "Bearish", "Neutral", "Unavailable"]},
                        "key_level": {"type": "STRING", "description": "The most significant strike price from Max OI."}
                    },
                     "required": ["sentiment", "key_level"]
                },
                 "newsBreakdown": {
                    "type": "OBJECT",
                    "description": "A summary of any recent news catalyst.",
                    "properties": {
                        "sentiment": {"type": "STRING", "enum": ["Positive", "Negative", "Neutral", "None"]},
                        "summary": {"type": "STRING", "description": "A one-sentence summary of the key news."}
                    },
                    "required": ["sentiment", "summary"]
                }
            },
            "required": [
                "scrybeScore", "signal", "confidence", "predicted_gain_pct", "gain_prediction_rationale",
                "keyInsight", "analystVerdict", "keyRisks_and_Mitigants", "thesisInvalidationPoint",
                "keyObservations", "technicalBreakdown", "fundamentalBreakdown", "optionsBreakdown", "newsBreakdown"
            ]
        }

        generation_config = genai.types.GenerationConfig(response_mime_type="application/json", response_schema=output_schema, max_output_tokens=16384)
        model = genai.GenerativeModel(config.PRO_MODEL, system_instruction=system_instruction, generation_config=generation_config)

        prompt_parts = [
            "## Omni-Context Performance Review ##",
            f"30-Day Strategic Review:\n{strategic_review}",
            f"\nPer-Stock Recent Trade History ({ticker}):\n{per_stock_history}",
            f"\nPrevious Day's Tactical Note ({ticker}):\n{tactical_lookback}",
            "\n## Today's Full Data Packet ##",
            json.dumps(full_context)
        ]

        # The standard retry logic from your previous functions would go here
        max_retries = 4
        delay = 2
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt_parts, request_options={"timeout": 300})
                
                # --- START: YOUR ENHANCED LOGIC ---
                if not response.parts:
                    finish_reason = "Unavailable"
                    if hasattr(response, "candidates") and response.candidates:
                        finish_reason = getattr(response.candidates[0], "finish_reason", "Unknown")
                    
                    log.warning(
                        f"[AI] Attempt {attempt + 1} for {ticker} returned an empty response "
                        f"(finish_reason: {finish_reason}). Retrying..."
                    )
                    raise ValueError("Empty response from API")
                # --- END: YOUR ENHANCED LOGIC ---
                
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