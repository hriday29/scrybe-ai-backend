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

    def get_momentum_analysis(self, live_financial_data: dict, latest_atr: float, model_name: str, charts: dict, trading_horizon_text: str, technical_indicators: dict, min_rr_ratio: float, market_context: dict, options_data: dict, macro_data: dict) -> dict:
        """Analyzes a stock for a trend-following momentum opportunity."""
        log.info(f"Generating Top Grade Analysis for {live_financial_data['rawDataSheet'].get('symbol', '')}...")

        # THE DEFINITIVE PROMPT
        definitive_scoring_prompt = f"""
        You are "Scrybe-Oracle," a world-class quantitative analyst specializing in MOMENTUM strategies. Your primary task is to produce a sophisticated analysis culminating in a "Scrybe Score" from -100 to +100, strictly following the protocol below.

        **Your Scoring Protocol (The 9 Layers of Analysis):**

        **1. Market Regime Context (Weight: 30%):** The `CURRENT_MARKET_REGIME` is your most important piece of context. You must heavily weigh this in your final score. A signal that contradicts the market regime (e.g., a BUY in a Bearish market) must be penalized significantly. Only an overwhelming confluence of other bullish factors (especially from Layers 2 and 7) can justify a contrarian signal.

        **2. Relative Strength Analysis (Weight: 20%):** Compare the `Stock_5D_Change` to the `Nifty50_5D_Change`. Exceptional relative strength in a bearish market is a powerful bullish signal, and vice-versa.
        
        **3. Sector Strength (Weight: 5%):** A stock in a strong sector gets a bonus. A stock in a weak sector gets a penalty.

        **4. Sentiment Analysis (Weight: 5%):** Bearish options data negatively impacts the score. Bullish sentiment has a positive impact. (Acknowledge if data is unavailable).

        **5. Fundamental Context (Weight: 10%):** A high 'Durability' score provides a strong safety net and a bonus for BUY signals.

        **6. Technical Deep-Dive (Weight: 15%):** Your core analysis of patterns and indicators (ADX, RSI, MACD, Volume Surge). A setup confirmed by a strong trend (ADX > {config.ADX_THRESHOLD}) gets a major score bonus.

        **7. Price Action Confirmation (Weight: 15%):** Analyze the `Confirmation_Candle` data. A strong technical setup (Layer 6) that is ALSO supported by strong confirmation from the last candle receives a significant score bonus. A setup with a weak or contradictory final candle must have its score penalized.
        
        **8. Confluence & Contradiction Check:** Explicitly identify the key points of confluence (factors that agree) and contradiction (factors that disagree) from the layers above.

        **9. Final Judgment & Justification:** Your final `scrybeScore` and `signal` must synthesize all layers.
            * Your `analystVerdict` must begin with your interpretation of the market regime, followed by the relative strength and price action confirmation.
            * Your `keyInsight` must be the single most important, actionable takeaway from your entire analysis, distilled into one powerful sentence.
            * **CONTRARIAN SIGNAL MANDATE:** If you issue a contrarian signal (a BUY in a Bearish market or SELL in a Bullish one), you MUST explicitly state the overwhelming factors from the other layers that justify this high-risk decision.
            * **CONFIRMATION MANDATE:** A high-conviction score (absolute value > 75) is ONLY PERMISSIBLE if there is strong price action confirmation from Layer 7. A perfect setup without a confirming final candle MUST be downgraded to a moderate-conviction signal.
            * **The `signal` and `confidence` MUST be derived logically from the `scrybeScore` based on the established protocol below:**
                * **+75 to +100:** High-conviction BUY
                * **+50 to +74:** Moderate-conviction BUY
                * **-49 to +49:** HOLD
                * **-50 to -74:** Moderate-conviction SELL
                * **-75 to -100:** High-conviction SELL
            * **The `isOnRadar` boolean MUST be `true` ONLY for 'HOLD' signals with scores between 40 to 49 and -40 to -49. For all other scores, it must be `false`.**
            * **CRITICAL FINAL RULE:** You are **NOT** responsible for the `tradePlan` object.
        """

        # The schema is correct and final.
        output_schema = {
            "type": "OBJECT", "properties": {
                "scrybeScore": {"type": "NUMBER"}, "signal": {"type": "STRING", "enum": ["BUY", "SELL", "HOLD"]},
                "confidence": {"type": "STRING", "enum": ["Low", "Medium", "High", "Very High"]}, "keyInsight": {"type": "STRING"},"analystVerdict": {"type": "STRING"},
                "keyObservations": {"type": "OBJECT", "properties": {
                    "confluencePoints": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "contradictionPoints": {"type": "ARRAY", "items": {"type": "STRING"}},
                }, "required": ["confluencePoints", "contradictionPoints"]},
                "reasonForHold": {"type": "STRING"}, "isOnRadar": {"type": "BOOLEAN"},
                "technicalBreakdown": { "type": "OBJECT", "properties": {
                    "Chart Pattern": {"type": "OBJECT", "properties": {"value": {"type": "STRING"}, "status": {"type": "STRING"}, "interpretation": {"type": "STRING"}}, "required": ["value", "status", "interpretation"]},
                    "Volatility Analysis": {"type": "OBJECT", "properties": {"character": {"type": "STRING"}, "implication": {"type": "STRING"}}, "required": ["character", "implication"]},
                    "ADX": {"type": "OBJECT", "properties": {"value": {"type": "STRING"}, "status": {"type": "STRING"}, "interpretation": {"type": "STRING"}}, "required": ["value", "status", "interpretation"]},
                    "RSI": {"type": "OBJECT", "properties": {"value": {"type": "STRING"}, "status": {"type": "STRING"}, "interpretation": {"type": "STRING"}}, "required": ["value", "status", "interpretation"]},
                    "MACD": {"type": "OBJECT", "properties": {"value": {"type": "STRING"}, "status": {"type": "STRING"}, "interpretation": {"type": "STRING"}}, "required": ["value", "status", "interpretation"]},
                }, "required": ["Chart Pattern", "Volatility Analysis", "ADX", "RSI", "MACD"]},
                "fundamentalBreakdown": { "type": "OBJECT", "properties": {
                    "Valuation": {"type": "OBJECT", "properties": {"value": {"type": "STRING"}, "status": {"type": "STRING"}, "interpretation": {"type": "STRING"}}, "required": ["value", "status", "interpretation"]},
                    "Profitability": {"type": "OBJECT", "properties": {"value": {"type": "STRING"}, "status": {"type": "STRING"}, "interpretation": {"type": "STRING"}}, "required": ["value", "status", "interpretation"]},
                    "Ownership": {"type": "OBJECT", "properties": {"value": {"type": "STRING"}, "status": {"type": "STRING"}, "interpretation": {"type": "STRING"}}, "required": ["value", "status", "interpretation"]}
                }, "required": ["Valuation", "Profitability", "Ownership"]},
                "tradePlan": {"type": "OBJECT", "properties": {
                    "timeframe": {"type": "STRING"}, "strategy": {"type": "STRING"},
                    "entryPrice": {"type": "OBJECT", "properties": {"price": {"type": "NUMBER"}, "rationale": {"type": "STRING"}}},
                    "target": {"type": "OBJECT", "properties": {"price": {"type": "NUMBER"}, "rationale": {"type": "STRING"}}},
                    "stopLoss": {"type": "OBJECT", "properties": {"price": {"type": "NUMBER"}, "rationale": {"type": "STRING"}}},
                    "riskRewardRatio": {"type": "NUMBER"}
                }},
            },
            "required": ["scrybeScore", "signal", "confidence", "keyInsight", "analystVerdict", "keyObservations", "technicalBreakdown", "fundamentalBreakdown"]
        }

        generation_config = genai.types.GenerationConfig(response_mime_type="application/json", response_schema=output_schema, max_output_tokens=16384)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # Create the model with the new safety settings included
        model = genai.GenerativeModel(
            model_name,
            safety_settings=safety_settings,
            system_instruction=definitive_scoring_prompt,
            generation_config=generation_config
        )
        fundamental_data_status = "Available" if live_financial_data.get('curatedData') else "Not Available / Sparse"
        prompt_parts = [
            "Please generate your complete, multi-layered analysis based on all the provided data and your Michelin-Star protocol. You do not need to calculate the tradePlan.",
            f"MARKET CONTEXT: {json.dumps(market_context)}",
            f"OPTIONS SENTIMENT: {json.dumps(options_data)}",
            f"CURRENT_VOLATILITY_ATR: {latest_atr:.2f}",
            f"Financial Data Snapshot (Status: {fundamental_data_status}): {json.dumps(live_financial_data['curatedData'])}",
            f"MACRO CONTEXT DATA: {json.dumps(macro_data)}",
            f"Key Technical Indicators: {json.dumps(technical_indicators)}"
        ]
        if charts:
            for key in sorted(charts.keys()):
                if charts[key]:
                    prompt_parts.append(f"This is the {key} chart:")
                    image_part = {"mime_type": "image/png", "data": base64.b64decode(charts[key])}
                    prompt_parts.append(image_part)
        max_retries = 4
        delay = 2
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt_parts, request_options={"timeout": 180})
                return json.loads(response.text)
            except Exception as e:
                if "429" in str(e) and "quota" in str(e).lower():
                    log.error("Quota exceeded. Raising exception to trigger key rotation.")
                    raise e
                log.warning(f"AI call attempt {attempt + 1} failed for {live_financial_data['rawDataSheet'].get('symbol', '')}. Error: {e}")
                if attempt < max_retries - 1:
                    log.info(f"Waiting for {delay} seconds before retrying...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    log.error(f"AI Scrybe Score generation failed after {max_retries} attempts.")
                    return None
                
    def get_mean_reversion_analysis(self, live_financial_data: dict, model_name: str, technical_indicators: dict, market_context: dict) -> dict:
        """
        Analyzes a stock for a mean-reversion opportunity.
        """
        log.info(f"Generating Mean-Reversion Analysis for {live_financial_data['rawDataSheet'].get('symbol', '')}...")

        mean_reversion_prompt = f"""
        You are "Scrybe-Oracle," a quantitative analyst specializing in **MEAN-REVERSION**. Your task is to identify stocks that are over-extended and poised for a profitable snap-back to their recent average.

        **Analysis Protocol:**
        1. **Context:** A mean-reversion trade is higher probability in a 'Neutral' or choppy market regime. You must consider the provided `CURRENT_MARKET_REGIME`.
        2. **The Setup:** Your primary goal is to find "stretched" stocks. Analyze the `technical_indicators` and charts for these key signals:
            * **RSI Extreme:** Is the RSI in a classic overbought (>70) for a SELL signal, or oversold (<30) for a BUY signal? This is your strongest signal.
            * **Bollinger Band Touch:** Is the price currently touching or exceeding the upper or lower Bollinger Band? This confirms the price is at a statistical extreme.
            * **Volume Spike:** Is there a high `Volume Surge`? A spike can signal "exhaustion" of the current trend, which is a perfect entry for a mean-reversion trade.

        **Final Scoring & JSON Output:**
        Synthesize your analysis into a `scrybeScore` from -100 to +100. The score must reflect your confidence in a profitable "snap-back."
        * A high positive score means strong conviction that an oversold stock will **rise (BUY)**.
        * A high negative score means strong conviction that an overbought stock will **fall (SELL)**.
        * **High-Conviction Mandate:** A high-conviction score (absolute value >= 75) is only justified if you see a clear RSI Extreme that is confirmed by either a Bollinger Band touch or a Volume Spike.
        * Your `analystVerdict` must clearly explain why the stock is a good mean-reversion candidate.
        """

        # This schema is simpler as it doesn't need the full 7-layer breakdown
        output_schema = {
            "type": "OBJECT", "properties": {
                "scrybeScore": {"type": "NUMBER"},
                "signal": {"type": "STRING", "enum": ["BUY", "SELL", "HOLD"]},
                "confidence": {"type": "STRING", "enum": ["Low", "Medium", "High", "Very High"]},
                "analystVerdict": {"type": "STRING"},
            }, "required": ["scrybeScore", "signal", "confidence", "analystVerdict"]
        }

        # Use the same robust safety settings
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        generation_config = genai.types.GenerationConfig(response_mime_type="application/json", response_schema=output_schema)
        model = genai.GenerativeModel(
            model_name,
            safety_settings=safety_settings,
            system_instruction=mean_reversion_prompt,
            generation_config=generation_config
        )

        prompt_parts = [
            "Please generate your mean-reversion analysis based on the provided data.",
            f"MARKET CONTEXT: {json.dumps(market_context)}",
            f"Key Technical Indicators: {json.dumps(technical_indicators)}"
        ]

        # This function uses a standard retry loop
        max_retries = 2 # Fewer retries for this simpler, secondary analysis
        delay = 2
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt_parts, request_options={"timeout": 120})
                return json.loads(response.text)
            except Exception as e:
                if "429" in str(e) and "quota" in str(e).lower():
                    log.error("Quota exceeded for Mean-Reversion. Raising exception to trigger key rotation.")
                    raise e # CRITICAL: Alert the main runner
                log.warning(f"Mean-Reversion AI call attempt {attempt + 1} failed. Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    log.error("Mean-Reversion analysis failed after all retries.")
                    return None # Return None on final failure

    def get_breakout_analysis(self, live_financial_data: dict, model_name: str, technical_indicators: dict, market_context: dict) -> dict:
        """
        Analyzes a stock for a volatility breakout opportunity.
        """
        log.info(f"Generating Breakout Analysis for {live_financial_data['rawDataSheet'].get('symbol', '')}...")

        breakout_prompt = f"""
        You are "Scrybe-Vulcan," a quantitative analyst specializing in **VOLATILITY BREAKOUT** strategies. Your sole purpose is to identify stocks that have been in a period of low volatility ("coiling") and are now breaking out with force.

        **PRIME BREAKOUT CHECKLIST:**
        You must evaluate the stock against the following four criteria. A high-conviction signal requires at least THREE of these to be met.

        1.  **Volatility Contraction (The Squeeze):** Is the stock in a state of "Quiet Consolidation"?
            * Analyze the `Bollinger Band Width Percent` and `ATR_Percent` from the technical indicators.
            * A true squeeze is indicated by exceptionally low values for both, suggesting energy is building up.

        2.  **The Breakout Candle (The Confirmation):** Did the most recent candle show a decisive move?
            * Analyze the `Confirmation_Candle` data.
            * A `position_in_range` close to 1.0 indicates a strong bullish breakout candle.
            * A `position_in_range` close to 0.0 indicates a strong bearish breakout candle.
            * A value near 0.5 is indecisive and fails this check.

        3.  **Volume Confirmation (The Fuel):** Was the breakout accompanied by a significant increase in volume?
            * Check the `Volume Surge` indicator. A "Yes" value strongly confirms the validity of the breakout.

        4.  **Trend Ignition (The Follow-Through):** Is a new trend potentially starting?
            * Analyze the `ADX` indicator. An ADX value rising above 20 suggests the breakout has enough strength to start a new trend. An ADX below 20 suggests the breakout might be a false move.

        **Final Scoring & JSON Output:**
        * Synthesize your checklist findings into a `scrybeScore` from -100 to +100.
        * A high positive score means strong conviction in a **bullish breakout (BUY)**.
        * A high negative score means strong conviction in a **bearish breakout (SELL)**.
        * Your `analystVerdict` MUST summarize which of the four checklist criteria were met to justify the score.
        """

        output_schema = {
            "type": "OBJECT", "properties": {
                "scrybeScore": {"type": "NUMBER"},
                "signal": {"type": "STRING", "enum": ["BUY", "SELL", "HOLD"]},
                "confidence": {"type": "STRING", "enum": ["Low", "Medium", "High", "Very High"]},
                "analystVerdict": {"type": "STRING"},
            }, "required": ["scrybeScore", "signal", "confidence", "analystVerdict"]
        }

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        generation_config = genai.types.GenerationConfig(response_mime_type="application/json", response_schema=output_schema)
        model = genai.GenerativeModel(
            model_name,
            safety_settings=safety_settings,
            system_instruction=breakout_prompt,
            generation_config=generation_config
        )

        prompt_parts = [
            "Please generate your volatility breakout analysis based on the provided data.",
            f"MARKET CONTEXT: {json.dumps(market_context)}",
            f"Key Technical Indicators: {json.dumps(technical_indicators)}"
        ]
        
        try:
            response = model.generate_content(prompt_parts, request_options={"timeout": 120})
            return json.loads(response.text)
        except Exception as e:
            if "429" in str(e) and "quota" in str(e).lower():
                log.error("Quota exceeded for Breakout. Raising exception to trigger key rotation.")
                raise e # CRITICAL: Alert the main runner
            log.error(f"Breakout AI call failed. Error: {e}")
            return None # Return None on failure
        
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

    def get_index_analysis(self, index_name: str, index_ticker: str) -> dict:
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
            
            macro_context = {"India GDP Growth (YoY)": "7.8%", "RBI Policy Stance": "Hawkish"}
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