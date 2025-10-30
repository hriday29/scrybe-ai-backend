#ai_analyzer.py
import time
import os
import config
import json
from logger_config import log
from typing import Optional, Any, Dict
from ai_providers import get_provider_from_env

class AIAnalyzer:
    """A class to handle all interactions with the selected AI provider (Azure OpenAI or Azure AI Foundry)."""

    def __init__(self, model_name: Optional[str] = None):
        # Initialize AI provider (factory validates required environment variables)
        self.provider = get_provider_from_env()
        # Preserve model_name for potential future use to avoid unused-arg warnings
        self.model_name = model_name
        
        # The model names in config.py should now correspond to either:
        # - Azure OpenAI deployment names (for AI_PROVIDER=azure-openai)
        # - Foundry model IDs (for AI_PROVIDER=azure-foundry, e.g., "grok-2")
        # Renamed to provider-agnostic terms: primary/secondary
        self.primary_model = config.PRIMARY_MODEL
        self.secondary_model = config.SECONDARY_MODEL

        log.info("AIAnalyzer initialized with dynamic provider selection")

    def _make_azure_call(self, system_instruction: str, user_prompt: str, deployment_name: str, output_schema: dict, timeout: int = 120, max_tokens: Optional[int] = None):
        """Helper function to make calls to Azure OpenAI, enforcing JSON output."""

        prompt_with_schema = f"""
        {user_prompt}

        Your response MUST be a JSON object that strictly adheres to the following schema. Do not include any other text or explanations outside of the JSON object.
        Schema:
        {json.dumps(output_schema, indent=2)}
        """

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_with_schema}
        ]

        try:
            response_text = self.provider.chat_completions(
                messages=messages,
                model=deployment_name,
                response_format={"type": "json_object"},
                timeout=timeout,
                max_tokens=max_tokens,
            )
            return json.loads(response_text)

        except json.JSONDecodeError as json_err:
            log.error(f"Azure API response was not valid JSON for deployment {deployment_name}. Response: {response_text[:500]}... Error: {json_err}")
            return {"error": f"Azure AI response parsing failed: {json_err}"}
        except Exception as e:
            # Catch other API errors, timeouts, etc.
            log.error(f"Azure API call failed for deployment {deployment_name}: {e}")
            # Consider checking error type for specific handling (e.g., rate limits, content filters)
            error_message = f"Failed to generate response from Azure AI: {e}"
            # Extract more specific error details if available (depends on exception type)
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                error_message += f" - API Response: {e.response.text}"
            return {"error": error_message}

    def get_technical_verdict(self, technical_data: dict, ticker: str) -> dict:
        """Analyzes ONLY technical data to form a directional bias."""
        log.info(f"Getting technical verdict for {ticker}...")
        
        system_instruction = """
        You are a Chartered Market Technician (CMT). Your sole focus is price action, volume, and technical indicators.
        Based ONLY on the data provided, determine the most probable 5-10 day directional bias.
        State your confidence and identify the primary technical pattern or reason for your conclusion.
        Ignore all fundamental or news-based context. Your analysis must be purely chart-based.
        """
        
        output_schema = {
            "type": "OBJECT", "properties": {
                "bias": {"type": "STRING", "enum": ["Bullish", "Bearish", "Neutral"]},
                "confidence": {"type": "STRING", "enum": ["Low", "Medium", "High"]},
                "primary_pattern": {"type": "STRING", "description": "e.g., 'Bull Flag Breakout', 'RSI Divergence', '200DMA Rejection'"},
                "rationale": {"type": "STRING", "description": "A brief, one-sentence justification based only on the technicals."}
            }, "required": ["bias", "confidence", "primary_pattern", "rationale"]
        }

        prompt = f"Provide a pure technical analysis for {ticker} using the following data:\n{json.dumps(technical_data, indent=2)}"

        return self._make_azure_call(
            system_instruction=system_instruction,
            user_prompt=prompt,
            deployment_name=self.secondary_model,
            output_schema=output_schema
        )

    def get_fundamental_verdict(self, fundamental_data: dict, ticker: str) -> dict:
        """Analyzes ONLY fundamental data to assess financial health."""
        log.info(f"Getting fundamental verdict for {ticker}...")
        
        system_instruction = """
        You are a CFA charterholder and fundamental analyst. Your task is to assess a company's financial health and growth trajectory based purely on the provided financial data.
        Ignore the stock's price chart and any market sentiment. Your conclusion must be grounded in financial metrics like revenue growth, profitability, and shareholder patterns.
        """
        
        output_schema = {
            "type": "OBJECT", "properties": {
                "health": {"type": "STRING", "enum": ["Strong", "Average", "Weak"]},
                "growth_trajectory": {"type": "STRING", "enum": ["Accelerating", "Stable", "Decelerating"]},
                "valuation": {"type": "STRING", "enum": ["Attractive", "Fair", "Stretched"]},
                "rationale": {"type": "STRING", "description": "A brief, one-sentence justification based only on the fundamentals."}
            }, "required": ["health", "growth_trajectory", "valuation", "rationale"]
        }

        prompt = f"Provide a pure fundamental analysis for {ticker} using the following data:\n{json.dumps(fundamental_data, indent=2)}"
        
        return self._make_azure_call(
            system_instruction=system_instruction,
            user_prompt=prompt,
            deployment_name=self.secondary_model,
            output_schema=output_schema
        )

    # def get_sentiment_verdict(self, sentiment_data: dict, ticker: str) -> dict:
    #     """Analyzes ONLY sentiment data (news, options) to gauge market emotion."""
    #     log.info(f"Getting sentiment verdict for {ticker}...")
        
    #     system_instruction = """
    #     You are a market sentiment analyst. Your job is to gauge the prevailing emotion and potential catalysts from news and options market data.
    #     Determine if the current mood is driven by fear, greed, or neutrality. Identify the most significant piece of information driving this sentiment.
    #     Ignore the company's financials and the long-term technical chart.
    #     """
        
    #     output_schema = {
    #         "type": "OBJECT", "properties": {
    #             "prevailing_emotion": {"type": "STRING", "enum": ["Greed", "Fear", "Neutral"]},
    #             "key_catalyst": {"type": "STRING", "description": "e.g., 'High call option volume', 'Negative earnings pre-announcement'"},
    #             "strength": {"type": "STRING", "enum": ["Low", "Medium", "High"]},
    #             "rationale": {"type": "STRING", "description": "A brief, one-sentence justification based only on sentiment indicators."}
    #         }, "required": ["prevailing_emotion", "key_catalyst", "strength", "rationale"]
    #     }

    #     prompt = f"Provide a pure sentiment analysis for {ticker} using the following data:\n{json.dumps(sentiment_data, indent=2)}"

    #     return self._make_azure_call(
    #         system_instruction=system_instruction,
    #         user_prompt=prompt,
    #         deployment_name=self.secondary_model,
    #         output_schema=output_schema
    #     )

    def get_volatility_and_futures_verdict(self, vol_futures_data: dict, ticker: str) -> dict:
        """
        Analyzes volatility proxies (ATR%, BBW%) and Futures-Spot Basis %
        to gauge market conditions relevant to the specific stock.
        Replaces the old sentiment analysis.
        """
        log.info(f"Getting Volatility & Futures verdict for {ticker}...")

        system_instruction = """
        You are a quantitative derivatives analyst. Your focus is ONLY on the provided volatility metrics (ATR%, Bollinger Band Width %) and the Futures-Spot Basis %.
        Based ONLY on this data, assess the current volatility environment and any directional bias suggested by the futures market premium or discount.
        Ignore all chart patterns, fundamentals, or news. Your analysis must be purely based on the provided derivatives and volatility proxies.
        Determine:
        1. Volatility State: Is volatility contracting (Squeeze), expanding, or normal?
        2. Basis Bias: Does the futures basis suggest bullish (premium), bearish (discount), or neutral sentiment?
        3. Confidence: How strong is the signal from these indicators combined?
        4. Rationale: Briefly justify your conclusions based *only* on the provided data.
        """

        output_schema = {
            "type": "OBJECT", "properties": {
                "volatility_state": {"type": "STRING", "enum": ["Squeeze", "Expansion", "Normal", "N/A"]},
                "basis_bias": {"type": "STRING", "enum": ["Bullish Premium", "Bearish Discount", "Neutral / Flat", "N/A"]},
                "confidence": {"type": "STRING", "enum": ["Low", "Medium", "High"]},
                "rationale": {"type": "STRING", "description": "Brief justification based ONLY on volatility/basis data."}
            }, "required": ["volatility_state", "basis_bias", "confidence", "rationale"]
        }

        # Ensure the input data is passed correctly as a JSON string in the prompt
        prompt = f"Provide a pure volatility and futures basis analysis for {ticker} using the following data:\n{json.dumps(vol_futures_data, indent=2)}"

        return self._make_azure_call(
            system_instruction=system_instruction,
            user_prompt=prompt,
            deployment_name=self.secondary_model, # Use the secondary (typically faster) model for this focused task
            output_schema=output_schema
        )
        
    def get_apex_analysis(self, ticker: str, technical_verdict: dict, fundamental_verdict: dict, volatility_futures_verdict: dict, market_state: dict, screener_reason: str) -> dict: # <-- MODIFIED PARAMETER
        """
        Synthesizes expert verdicts (Technical, Fundamental, Volatility/Futures)
        into a final, institutional-grade trade decision.
        """
        log.info(f"Generating APEX Synthesis for {ticker} using new Volatility/Futures input...")

        # --- MODIFIED SYSTEM PROMPT ---
        system_instruction = """
        You are "Scrybe," the Head of Strategy for a top-tier hedge fund. You have just received reports from your specialist analyst team: a technical analyst (CMT), a fundamental analyst (CFA), and a **Volatility & Derivatives strategist**. # <-- UPDATED ANALYST TYPE

        Your task is to synthesize these expert, and sometimes conflicting, reports into a single, decisive trade recommendation. Your decision is governed by the iron-clad Fund Mandate.

        **FUND MANDATE & DECISION HIERARCHY:**
        1.  **Market Regime is KING:** Your primary goal is to trade in alignment with the `market_regime`. Counter-trend trades are forbidden unless ALL three specialist reports (Technical, Fundamental, Volatility/Futures) show high-confidence agreement against the market regime.
        2.  **Confluence Builds Conviction:** Your `confidence` level should reflect the degree of alignment. If technical and fundamental verdicts align with the market regime, confidence is 'High'. If only one aligns, it's 'Medium'.
        3.  **Volatility/Futures is a Modifier:** Use the Volatility/Futures report primarily to adjust confidence or veto trades in extreme conditions.
            * **Volatility:** A 'Squeeze' might precede a breakout, potentially increasing confidence if other factors align. High 'Expansion' might warrant *reducing* position size or skipping the trade due to risk, even if the direction is right.
            * **Basis:** A strong 'Bullish Premium' can boost confidence in a BUY signal. A strong 'Bearish Discount' can boost confidence in a SHORT signal. A conflicting basis (e.g., Bullish Premium on a SHORT signal) should decrease confidence or potentially veto the trade.

        **CRITICAL SCORING RULE:** You MUST use the sign of the scrybeScore to indicate direction.
        - Positive (+) scores are ONLY for BUY signals.
        - Negative (-) scores are ONLY for SHORT signals.
        - The magnitude (0-100) represents your conviction. A 'High' confidence SHORT must have a large NEGATIVE score (e.g., -70 or lower).
        - Scores between -20 and +20 will be considered a HOLD.

        Your role is to determine the qualitative direction (Signal), the conviction (Score & Confidence), and the justification (Thesis). You will NOT determine price targets, stop losses, or risk/reward ratios; the execution team will calculate those based on your high-level guidance.

        Synthesize the reports, weigh the evidence according to the mandate, and generate the final trade plan in the required JSON format.
        """
        # --- END MODIFIED SYSTEM PROMPT ---

        # Output schema remains the same
        output_schema = {
            "type": "OBJECT",
            "properties": {
                "scrybeScore": {"type": "NUMBER"},
                "signal": {"type": "STRING", "enum": ["BUY", "SHORT", "HOLD"]},
                "thesisType": {"type": "STRING"},
                "confidence": {"type": "STRING", "enum": ["Low", "Medium", "High", "Very High"]},
                "keyInsight": {"type": "STRING"},
                "analystVerdict": {"type": "STRING"},
                "keyRisks_and_Mitigants": {
                    "type": "OBJECT",
                    "properties": { "risk_1": {"type": "STRING"}, "risk_2": {"type": "STRING"}, "mitigant": {"type": "STRING"} },
                    "required": ["risk_1", "risk_2", "mitigant"]
                },
                "keyObservations": {
                    "type": "OBJECT",
                    "properties": { "confluencePoints": {"type": "ARRAY", "items": {"type": "STRING"}}, "contradictionPoints": {"type": "ARRAY", "items": {"type": "STRING"}} },
                    "required": ["confluencePoints", "contradictionPoints"]
                }
            },
            "required": [
                "scrybeScore", "signal", "thesisType", "confidence",
                "keyInsight", "analystVerdict", "keyRisks_and_Mitigants", "keyObservations"
            ]
        }

        # --- MODIFIED PROMPT CONTENT ---
        prompt_content = "\n".join([
            "## Analyst Reports & Market State ##",
            f"Ticker: {ticker}",
            f"Screener Reason: {screener_reason}",
            f"Market State: {json.dumps(market_state, indent=2)}",
            f"Technical Verdict: {json.dumps(technical_verdict, indent=2)}",
            f"Fundamental Verdict: {json.dumps(fundamental_verdict, indent=2)}",
            f"Volatility/Futures Verdict: {json.dumps(volatility_futures_verdict, indent=2)}", # <-- MODIFIED LINE
            "\nSynthesize these reports into the final trade decision as per the Fund Mandate."
        ])
        # --- END MODIFIED PROMPT CONTENT ---

        log.info(f"--- Attempting APEX synthesis with model '[{self.primary_model}]' ---")
        try:
            analysis_result = self._make_azure_call(
                system_instruction=system_instruction,
                user_prompt=prompt_content,
                deployment_name=self.primary_model, # Use primary model for the final synthesis
                output_schema=output_schema,
                timeout=180,
                max_tokens=8192 # Keep max tokens high for complex synthesis
            )

            if "error" in analysis_result:
                raise Exception(analysis_result["error"])

            analysis_result["model_used"] = self.primary_model
            log.info(f"✅ Success on APEX synthesis for {ticker}.")
            return analysis_result
        except Exception as e:
            log.critical(f"CRITICAL: APEX synthesis failed for {ticker}. Error: {e}. Returning a default HOLD signal.")
            # Return a default "error" or "HOLD" structure that matches the schema
            return {
                "scrybeScore": 0,
                "signal": "HOLD",
                "thesisType": "Error",
                "confidence": "Low",
                "keyInsight": f"APEX analysis failed due to an internal error: {e}",
                "analystVerdict": "Inconclusive due to analysis failure.",
                "keyRisks_and_Mitigants": {
                    "risk_1": "Analysis engine failure.",
                    "risk_2": "No data available.",
                    "mitigant": "Defaulting to HOLD. No action taken."
                },
                "keyObservations": {
                    "confluencePoints": [],
                    "contradictionPoints": ["APEX synthesis failed to run."]
                },
                "model_used": self.primary_model,
                "error": str(e)  # Add an explicit error field
            }

    def get_single_news_impact_analysis(self, article: dict) -> Optional[Dict[str, Any]]:
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

        result = self._make_azure_call(
            system_instruction=system_instruction,
            user_prompt=prompt,
            deployment_name=self.secondary_model,
            output_schema=output_schema
        )
        return result if "error" not in result else None

    
    def get_intraday_short_signal(self, prompt_data: dict) -> Optional[Dict[str, Any]]:
        """
    Analyzes holistic data to find high-probability intraday short candidates
    using the primary model.
        """
        ticker = prompt_data.get("ticker", "N/A")
        log.info(f"[AI] Getting intraday short signal for {ticker} with secondary model...")

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

        result = self._make_azure_call(
            system_instruction=system_instruction,
            user_prompt=prompt,
            deployment_name=self.secondary_model,
            output_schema=output_schema
        )
        return result if "error" not in result else None

    def get_index_analysis(
        self,
        index_name: str,
        macro_context: dict,
        latest_technicals: dict,
        vix_value: Optional[str],
        options_data: dict
    ) -> Optional[Dict[str, Any]]:
        """
        Generates a CIO-grade, in-depth analysis for a market index,
        including a fallback for deriving key levels from technicals
        if options data is unavailable.
        """
        log.info(f"Generating definitive CIO-grade index analysis for {index_name}...")

        system_instruction = """
        You are the Chief Investment Officer (CIO) of a global macro fund. Your task is to synthesize technical, macroeconomic, and (when available) options market data into a clear, institutional-grade strategic report on a major stock market index.

        **Multi-Factor Synthesis Protocol:**
        1. **Technical Health Assessment:** Analyze the provided technical indicator data (Price vs. MAs, RSI) to determine the current trend strength and momentum.
        2. **Key Levels Analysis (CRITICAL):**
            - If options data (Max OI) is provided, you MUST use the high OI strike prices as the primary psychological support/resistance levels.
            - **If options data is "Not Available", you MUST derive the Key Support and Resistance levels from the technical data provided (e.g., recent swing highs/lows, significant moving averages).** The report must contain key levels.
        3. **Correlation & Context Check:**
            - Analyze the stock's correlation to the NIFTY 50. Is it moving with the market (high positive correlation) or against it?
            - A 'BUY' signal in a stock that is strongly correlated with a bearish NIFTY 50 requires extra caution and a pristine setup. Acknowledge this context in your verdict.
        4. **Sentiment Analysis:** Use the Put-Call Ratio (PCR) and Volatility Index (VIX) to gauge current market sentiment.
        5. **Macroeconomic Overlay:** Interpret how the macro data influences the index's trajectory.
        6. **Final Synthesis:** Combine all factors to produce a cohesive report, completing all fields in the required JSON format with detailed, well-reasoned insights.
        """

        output_schema = {
            "type": "OBJECT",
            "properties": {
                "marketPulse": {
                    "type": "OBJECT",
                    "properties": {
                        "overallBias": {"type": "STRING"},
                        "volatilityIndexStatus": {"type": "STRING"}
                    },
                    "required": ["overallBias", "volatilityIndexStatus"]
                },
                "trendAnalysis": {
                    "type": "OBJECT",
                    "properties": {
                        "shortTermTrend": {"type": "STRING"},
                        "mediumTermTrend": {"type": "STRING"},
                        "keyTrendIndicators": {"type": "STRING"}
                    },
                    "required": ["shortTermTrend", "mediumTermTrend", "keyTrendIndicators"]
                },
                "keyLevels": {
                    "type": "OBJECT",
                    "properties": {
                        "resistance": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "support": {"type": "ARRAY", "items": {"type": "STRING"}}
                    },
                    "required": ["resistance", "support"]
                },
                "optionsMatrix": {
                    "type": "OBJECT",
                    "properties": {
                        "maxOpenInterestCall": {"type": "STRING"},
                        "maxOpenInterestPut": {"type": "STRING"},
                        "putCallRatioAnalysis": {"type": "STRING"}
                    },
                    "required": ["maxOpenInterestCall", "maxOpenInterestPut", "putCallRatioAnalysis"]
                },
                "forwardOutlook": {
                    "type": "OBJECT",
                    "properties": {
                        "next7Days": {"type": "STRING"},
                        "primaryRisk": {"type": "STRING"}
                    },
                    "required": ["next7Days", "primaryRisk"]
                }
            },
            "required": ["marketPulse", "trendAnalysis", "keyLevels", "optionsMatrix", "forwardOutlook"]
        }

        # Convert inputs to strings for the prompt
        vix_value_str = vix_value if vix_value else "Not Available"
        options_data_str = json.dumps(options_data) if options_data else "Not Available"
        technicals_str = json.dumps(latest_technicals) if latest_technicals else "Not Available"

        prompt = f"""
        Generate a CIO-level strategic report for the {index_name}. Synthesize all of the following data:
        - Latest Technicals: {technicals_str}
        - Latest Volatility Index (India VIX): {vix_value_str}
        - Latest Options Data: {options_data_str}
        - Macroeconomic Context: {json.dumps(macro_context)}
        Provide your full analysis in the required JSON format.
        If options data is not available, you must still provide key support and resistance levels based on technicals.
        """

        result = self._make_azure_call(
            system_instruction=system_instruction,
            user_prompt=prompt,
            deployment_name=self.secondary_model,
            output_schema=output_schema
        )
        
        return result if "error" not in result else None


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
            
    def get_dvm_scores(self, live_financial_data: dict, technical_indicators: dict) -> Optional[Dict[str, Any]]:
        """A method to generate Durability, Valuation, and Momentum scores."""
        log.info("Generating DVM scores...")
        
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
        prompt = "\n".join([
            "Generate the DVM scores based on this data:",
            f"Financial Data: {json.dumps(live_financial_data['curatedData'])}",
            f"Technical Indicators: {json.dumps(technical_indicators)}"
        ])

        # Simplified retry logic for Azure
        max_retries = 3
        delay = 2
        for attempt in range(max_retries):
            try:
                result = self._make_azure_call(
                    system_instruction=system_instruction,
                    user_prompt=prompt,
                    deployment_name=self.secondary_model,
                    output_schema=output_schema
                )
                if "error" not in result:
                    return result
                
                log.warning(f"[DVM Scoring] Attempt {attempt + 1} failed with API error: {result['error']}")
            except Exception as e:
                 log.warning(f"[DVM Scoring] Attempt {attempt + 1} failed with system error: {e}")

            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
        
        log.error(f"[DVM Scoring] Failed after {max_retries} attempts.")
        return None
    
    def get_conversational_answer(self, question: str, analysis_context: dict) -> Optional[Dict[str, Any]]:
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
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
        
        try:
            answer_text = self.provider.chat_completions(
                model=self.secondary_model,
                messages=messages,
                temperature=0,
                timeout=120,
            )
            return {"answer": answer_text}
        except Exception as e:
            log.error(f"Conversational Q&A call failed. Error: {e}")
            return None
