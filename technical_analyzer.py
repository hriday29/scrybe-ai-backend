# technical_analyzer.py
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from logger_config import log
import data_retriever
import pandas_ta as ta
import angelone_retriever

def get_all_technicals(data: pd.DataFrame) -> dict:
    """
    Calculates a comprehensive set of technical indicators, including
    volatility proxies (ATR%, BBW%) and volume analysis.
    """
    log.info("Calculating comprehensive technical indicators...")
    if data is None or len(data) < 50: # Need enough data for rolling calcs
        return {"error": "Insufficient historical data for technical analysis."}

    data = data.copy()
    technicals = {} # Initialize empty dict

    # --- Calculate all indicators using pandas_ta ---
    # Ensure 'close' column exists and is numeric
    if 'close' not in data.columns or not pd.api.types.is_numeric_dtype(data['close']):
         log.error("Input data missing valid 'close' column.")
         return {"error": "Input data missing valid 'close' column."}

    data.ta.macd(append=True)
    data.ta.bbands(length=20, std=2, append=True) # Standard BBands
    data.ta.supertrend(append=True)
    data.ta.rsi(length=14, append=True)
    data.ta.adx(length=14, append=True)
    data.ta.atr(length=14, append=True) # Calculate standard ATR first
    data['volume_20d_avg'] = data['volume'].rolling(window=20).mean()

    # Drop rows with NaNs created by indicators
    data.dropna(inplace=True)

    if data.empty:
        return {"error": "Data became empty after calculating indicators and dropping NaNs."}

    latest_row = data.iloc[-1]
    previous_row = data.iloc[-2] if len(data) > 1 else latest_row # For histogram comparison

    # --- Build the technicals dictionary ---
    technicals["daily_close"] = latest_row.get("close", None)

    # Standard Indicators
    if "RSI_14" in latest_row: technicals["RSI_14"] = f"{latest_row['RSI_14']:.2f}"
    if "ADX_14" in latest_row: technicals["ADX_14_trend_strength"] = f"{latest_row['ADX_14']:.2f}"

    # --- Volatility Proxies (as percentages) ---
    if "ATRr_14" in latest_row and latest_row['close'] > 0:
        # pandas_ta calculates ATRp (percent) directly if column='close' is specified
        # Or calculate manually: (latest_row['ATR_14'] / latest_row['close']) * 100
        technicals["ATR_14_percent"] = f"{latest_row['ATRr_14']:.2f}%" # Assuming ATRr_14 is the percentage ATR
    else:
         technicals["ATR_14_percent"] = "N/A"

    if all(k in latest_row for k in ["BBU_20_2.0", "BBL_20_2.0", "BBM_20_2.0"]) and latest_row['BBM_20_2.0'] > 0:
        band_width_pct = ((latest_row['BBU_20_2.0'] - latest_row['BBL_20_2.0']) / latest_row['BBM_20_2.0']) * 100
        technicals["Bollinger_Band_Width_Percent"] = f"{band_width_pct:.2f}%"
        # Add interpretation for volatility qualifier later if needed
        technicals["bollinger_bands_interpretation"] = {
            "price_position": "Above Upper Band" if latest_row['close'] > latest_row['BBU_20_2.0'] else "Below Lower Band" if latest_row['close'] < latest_row['BBL_20_2.0'] else "Inside Bands",
            "volatility_state": "Squeeze (<5%)" if band_width_pct < 5.0 else ("Expansion (>15%)" if band_width_pct > 15 else "Normal")
         }
    else:
         technicals["Bollinger_Band_Width_Percent"] = "N/A"
         technicals["bollinger_bands_interpretation"] = {}


    # Volume Surge Analysis
    if "volume" in latest_row and "volume_20d_avg" in latest_row and latest_row['volume_20d_avg'] > 0:
        surge_ratio = latest_row['volume'] / latest_row['volume_20d_avg']
        technicals["volume_analysis"] = {
            "latest_volume": f"{latest_row['volume']:,.0f}",
            "20d_avg_volume": f"{latest_row['volume_20d_avg']:,.0f}",
            "surge_factor": f"{surge_ratio:.2f}x",
            "interpretation": "Significant Surge (>1.8x)" if surge_ratio > 1.8 else "Normal"
        }
    else:
         technicals["volume_analysis"] = {"interpretation": "N/A"}

    # Enhanced MACD Interpretation
    if all(k in latest_row for k in ["MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"]):
        status = "Neutral"
        macd_val = latest_row['MACD_12_26_9']
        signal_val = latest_row['MACDs_12_26_9']
        hist_val = latest_row['MACDh_12_26_9']
        prev_hist_val = previous_row['MACDh_12_26_9']

        if macd_val > signal_val: # MACD above signal
            status = "Bullish Crossover"
            if hist_val > prev_hist_val:
                status = "Bullish Momentum Accelerating"
            elif hist_val < prev_hist_val:
                 status = "Bullish Momentum Decelerating"
        elif macd_val < signal_val: # MACD below signal
            status = "Bearish Crossover"
            if hist_val < prev_hist_val:
                status = "Bearish Momentum Accelerating"
            elif hist_val > prev_hist_val:
                 status = "Bearish Momentum Decelerating"

        technicals["MACD_status"] = {
            "value": f"{macd_val:.2f}",
            "signal_line": f"{signal_val:.2f}",
            "histogram": f"{hist_val:.2f}",
            "interpretation": status
        }
    else:
        technicals["MACD_status"] = {"interpretation": "N/A"}

    # Supertrend
    if all(k in latest_row for k in ["SUPERT_7_3.0", "SUPERTd_7_3.0"]):
        technicals["supertrend_7_3"] = {
            "trend": "Uptrend" if latest_row['SUPERTd_7_3.0'] == 1 else "Downtrend",
            "value": f"{latest_row['SUPERT_7_3.0']:.2f}" # This is the stop level
        }
    else:
        technicals["supertrend_7_3"] = {"trend": "N/A"}

    log.info("Successfully calculated technical indicators.")
    return technicals

def get_relative_strength(stock_data: pd.DataFrame, full_nifty_data: pd.DataFrame) -> str:
    """
    Calculates the 5-day relative strength of a stock vs. the Nifty 50
    using a point-in-time slice to prevent lookahead bias.
    """
    try:
        # Ensure we have enough data to proceed
        if full_nifty_data is None or stock_data is None or len(stock_data) < 6:
            return "Data Not Available"

        # Get the current point-in-time from the stock's data
        point_in_time = stock_data.index[-1]

        # Create a slice of the Nifty data that ends at the same point in time
        nifty_slice = full_nifty_data.loc[:point_in_time].tail(6)
        
        if len(nifty_slice) < 6:
            return "Not enough Nifty data for comparison."

        # Now calculate the 5-day change for both
        nifty_5d_change = (nifty_slice['close'].iloc[-1] / nifty_slice['close'].iloc[0] - 1) * 100
        stock_5d_change = (stock_data['close'].iloc[-1] / stock_data['close'].iloc[-6] - 1) * 100
        
        if stock_5d_change > nifty_5d_change:
            return f"Outperforming ({stock_5d_change:.2f}% vs NIFTY's {nifty_5d_change:.2f}%)"
        else:
            return f"Underperforming ({stock_5d_change:.2f}% vs NIFTY's {nifty_5d_change:.2f}%)"
    except Exception as e:
        log.warning(f"Could not calculate relative strength: {e}")
        return "Data Not Available"

def _generate_intraday_chart(data: pd.DataFrame, ticker: str, title: str):
    """Generates a specialized 1-Day chart with Price and VWAP."""
    if data.empty or len(data) < 2:
        return None

    # Calculate VWAP
    data['VWAP'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
    
    fig, ax = plt.subplots(figsize=(15, 6), facecolor='#1C2130')
    fig.suptitle(title, fontsize=16, color='white')

    ax.set_facecolor('#1C2130')
    ax.plot(data.index, data['close'], label='Price', color='#3b82f6', linewidth=1.5)
    ax.plot(data.index, data['VWAP'], label='VWAP', color='#ffc107', linestyle='--', linewidth=2)
    
    ax.set_ylabel('Price', color='white', fontsize=10)
    ax.set_xlabel('Time', color='white', fontsize=10)
    ax.legend()
    ax.grid(True, color='#333A4C', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='y', labelcolor='white')
    ax.tick_params(axis='x', labelcolor='white', rotation=20)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), transparent=True)
    plt.close(fig)
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return chart_base64

def _generate_single_chart(data: pd.DataFrame, ticker: str, title: str):
    """
    An internal helper function to generate one chart for a specific timeframe.
    """
    if data.empty or len(data) < 2:
        return None

    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True, 
                             facecolor='#1C2130', 
                             gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    fig.suptitle(f'{title} for {ticker}', fontsize=16, color='white')

    plot_params = {'color': 'white', 'fontsize': 10}
    grid_params = {'color': '#333A4C', 'linestyle': '--', 'linewidth': 0.5}

    # Plot 1: Price and Bollinger Bands
    ax1 = axes[0]
    ax1.set_facecolor('#1C2130')
    ax1.plot(data.index, data['close'], label='Close Price', color='#3b82f6', linewidth=2)
    if all(k in data.columns for k in ['BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0']):
        ax1.plot(data.index, data['BBU_20_2.0'], label='Upper Band', color='gray', linestyle='--')
        ax1.plot(data.index, data['BBM_20_2.0'], label='Middle Band', color='#ffc107', linestyle='--')
        ax1.plot(data.index, data['BBL_20_2.0'], label='Lower Band', color='gray', linestyle='--')
        ax1.fill_between(data.index, data['BBL_20_2.0'], data['BBU_20_2.0'], color='#333A4C', alpha=0.5)
    ax1.set_ylabel('Price', **plot_params)
    ax1.legend()
    ax1.grid(True, **grid_params)
    ax1.tick_params(axis='y', labelcolor='white')

    # Plot 2: MACD
    ax2 = axes[1]
    ax2.set_facecolor('#1C2130')
    if all(k in data.columns for k in ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']):
        ax2.plot(data.index, data['MACD_12_26_9'], label='MACD', color='#3b82f6')
        ax2.plot(data.index, data['MACDs_12_26_9'], label='Signal', color='#ef4444')
        bar_colors = ['#22c55e' if val >= 0 else '#ef4444' for val in data['MACDh_12_26_9']]
        ax2.bar(data.index, data['MACDh_12_26_9'], color=bar_colors, alpha=0.5)
    ax2.set_ylabel('MACD', **plot_params)
    ax2.legend()
    ax2.grid(True, **grid_params)
    ax2.tick_params(axis='y', labelcolor='white')

    # Plot 3: RSI
    ax3 = axes[2]
    ax3.set_facecolor('#1C2130')
    if 'RSI_14' in data.columns:
        ax3.plot(data.index, data['RSI_14'], label='RSI', color='#a855f7')
        ax3.axhline(70, linestyle='--', color='#ef4444', alpha=0.7)
        ax3.axhline(30, linestyle='--', color='#22c55e', alpha=0.7)
    ax3.set_ylabel('RSI', **plot_params)
    ax3.legend()
    ax3.grid(True, **grid_params)
    ax3.tick_params(axis='y', labelcolor='white')

    # Plot 4: ADX
    ax4 = axes[3]
    ax4.set_facecolor('#1C2130')
    if 'ADX_14' in data.columns:
        ax4.plot(data.index, data['ADX_14'], label='ADX', color='#22c55e')
        ax4.axhline(25, linestyle='--', color='#3b82f6', alpha=0.7)
    ax4.set_ylabel('ADX', **plot_params)
    ax4.set_xlabel('Date', **plot_params)
    ax4.legend()
    ax4.grid(True, **grid_params)
    ax4.tick_params(axis='x', labelcolor='white', rotation=20)
    ax4.tick_params(axis='y', labelcolor='white')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), transparent=True)
    plt.close(fig)
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return chart_base64

def generate_focused_charts(full_data: pd.DataFrame, ticker: str) -> dict:
    """
    Generates a dictionary of charts for 3M, 1M, 1W, and 1D timeframes.
    """
    log.info(f"Generating multi-timescale charts for {ticker}...")
    charts = {}
    
    # Calculate indicators on the full dataset once
    full_data = full_data.copy()
    full_data.ta.bbands(append=True)
    full_data.ta.macd(append=True)
    full_data.ta.rsi(length=14, append=True)
    full_data.ta.adx(length=14, append=True)

    # --- Generate Daily Charts (3M, 1M, 1W) ---
    daily_timeframes = {"3M": 63, "1M": 21, "1W": 5}
    for key, days in daily_timeframes.items():
        data_slice = full_data.tail(days)
        charts[key] = _generate_single_chart(data_slice, ticker, f'{key} View')
        
    # --- Generate Intraday Chart (1D) ---
    try:
        intraday_data = data_retriever.get_intraday_data(ticker)
        if intraday_data is not None and not intraday_data.empty:
            latest_day = intraday_data.index.normalize().max()
            data_1d_slice = intraday_data[intraday_data.index.normalize() == latest_day].copy()
            charts["1D"] = _generate_intraday_chart(data_1d_slice, ticker, '1D Intraday View')
    except Exception as e:
        log.error(f"Failed to generate 1D intraday chart for {ticker}. Error: {e}")
        charts["1D"] = None
        
    log.info(f"Successfully generated charts for {ticker}.")
    return charts

def build_analysis_context(
    ticker: str,
    historical_data: pd.DataFrame,
    market_state: dict,
    is_backtest: bool = False,
    full_nifty_data: pd.DataFrame = None,   # For relative strength
    full_data_cache: dict = None            # For futures data lookup (optional)
) -> dict:
    """
    Builds a clean, structured context packet combining:
    - Technicals (including RS & F&O basis)
    - Fundamentals (point-in-time)
    - Volatility + Futures proxies (replaces sentiment in new architecture)

    Designed to feed AI analysts or analysis pipelines.
    """
    log.info(f"Building context for {ticker} (Backtest Mode: {is_backtest})...")

    # --- 1. Technical Data Packet ---
    technicals = get_all_technicals(historical_data)
    if "error" in technicals:
        log.error(f"Skipping context build for {ticker} due to technicals error: {technicals['error']}")
        return {
            "technical_indicators": {"status": "Error", "message": technicals['error']},
            "fundamental_data": {"status": "Not Processed"},
            "volatility_futures_data": {"status": "Not Processed"}
        }

    # Relative Strength vs Nifty
    technicals["relative_strength_vs_nifty50"] = get_relative_strength(historical_data, full_nifty_data)

    # --- 2. Futures Data & Basis Calculation ---
    futures_basis_pct = "N/A"
    futures_ticker = None

    if ".NS" in ticker:
        base_symbol = ticker.replace(".NS", "")
        potential_futures_ticker = f"{base_symbol}-F.NS"  # e.g., RELIANCE-F.NS
        log.info(f"Attempting to fetch futures data for {ticker} using {potential_futures_ticker}")

        if full_data_cache and potential_futures_ticker in full_data_cache:
            futures_data = full_data_cache[potential_futures_ticker]
            if futures_data is not None and not futures_data.empty:
                common_index = historical_data.index.intersection(futures_data.index)
                if not common_index.empty:
                    latest_spot = historical_data.loc[common_index[-1], "close"]
                    latest_future = futures_data.loc[common_index[-1], "close"]
                    if latest_spot > 0:
                        basis = latest_future - latest_spot
                        futures_basis_pct = f"{(basis / latest_spot) * 100:.2f}%"
                        futures_ticker = potential_futures_ticker
                        log.info(f"Calculated Futures Basis for {ticker}: {futures_basis_pct}")
                    else:
                        log.warning(f"Spot price zero for {ticker}, skipping basis calc.")
                else:
                    log.warning(f"No overlapping dates for {ticker} and its futures data.")
            else:
                log.info(f"No futures data found in cache for {potential_futures_ticker}.")
        else:
            log.info(f"Futures ticker {potential_futures_ticker} not in cache; skipping fetch for now.")

    technicals["futures_spot_basis_percent"] = futures_basis_pct
    if futures_ticker:
        technicals["futures_ticker_used"] = futures_ticker

    # --- 3. Fundamental Data Packet ---
    fundamentals = data_retriever.get_stored_fundamentals(ticker, point_in_time=historical_data.index[-1])
    if fundamentals and "asOfDate" in fundamentals and hasattr(fundamentals["asOfDate"], "isoformat"):
        fundamentals["asOfDate"] = fundamentals["asOfDate"].isoformat()
    if not fundamentals:
        fundamentals = {"status": "Fundamental data not available for this date."}

    # --- 4. Volatility & Futures Data Packet (Replaces Sentiment) ---
    volatility_futures_data = {
        "volatility_atr_percent": technicals.get("ATR_14_percent", "N/A"),
        "volatility_bbw_percent": technicals.get("Bollinger_Band_Width_Percent", "N/A"),
        "volatility_interpretation": technicals.get("bollinger_bands_interpretation", {}).get("volatility_state", "N/A"),
        "futures_spot_basis_percent": futures_basis_pct,
        "basis_interpretation": (
            "Premium (Bullish Bias)" if isinstance(futures_basis_pct, str) and futures_basis_pct != "N/A" and float(futures_basis_pct.replace("%", "")) > 0.1 else
            "Discount (Bearish Bias)" if isinstance(futures_basis_pct, str) and futures_basis_pct != "N/A" and float(futures_basis_pct.replace("%", "")) < -0.1 else
            "Near Flat / Unavailable"
        )
    }

    # --- 5. Assemble Final Context ---
    context = {
        "technical_indicators": technicals,
        "fundamental_data": fundamentals,
        "volatility_futures_data": volatility_futures_data  # replaces sentiment
    }

    log.info(f"Successfully built context for {ticker}.")
    return context