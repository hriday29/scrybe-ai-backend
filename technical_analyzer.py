# technical_analyzer.py (CORRECTED)
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from logger_config import log
import data_retriever
import pandas_ta as ta # Ensure pandas_ta is imported

def get_all_technicals(data: pd.DataFrame) -> dict:
    """
    Calculates a comprehensive set of technical indicators from historical data.
    This logic is ported from main_orchestrator.py to ensure parity.
    """
    log.info(f"Calculating technical indicators...")
    if data is None or len(data) < 50:
        return {"error": "Insufficient historical data for technical analysis."}

    # Use a copy to avoid modifying the original DataFrame
    data = data.copy()
    failed_indicators = []
    
    try:
        # --- Calculate all indicators first ---
        try:
            data.ta.macd(append=True)
        except Exception: failed_indicators.append("MACD")
        
        try:
            data.ta.bbands(append=True)
        except Exception: failed_indicators.append("Bollinger Bands")
        
        try:
            data.ta.supertrend(append=True)
        except Exception: failed_indicators.append("Supertrend")
            
        try:
            data.ta.rsi(length=14, append=True)
        except Exception: failed_indicators.append("RSI")

        try:
            data.ta.adx(length=14, append=True)
        except Exception: failed_indicators.append("ADX")
            
        try:
            data.ta.atr(length=14, append=True)
        except Exception: failed_indicators.append("ATR")

        latest_row = data.iloc[-1]
        
        # --- Build the technicals dictionary ---
        technicals = {"daily_close": latest_row.get("close", None)}
        
        if "RSI_14" in data.columns: technicals["RSI_14"] = f"{latest_row['RSI_14']:.2f}"
        if "ADX_14" in data.columns: technicals["ADX_14_trend_strength"] = f"{latest_row['ADX_14']:.2f}"
        if "ATRr_14" in data.columns: technicals["ATR_14"] = latest_row['ATRr_14']
        
        if all(k in data.columns for k in ["MACD_12_26_9", "MACDs_12_26_9"]):
            technicals["MACD_status"] = {
                "value": f"{latest_row['MACD_12_26_9']:.2f}",
                "signal_line": f"{latest_row['MACDs_12_26_9']:.2f}",
                "interpretation": "Bullish Crossover" if latest_row['MACD_12_26_9'] > latest_row['MACDs_12_26_9'] else "Bearish Crossover"
            }
        
        if all(k in data.columns for k in ["BBU_20_2.0", "BBL_20_2.0", "BBM_20_2.0"]):
            technicals["bollinger_bands"] = {
                "price_position": "Above Upper Band" if latest_row['close'] > latest_row['BBU_20_2.0'] else "Below Lower Band" if latest_row['close'] < latest_row['BBL_20_2.0'] else "Inside Bands",
                "upper_band": f"{latest_row['BBU_20_2.0']:.2f}",
                "lower_band": f"{latest_row['BBL_20_2.0']:.2f}",
                "band_width_pct": f"{((latest_row['BBU_20_2.0'] - latest_row['BBL_20_2.0']) / latest_row['BBM_20_2.0']) * 100:.2f}%"
            }
            
        if all(k in data.columns for k in ["SUPERT_7_3.0", "SUPERTd_7_3.0"]):
            technicals["supertrend_7_3"] = {
                "trend": "Uptrend" if latest_row['SUPERTd_7_3.0'] == 1 else "Downtrend",
                "value": f"{latest_row['SUPERT_7_3.0']:.2f}"
            }
            
        if failed_indicators:
            technicals["errors"] = f"Failed to calculate: {', '.join(failed_indicators)}"
            
        return technicals
        
    except Exception as e:
        log.error(f"Critical error during indicator calculation: {e}", exc_info=True)
        return {"error": f"Calculation failed. Missing indicators: {', '.join(failed_indicators)}"}

def get_relative_strength(stock_data: pd.DataFrame) -> str:
    """
    Calculates the 5-day relative strength of a stock vs. the Nifty 50.
    """
    try:
        nifty_data = data_retriever.get_historical_stock_data("^NSEI")
        if nifty_data is None or len(nifty_data) < 6 or stock_data is None or len(stock_data) < 6:
            return "Data Not Available"

        nifty_slice = nifty_data.loc[:stock_data.index[-1]]
        if len(nifty_slice) < 6: return "Data Not Available"

        nifty_5d_change = (nifty_slice['close'].iloc[-1] / nifty_slice['close'].iloc[-6] - 1) * 100
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

def build_live_context(ticker: str, historical_data: pd.DataFrame, market_regime_context: dict, sector_performance_context: dict) -> dict:
    """
    Constructs the complete, live context dictionary for the AI by mirroring
    the logic from the backtester's context builder.
    """
    technicals = get_all_technicals(historical_data)
    relative_strength = get_relative_strength(historical_data)
    
    # These calls fetch live, real-time data
    options_data = data_retriever.get_options_data(ticker)
    news = data_retriever.get_news_articles_for_ticker(ticker)
    
    context = {
        "ticker": ticker,
        "market_regime_analysis": market_regime_context,
        "sector_and_relative_strength": {
            "sector_performance": sector_performance_context,
            "relative_strength_vs_nifty50": relative_strength
        },
        "fundamental_proxy_analysis": data_retriever.get_fundamental_proxies(historical_data),
        "technical_analysis": technicals,
        "options_sentiment_analysis": options_data,
        "news_and_events_analysis": news
    }
    return context