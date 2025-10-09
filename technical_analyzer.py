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
    Calculates a comprehensive set of technical indicators, now including volume analysis.
    """
    log.info("Calculating comprehensive technical indicators...")
    if data is None or len(data) < 50:
        return {"error": "Insufficient historical data for technical analysis."}

    data = data.copy()
    
    # --- Calculate all indicators ---
    data.ta.macd(append=True)
    data.ta.bbands(append=True)
    data.ta.supertrend(append=True)
    data.ta.rsi(length=14, append=True)
    data.ta.adx(length=14, append=True)
    data.ta.atr(length=14, append=True)
    data['volume_20d_avg'] = data['volume'].rolling(window=20).mean()
    
    latest_row = data.iloc[-1]
    
    # --- Build the technicals dictionary ---
    technicals = {"daily_close": latest_row.get("close", None)}
    
    # Standard Indicators
    if "RSI_14" in data.columns: technicals["RSI_14"] = f"{latest_row['RSI_14']:.2f}"
    if "ADX_14" in data.columns: technicals["ADX_14_trend_strength"] = f"{latest_row['ADX_14']:.2f}"
    if "ATRr_14" in data.columns: technicals["ATR_14_volatility"] = f"{latest_row['ATRr_14']:.2f}"
    
    # Volume Surge Analysis
    if "volume" in data.columns and "volume_20d_avg" in data.columns:
        surge_ratio = latest_row['volume'] / latest_row['volume_20d_avg'] if latest_row['volume_20d_avg'] > 0 else 0
        technicals["volume_analysis"] = {
            "latest_volume": f"{latest_row['volume']:,}",
            "20d_avg_volume": f"{latest_row['volume_20d_avg']:,}",
            "surge_factor": f"{surge_ratio:.2f}x",
            "interpretation": "Significant Volume Surge" if surge_ratio > 1.8 else "Normal Volume"
        }

    # Enhanced MACD Interpretation
    if all(k in data.columns for k in ["MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"]):
        status = "Neutral"
        if latest_row['MACD_12_26_9'] > latest_row['MACDs_12_26_9']:
            status = "Bullish Crossover"
            if latest_row['MACDh_12_26_9'] > data['MACDh_12_26_9'].iloc[-2]:
                status = "Bullish Momentum Accelerating"
        else:
            status = "Bearish Crossover"
            if latest_row['MACDh_12_26_9'] < data['MACDh_12_26_9'].iloc[-2]:
                status = "Bearish Momentum Accelerating"
        
        technicals["MACD_status"] = {
            "value": f"{latest_row['MACD_12_26_9']:.2f}",
            "signal_line": f"{latest_row['MACDs_12_26_9']:.2f}",
            "histogram": f"{latest_row['MACDh_12_26_9']:.2f}",
            "interpretation": status
        }
    
    # Bollinger Bands
    if all(k in data.columns for k in ["BBU_20_2.0", "BBL_20_2.0", "BBM_20_2.0"]):
        band_width = ((latest_row['BBU_20_2.0'] - latest_row['BBL_20_2.0']) / latest_row['BBM_20_2.0']) * 100
        technicals["bollinger_bands"] = {
            "price_position": "Above Upper Band" if latest_row['close'] > latest_row['BBU_20_2.0'] else "Below Lower Band" if latest_row['close'] < latest_row['BBL_20_2.0'] else "Inside Bands",
            "band_width_pct": f"{band_width:.2f}%",
            "interpretation": "Volatility Squeeze" if band_width < 5.0 else "Volatility Expansion"
        }
            
    # Supertrend
    if all(k in data.columns for k in ["SUPERT_7_3.0", "SUPERTd_7_3.0"]):
        technicals["supertrend_7_3"] = {
            "trend": "Uptrend" if latest_row['SUPERTd_7_3.0'] == 1 else "Downtrend",
            "value": f"{latest_row['SUPERT_7_3.0']:.2f}"
        }
            
    return technicals

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

def build_analysis_context(
    ticker: str, 
    historical_data: pd.DataFrame, 
    market_state: dict,
    is_backtest: bool = False
) -> dict:
    """
    Builds the ultimate, institutional-grade context packet for the AI.
    This version uses consistent, real data for both backtesting and live analysis.
    """
    log.info(f"Building INSTITUTIONAL-GRADE context for {ticker} (Backtest Mode: {is_backtest})...")

    # --- Fetch Core Data Components ---
    technicals = get_all_technicals(historical_data)
    
    # --- FIX: Fetch REAL fundamentals for both backtest and live ---
    # This ensures backtest results are representative of live performance.
    # Assumes `get_stored_fundamentals` can fetch point-in-time data.
    fundamentals = data_retriever.get_stored_fundamentals(ticker, historical_data.index[-1])
    if not fundamentals:
        fundamentals = {"error": "No fundamental data found for this date."}

    # --- Fetch Live-Only or Placeholder Data ---
    options_sentiment, news, market_depth = {}, {}, {}

    if is_backtest:
        # For backtesting, we use placeholders for data that cannot be reliably recreated.
        options_sentiment = {"status": "Unavailable in backtest"}
        news = {"status": "Unavailable in backtest"}
        market_depth = {"status": "Unavailable in backtest"}
    else:
        # For live runs, we fetch from our rich, real-time data sources.
        # --- FIX: Fetch REAL option greeks, not just placeholder data ---
        options_sentiment = angelone_retriever.get_option_greeks(ticker)
        if not options_sentiment:
            options_sentiment = {"status": "Option greeks data not available for this ticker."}
            
        news = data_retriever.get_news_articles_for_ticker(ticker)
        market_depth = data_retriever.get_live_market_depth(ticker)

    # --- Assemble the Final Context Dictionary ---
    context = {
        "ticker": ticker,
        "macro_context": {
            "overall_market_regime": market_state.get("market_regime", {}).get("regime_status"),
            "volatility_regime": market_state.get("volatility_regime", {}).get("volatility_status"),
            "market_breadth": market_state.get("market_breadth"),
            "global_benchmarks": market_state.get("benchmark_performance")
        },
        "sector_and_relative_strength": {
            "strong_sectors_of_day": market_state.get("strong_sectors"),
            "stock_vs_nifty50_rs": get_relative_strength(historical_data)
        },
        "fundamental_analysis": fundamentals,
        "technical_analysis": technicals,
        "market_internals": {
            "options_sentiment_greeks": options_sentiment,
            "live_order_book": market_depth
        },
        "news_and_events": news
    }
    return context