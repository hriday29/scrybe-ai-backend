# technical_analyzer.py
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from logger_config import log

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
    ax1.plot(data.index, data['BBU_20_2.0'], label='Upper Band', color='gray', linestyle='--')
    ax1.plot(data.index, data['BBM_20_2.0'], label='Middle Band', color='#ffc107', linestyle='--')
    ax1.plot(data.index, data['BBL_20_2.0'], label='Lower Band', color='gray', linestyle='--')
    ax1.fill_between(data.index, data['BBL_20_2.0'], data['BBU_20_2.0'], color='#333A4C', alpha=0.5)
    ax1.set_ylabel('Price', **plot_params)
    ax1.legend()
    ax1.grid(True, **grid_params)
    ax1.tick_params(axis='y', labelcolor='white')

    # Other plots (MACD, RSI, ADX) follow the same logic...
    # Plot 2: MACD
    ax2 = axes[1]
    ax2.set_facecolor('#1C2130')
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
    Generates a dictionary of charts for 3M, 1M, and 1W timeframes.
    """
    log.info(f"Generating multi-timescale charts for {ticker}...")
    charts = {}
    
    # Define timeframes using approximate trading days
    timeframes = {
        "3M": 63,
        "1M": 21,
        "1W": 5
    }
    
    for key, days in timeframes.items():
        # Slice the data for the specific timeframe
        data_slice = full_data.tail(days)
        charts[key] = _generate_single_chart(data_slice, ticker, f'{key} View')
        
    log.info(f"Successfully generated {len(charts)} charts for {ticker}.")
    return charts