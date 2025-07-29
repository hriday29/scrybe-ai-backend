import pandas as pd
from logger_config import log

def calculate_historical_performance(historical_data: pd.DataFrame) -> dict:
    """
    Calculates the stock's price performance over standard trading day periods.
    """
    if historical_data.empty:
        return {}

    log.info("Calculating historical performance snapshot...")
    performance = {}
    # Use standard trading days instead of calendar days
    periods = {'1-Week': 5, '1-Month': 21, '3-Month': 63, '6-Month': 126, '1-Year': 252}
    latest_price = historical_data['close'].iloc[-1]
    
    for name, days in periods.items():
        if len(historical_data) > days:
            old_price = historical_data['close'].iloc[-days]
            change_pct = ((latest_price - old_price) / old_price) * 100
            performance[name] = {'change_percent': f"{change_pct:.2f}%"}
        else:
            performance[name] = {'change_percent': 'N/A'}
    
    log.info("Successfully calculated performance snapshot.")
    return performance

def calculate_correlations(stock_historical_data: pd.DataFrame, benchmarks_data: pd.DataFrame) -> dict:
    """
    Calculates the 90-day correlation between a stock's returns and the returns
    of key benchmarks.
    """
    log.info("Calculating inter-market correlations...")
    if stock_historical_data is None or benchmarks_data is None or stock_historical_data.empty or benchmarks_data.empty:
        return {}
        
    try:
        # Using pct_change() for returns is the more robust method
        stock_returns = stock_historical_data['close'].pct_change().rename('Stock')
        benchmark_returns = benchmarks_data.pct_change()
        
        combined_returns = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
        
        if len(combined_returns) < 90:
            log.warning("Not enough data for 90-day correlation.")
            return {}

        correlation_matrix = combined_returns.tail(90).corr()
        stock_correlations = correlation_matrix['Stock'].drop('Stock')

        formatted_correlations = {
            f"{k} Correlation": f"{v:.2f}" for k, v in stock_correlations.items()
        }
        log.info(f"Successfully calculated {len(formatted_correlations)} correlations.")
        return formatted_correlations
        
    except Exception as e:
        log.error(f"An error occurred during correlation calculation: {e}")
        return {}