# utils.py
import config
from logger_config import log

def sanitize_context(context):
    """
    Recursively sanitizes a dictionary or list to replace null-like values
    with a standard 'Data Not Available' string.
    """
    if isinstance(context, dict):
        # If it's a dictionary, recurse on its values
        return {k: sanitize_context(v) for k, v in context.items()}
    
    elif isinstance(context, list):
        # If it's a list, recurse on its elements
        return [sanitize_context(elem) for elem in context]
        
    elif context is None or str(context).strip().lower() in ["unavailable", "n/a", "none", "null", "unavailable in backtest"]:
        # This is the base case: a value that needs to be replaced
        return "Data Not Available"
        
    else:
        # This is a valid primitive value (int, str, float, bool), return as is
        return context