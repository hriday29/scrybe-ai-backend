# utils.py
import config
from logger_config import log

def sanitize_context(context: dict) -> dict:
    """Sanitizes the context dictionary to replace null-like values."""
    sanitized_context = {}
    for layer, details in context.items():
        if isinstance(details, dict):
            sanitized_details = {}
            for k, v in details.items():
                # This check correctly handles only null values,
                # preserving empty lists for the frontend.
                if v is None or str(v).strip().lower() in ["unavailable", "n/a", "none", "null", "unavailable in backtest"]:
                    sanitized_details[k] = "Data Not Available"
                else:
                    sanitized_details[k] = v
            sanitized_context[layer] = sanitized_details
        else:
            sanitized_context[layer] = details
    return sanitized_context
