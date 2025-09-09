# utils.py
import itertools
import config
from logger_config import log

def setup_api_key_iterator():
    """Creates a cycling iterator over the available API keys."""
    key_pool = config.GEMINI_API_KEY_POOL
    if not key_pool:
        raise ValueError("API key pool is empty. Check your .env and config.py files.")
    log.info(f"API key rotator setup with {len(key_pool)} keys.")
    return itertools.cycle(key_pool)

def sanitize_context(context: dict) -> dict:
    """Sanitizes the context dictionary to replace null-like values."""
    sanitized_context = {}
    for layer, details in context.items():
        if isinstance(details, dict):
            sanitized_details = {}
            for k, v in details.items():
                if not v or str(v).strip().lower() in ["unavailable", "n/a", "none", "null", "unavailable in backtest"]:
                    sanitized_details[k] = "Data Not Available"
                else:
                    sanitized_details[k] = v
            sanitized_context[layer] = sanitized_details
        else:
            sanitized_context[layer] = details
    return sanitized_context

class APIKeyManager:
    """A simple class to manage and rotate a list of API keys."""

    def __init__(self, api_keys: list):
        if not api_keys:
            raise ValueError("API key list cannot be empty.")
        self.api_keys = api_keys
        self.current_index = 0
        log.info(f"APIKeyManager initialized with {len(self.api_keys)} keys.")

    def get_key(self) -> str:
        """Returns the current API key."""
        return self.api_keys[self.current_index]

    def rotate_key(self) -> str:
        """
        Rotates to the next API key in the list and returns it.
        Wraps around to the beginning if it reaches the end.
        """
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        new_key = self.get_key()
        log.warning(f"API key quota likely exceeded. Rotating to key #{self.current_index + 1}.")
        return new_key