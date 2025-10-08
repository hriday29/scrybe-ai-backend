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
    return iter(key_pool)

def sanitize_context(context: dict) -> dict:
    """Sanitizes the context dictionary to replace null-like values."""
    sanitized_context = {}
    for layer, details in context.items():
        if isinstance(details, dict):
            sanitized_details = {}
            for k, v in details.items():
                # --- THIS IS THE FIX ---
                # The old check 'if not v' incorrectly converted empty lists [].
                # This new check 'if v is None' correctly handles only null values,
                # preserving empty lists for the frontend.
                if v is None or str(v).strip().lower() in ["unavailable", "n/a", "none", "null", "unavailable in backtest"]:
                    sanitized_details[k] = "Data Not Available"
                else:
                    sanitized_details[k] = v
            sanitized_context[layer] = sanitized_details
        else:
            sanitized_context[layer] = details
    return sanitized_context

class APIKeyManager:
    """
    A robust class to manage and rotate a pool of API keys.
    It deactivates keys that fail (e.g., due to rate limits) for the current run.
    """

    def __init__(self, api_keys: list):
        if not api_keys:
            raise ValueError("API key list cannot be empty.")
        # Use list.copy() to avoid modifying the original config list
        self.active_keys = api_keys.copy()
        self.deactivated_keys = []
        log.info(f"APIKeyManager initialized with {len(self.active_keys)} active keys.")

    def get_key(self) -> str | None:
        """
        Returns the current active API key. Returns None if all keys are exhausted.
        """
        if not self.active_keys:
            log.error("FATAL: All API keys have been exhausted.")
            return None
        return self.active_keys[0]

    def deactivate_current_key_and_get_next(self) -> str | None:
        """
        Moves the current key to the deactivated pool and returns the next available key.
        Returns None if no more active keys are available.
        """
        if not self.active_keys:
            log.error("Attempted to deactivate a key, but the active pool is already empty.")
            return None

        # Deactivate the current key (which is always the first one)
        failing_key = self.active_keys.pop(0)
        self.deactivated_keys.append(failing_key)
        log.warning(f"Deactivating an API key. {len(self.active_keys)} keys remaining.")
        
        # Return the new first key, or None if the pool is now empty
        return self.get_key()
