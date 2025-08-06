# utils.py
from logger_config import log

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