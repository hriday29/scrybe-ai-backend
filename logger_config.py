"""
logger_config.py

Purpose
- Provide a single, standardized application logger configured for console output.

How it fits
- Imported by all modules to ensure consistent formatting and to avoid duplicate handler setup.

Main role
- Create and export `log`, an INFO-level logger with timestamped messages suitable for both
    local runs and container logs.
"""
import logging
import sys

def setup_logger():
    """Sets up a standardized logger for the application."""
    logger = logging.getLogger("ScrybeAI")
    logger.setLevel(logging.INFO)

    # Prevent adding duplicate handlers if this function is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
            '%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

# Create a single logger instance to be imported by other modules
log = setup_logger()