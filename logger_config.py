# logger_config.py
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