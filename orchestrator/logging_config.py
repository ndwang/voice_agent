"""
Logging Configuration

Centralized logging configuration with time formatting.
"""
import logging
import sys


def setup_logging(level=logging.INFO, format_string=None):
    """
    Set up logging configuration with time formatting.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string. If None, uses default with time info.
    
    Returns:
        Logger instance
    """
    if format_string is None:
        # Default format with time info: YYYY-MM-DD HH:MM:SS,mmm - name - level - message
        format_string = '%(asctime)s,%(msecs)03d - %(name)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
    else:
        date_format = None
    
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt=date_format,
        stream=sys.stdout,
        force=True  # Override any existing configuration
    )
    
    return logging.getLogger(__name__)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

