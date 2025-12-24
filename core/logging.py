"""
Logging Configuration

Centralized logging configuration with time formatting.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logging(level=logging.INFO, format_string=None, log_file: Optional[str] = None, service_name: Optional[str] = None):
    """
    Set up logging configuration with time formatting.
    
    Args:
        level: Logging level (default: INFO). Can be a logging constant or string like "INFO", "DEBUG", etc.
        format_string: Custom format string. If None, uses default with time info.
        log_file: Optional path to log file. If None, defaults to 'logs/{service_name}.log' in project root.
                  If empty string, disables file logging.
        service_name: Optional service name used for default log file naming (e.g., "orchestrator", "stt", "tts").
    
    Returns:
        Logger instance
    """
    # Convert string log level to logging constant if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    if format_string is None:
        # Default format with time info: YYYY-MM-DD HH:MM:SS,mmm - name - level - message
        format_string = '%(asctime)s,%(msecs)03d - %(name)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
    else:
        date_format = None
    
    # Create formatter
    formatter = logging.Formatter(format_string, datefmt=date_format)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log_file is not empty string
    if log_file != "":
        if log_file is None:
            # Default to logs/{service_name}.log in project root, or logs/app.log if no service_name
            project_root = Path(__file__).parent.parent
            if service_name:
                log_file = project_root / "logs" / f"{service_name}.log"
            else:
                log_file = project_root / "logs" / "app.log"
        else:
            log_file = Path(log_file)
        
        # Create logs directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Use RotatingFileHandler to prevent log files from growing too large
        # Max 10MB per file, keep 5 backup files
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
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


