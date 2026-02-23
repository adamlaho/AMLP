# multi_agent_dft/utils/logging.py
import logging
import sys
from pathlib import Path
from typing import Optional

# Configure logging only once
_logger_configured = False

def configure_logging() -> None:
    """
    Configure the logging system based on the configuration settings.
    """
    global _logger_configured
    
    if _logger_configured:
        return
    
    # Set default logging configuration
    log_level = "INFO"
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console_logging = True
    
    # Convert string log level to actual level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler if enabled
    if console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    _logger_configured = True
    
    # Log configuration completion
    logger = logging.getLogger(__name__)
    logger.debug("Logging system configured")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically the module name)
        
    Returns:
        A configured logger instance
    """
    configure_logging()
    return logging.getLogger(name)