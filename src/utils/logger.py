"""
Logging configuration for Medical AI Assistant.
Provides structured logging with different levels for development and production.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
            
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
            
        return json.dumps(log_data)


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    structured: bool = False
) -> logging.Logger:
    """
    Setup and configure logger.
    
    Args:
        name: Logger name (typically __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging output
        structured: If True, use JSON structured logging
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if structured:
        console_formatter = StructuredFormatter()
    else:
        console_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        if structured:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
