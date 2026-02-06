"""
Logging utility for the electoral inference project.
Provides structured logging with console and file handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from .config import get_config


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = None,
    level: str = None,
    log_file: str = None,
    console: bool = True,
    file_mode: str = 'a'
) -> logging.Logger:
    """
    Set up a logger with console and/or file handlers.

    Args:
        name: Logger name. If None, returns root logger.
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
               If None, reads from config.
        log_file: Path to log file. If None, reads from config.
        console: Whether to add console handler.
        file_mode: File mode for file handler ('a' for append, 'w' for write).

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logger('data_loader', level='DEBUG')
        >>> logger.info('Loading data...')
    """
    # Get configuration
    try:
        config = get_config()
        if level is None:
            level = config.get('logging.level', 'INFO')
        if log_file is None:
            log_file = config.get('logging.file')
        format_str = config.get(
            'logging.format',
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    except FileNotFoundError:
        # Fallback if config not available
        level = level or 'INFO'
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler with colors
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_formatter = ColoredFormatter(format_str)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode=file_mode, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(format_str)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get an existing logger or create a new one with default configuration.

    Args:
        name: Logger name. If None, returns root logger.

    Returns:
        Logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info('Processing data...')
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, set it up
    if not logger.handlers:
        setup_logger(name)

    return logger


class LoggerContext:
    """Context manager for temporary logging level changes."""

    def __init__(self, logger: logging.Logger, level: str):
        """
        Initialize context manager.

        Args:
            logger: Logger instance.
            level: Temporary logging level.

        Example:
            >>> logger = get_logger(__name__)
            >>> with LoggerContext(logger, 'DEBUG'):
            ...     logger.debug('This will be shown')
            >>> logger.debug('This will not be shown if level was INFO')
        """
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = None

    def __enter__(self):
        """Store old level and set new level."""
        self.old_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore old logging level."""
        self.logger.setLevel(self.old_level)
        return False


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls with arguments and return values.

    Args:
        logger: Logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> @log_function_call(logger)
        ... def process_data(data):
        ...     return len(data)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func_name} returned {result}")
                return result
            except Exception as e:
                logger.error(f"{func_name} raised {type(e).__name__}: {e}")
                raise
        return wrapper
    return decorator


# Module-level logger for this package
_module_logger = None


def get_module_logger() -> logging.Logger:
    """Get logger for the utils module."""
    global _module_logger
    if _module_logger is None:
        _module_logger = setup_logger('utils')
    return _module_logger
