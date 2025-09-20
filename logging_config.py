"""
Centralized logging configuration for Terra Constellata project.
Provides structured logging with Loguru and JSON formatting for production.
"""

import sys
from pathlib import Path
from loguru import logger

# Project root
PROJECT_ROOT = Path(__file__).parent


def setup_logging(log_level: str = "INFO", log_to_file: bool = True):
    """Configure logging for the entire application."""

    # Remove default handler
    logger.remove()

    # Console handler with colored output for development
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File handler with JSON format for production
    if log_to_file:
        log_file = PROJECT_ROOT / "logs" / "terra_constellata.log"
        log_file.parent.mkdir(exist_ok=True)

        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            level=log_level,
            rotation="10 MB",
            retention="30 days",
            encoding="utf-8",
            serialize=False,  # False for custom string format; True for JSON
        )

        # Separate error log
        error_log = PROJECT_ROOT / "logs" / "errors.log"
        logger.add(
            error_log,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message} | {exception}",
            level="ERROR",
            rotation="10 MB",
            retention="30 days",
            encoding="utf-8",
            backtrace=True,
            diagnose=True,
        )

    # Add custom log levels for business logic
    logger.level("BUSINESS", no=25, color="<yellow>", icon="💼")
    logger.level("METRICS", no=26, color="<blue>", icon="📊")

    return logger


# Global logger instance
app_logger = setup_logging()


def get_logger(name: str):
    """Get a logger instance for a specific module."""
    return logger.bind(module=name)


# Structured logging helpers
def log_request(request_id: str, method: str, path: str, user_id: str = None):
    """Log API request with structured data."""
    logger.bind(request_id=request_id, method=method, path=path, user_id=user_id).info(
        "API Request"
    )


def log_response(request_id: str, status_code: int, response_time: float):
    """Log API response with structured data."""
    logger.bind(
        request_id=request_id, status_code=status_code, response_time=response_time
    ).info("API Response")


def log_error(
    error_type: str, error_message: str, traceback: str = None, user_id: str = None
):
    """Log error with structured data."""
    logger.bind(
        error_type=error_type,
        error_message=error_message,
        traceback=traceback,
        user_id=user_id,
    ).error("Application Error")


def log_business_event(event_type: str, event_data: dict, user_id: str = None):
    """Log business event with structured data."""
    logger.bind(event_type=event_type, event_data=event_data, user_id=user_id).log(
        "BUSINESS", f"Business Event: {event_type}"
    )


def log_metrics(metric_name: str, metric_value: float, tags: dict = None):
    """Log metrics with structured data."""
    logger.bind(
        metric_name=metric_name, metric_value=metric_value, tags=tags or {}
    ).log("METRICS", f"Metric: {metric_name} = {metric_value}")
