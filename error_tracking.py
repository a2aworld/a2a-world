"""
Error tracking and monitoring for Terra Constellata.
Integrates Sentry for comprehensive error reporting and monitoring.
"""

import os
import sentry_sdk

# Attempt to import FastAPI integration; skip if not available
try:
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    HAS_FASTAPI_INTEGRATION = True
except ImportError:
    FastApiIntegration = None
    HAS_FASTAPI_INTEGRATION = False

# Attempt to import SQLAlchemy integration; skip if not available
try:
    from sentry_sdk.integrations.sqlalchemy import SqlAlchemyIntegration
    HAS_SQLALCHEMY_INTEGRATION = True
except ImportError:
    SqlAlchemyIntegration = None
    HAS_SQLALCHEMY_INTEGRATION = False

from .logging_config import app_logger


def before_send(event, hint):
    """Filter and modify events before sending to Sentry."""
    # Don't send events in development for certain log levels
    if os.getenv("ENVIRONMENT") == "development":
        if event.get("level") == "info":
            return None

    # Add custom tags
    if "tags" not in event:
        event["tags"] = {}

    event["tags"]["service"] = "terra-constellata-backend"
    event["tags"]["component"] = "api"

    return event


def init_sentry():
    """Initialize Sentry error tracking."""
    sentry_dsn = os.getenv("SENTRY_DSN")
    environment = os.getenv("ENVIRONMENT", "development")

    if sentry_dsn:
        integrations = []
        if HAS_FASTAPI_INTEGRATION:
            integrations.append(FastApiIntegration())
        integrations.append(
            sentry_sdk.integrations.logging.LoggingIntegration(
                level=None,  # Capture all log levels
                event_level=None,  # Send all events to Sentry
            )
        )
        if HAS_SQLALCHEMY_INTEGRATION:
            integrations.append(SqlAlchemyIntegration())

        sentry_sdk.init(
            dsn=sentry_dsn,
            environment=environment,
            integrations=integrations,
            # Performance monitoring
            traces_sample_rate=1.0 if environment == "production" else 0.1,
            # Release tracking
            release=os.getenv("RELEASE_VERSION", "1.0.0"),
            # Error filtering
            before_send=before_send,
            # User feedback
            send_default_pii=True,
        )

        app_logger.info(
            "Sentry error tracking initialized",
            extra={
                "environment": environment,
                "release": os.getenv("RELEASE_VERSION", "1.0.0"),
            },
        )
    else:
        app_logger.warning("SENTRY_DSN not configured, error tracking disabled")


def capture_exception(exc, **kwargs):
    """Capture an exception with additional context."""
    with sentry_sdk.configure_scope() as scope:
        for key, value in kwargs.items():
            scope.set_tag(key, value)

        sentry_sdk.capture_exception(exc)


def capture_message(message, level="info", **kwargs):
    """Capture a message with additional context."""
    with sentry_sdk.configure_scope() as scope:
        for key, value in kwargs.items():
            scope.set_tag(key, value)

        sentry_sdk.capture_message(message, level=level)


def set_user_context(user_id=None, email=None, username=None):
    """Set user context for error tracking."""
    with sentry_sdk.configure_scope() as scope:
        user_dict = {}
        if user_id:
            user_dict["id"] = user_id
        if email:
            user_dict["email"] = email
        if username:
            user_dict["username"] = username
        if user_dict:
            scope.user = user_dict


def set_request_context(request_id=None, method=None, path=None):
    """Set request context for error tracking."""
    with sentry_sdk.configure_scope() as scope:
        if request_id is not None:
            scope.set_tag("request_id", request_id)
        if method is not None:
            scope.set_tag("method", method)
        if path is not None:
            scope.set_tag("path", path)


def track_performance(operation_name):
    """Track performance of operations."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            with sentry_sdk.start_transaction(op=operation_name, name=operation_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Initialize Sentry on module import
init_sentry()
