"""
A2A Protocol Message Validation

This module provides validation for incoming JSON-RPC messages and A2A protocol
messages, ensuring they conform to the expected schemas and business rules.
"""

import json
import logging
from typing import Any, Dict, Optional, Union
from datetime import datetime, timezone
from pydantic import ValidationError

from schemas import (
    A2AMessage,
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCError,
    JSONRPCErrorResponse,
    JSONRPCNotification,
    MESSAGE_TYPES,
    get_message_class,
)

logger = logging.getLogger(__name__)


class MessageValidator:
    """Validator for A2A protocol messages"""

    @staticmethod
    def validate_jsonrpc_message(
        raw_message: str,
    ) -> Union[JSONRPCRequest, JSONRPCNotification, JSONRPCErrorResponse]:
        """
        Validate and parse a raw JSON-RPC message.

        Args:
            raw_message: Raw JSON string

        Returns:
            Parsed JSON-RPC message object

        Raises:
            ValueError: If message is invalid
        """
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            return MessageValidator._create_error_response(
                -32700, "Parse error", str(e), None
            )

        # Check for required jsonrpc field
        if not isinstance(data, dict) or data.get("jsonrpc") != "2.0":
            return MessageValidator._create_error_response(
                -32600, "Invalid Request", "Invalid JSON-RPC version", None
            )

        # Determine message type
        if "method" not in data:
            return MessageValidator._create_error_response(
                -32600, "Invalid Request", "Missing method field", None
            )

        method = data["method"]
        params = data.get("params", {})
        msg_id = data.get("id")

        # Check if it's a notification (no id)
        if msg_id is None:
            try:
                return JSONRPCNotification(method=method, params=params)
            except ValidationError as e:
                return MessageValidator._create_error_response(
                    -32602, "Invalid params", str(e), None
                )

        # It's a request
        try:
            return JSONRPCRequest(method=method, params=params, id=msg_id)
        except ValidationError as e:
            return MessageValidator._create_error_response(
                -32602, "Invalid params", str(e), msg_id
            )

    @staticmethod
    def validate_a2a_message(
        method: str, params: Dict[str, Any]
    ) -> Optional[A2AMessage]:
        """
        Validate A2A message parameters against the expected schema.

        Args:
            method: JSON-RPC method name
            params: Method parameters

        Returns:
            Validated A2A message object or None if invalid
        """
        message_class = get_message_class(method)
        if not message_class:
            logger.warning(f"Unknown message type: {method}")
            return None

        try:
            message = message_class(**params)
            logger.info(f"Validated message: {method} from {message.sender_agent}")
            return message
        except ValidationError as e:
            logger.error(f"Message validation failed for {method}: {e}")
            return None

    @staticmethod
    def validate_business_rules(message: A2AMessage) -> bool:
        """
        Validate business rules for A2A messages.

        Args:
            message: The A2A message to validate

        Returns:
            True if valid, False otherwise
        """
        # Example business rules
        if hasattr(message, "confidence") and message.confidence < 0.1:
            logger.warning(f"Low confidence message: {message.confidence}")
            return False

        if hasattr(message, "rating") and message.rating is not None:
            if not (1 <= message.rating <= 5):
                logger.error(f"Invalid rating: {message.rating}")
                return False

        # Check timestamp is not in future
        if message.timestamp > datetime.now(timezone.utc):
            logger.warning("Message timestamp is in the future")
            return False

        return True

    @staticmethod
    def _create_error_response(
        code: int, message: str, data: Any, msg_id: Any
    ) -> JSONRPCErrorResponse:
        """Create a JSON-RPC error response"""
        error = JSONRPCError(code=code, message=message, data=data)
        return JSONRPCErrorResponse(error=error, id=msg_id)

    @staticmethod
    def create_success_response(
        result: Any, msg_id: Union[str, int]
    ) -> JSONRPCResponse:
        """Create a JSON-RPC success response"""
        return JSONRPCResponse(result=result, id=msg_id)
