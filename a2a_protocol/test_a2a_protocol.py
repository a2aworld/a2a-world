"""
Unit Tests for A2A Protocol Components

This module contains comprehensive unit tests for all A2A protocol components.
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from schemas import (
    A2AMessage,
    GeospatialAnomalyIdentified,
    InspirationRequest,
    CreationFeedback,
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCNotification,
)
from validation import MessageValidator
from client import A2AClient
from server import A2AServer
from extensibility import MessageTypeRegistry


class TestSchemas:
    """Test message schemas"""

    def test_geospatial_anomaly_creation(self):
        """Test creating a geospatial anomaly message"""
        anomaly = GeospatialAnomalyIdentified(
            sender_agent="test_agent",
            anomaly_type="cultural_pattern",
            location={"lat": 40.7128, "lon": -74.0060},
            confidence=0.85,
            description="Test anomaly",
            data_source="test_source",
        )

        assert anomaly.sender_agent == "test_agent"
        assert anomaly.anomaly_type == "cultural_pattern"
        assert anomaly.location["lat"] == 40.7128
        assert anomaly.confidence == 0.85

    def test_inspiration_request_creation(self):
        """Test creating an inspiration request message"""
        request = InspirationRequest(
            sender_agent="test_agent",
            context="Create a story",
            domain="mythology",
            constraints=["must include heroes", "no modern elements"],
        )

        assert request.context == "Create a story"
        assert request.domain == "mythology"
        assert len(request.constraints) == 2

    def test_creation_feedback_creation(self):
        """Test creating creation feedback message"""
        feedback = CreationFeedback(
            sender_agent="test_agent",
            original_request_id="req_123",
            feedback_type="positive",
            content="Great work!",
            rating=5,
        )

        assert feedback.feedback_type == "positive"
        assert feedback.rating == 5

    def test_jsonrpc_request_creation(self):
        """Test JSON-RPC request creation"""
        request = JSONRPCRequest(
            method="GEOSPATIAL_ANOMALY_IDENTIFIED", params={"test": "data"}, id="req_1"
        )

        assert request.jsonrpc == "2.0"
        assert request.method == "GEOSPATIAL_ANOMALY_IDENTIFIED"
        assert request.id == "req_1"

    def test_jsonrpc_notification_creation(self):
        """Test JSON-RPC notification creation"""
        notification = JSONRPCNotification(
            method="GEOSPATIAL_ANOMALY_IDENTIFIED", params={"test": "data"}
        )

        assert notification.jsonrpc == "2.0"
        assert notification.method == "GEOSPATIAL_ANOMALY_IDENTIFIED"
        assert not hasattr(notification, "id")


class TestMessageValidator:
    """Test message validation"""

    def test_validate_valid_jsonrpc_message(self):
        """Test validating a valid JSON-RPC message"""
        message = {
            "jsonrpc": "2.0",
            "method": "GEOSPATIAL_ANOMALY_IDENTIFIED",
            "params": {
                "sender_agent": "test_agent",
                "anomaly_type": "test",
                "location": {"lat": 0, "lon": 0},
                "confidence": 0.8,
                "description": "test",
                "data_source": "test",
            },
            "id": "test_1",
        }

        raw_message = json.dumps(message)
        result = MessageValidator.validate_jsonrpc_message(raw_message)

        assert isinstance(result, JSONRPCRequest)
        assert result.method == "GEOSPATIAL_ANOMALY_IDENTIFIED"

    def test_validate_invalid_json(self):
        """Test validating invalid JSON"""
        result = MessageValidator.validate_jsonrpc_message("invalid json")

        assert hasattr(result, "error")
        assert result.error.code == -32700

    def test_validate_invalid_jsonrpc_version(self):
        """Test validating message with wrong JSON-RPC version"""
        message = {"jsonrpc": "1.0", "method": "test", "id": "test_1"}

        raw_message = json.dumps(message)
        result = MessageValidator.validate_jsonrpc_message(raw_message)

        assert hasattr(result, "error")
        assert result.error.code == -32600

    def test_validate_a2a_message_valid(self):
        """Test validating a valid A2A message"""
        params = {
            "sender_agent": "test_agent",
            "anomaly_type": "test",
            "location": {"lat": 0, "lon": 0},
            "confidence": 0.8,
            "description": "test",
            "data_source": "test",
        }

        result = MessageValidator.validate_a2a_message(
            "GEOSPATIAL_ANOMALY_IDENTIFIED", params
        )

        assert isinstance(result, GeospatialAnomalyIdentified)
        assert result.sender_agent == "test_agent"

    def test_validate_a2a_message_invalid(self):
        """Test validating an invalid A2A message"""
        params = {"sender_agent": "test_agent", "invalid_field": "test"}

        result = MessageValidator.validate_a2a_message(
            "GEOSPATIAL_ANOMALY_IDENTIFIED", params
        )

        assert result is None

    def test_validate_business_rules_valid(self):
        """Test validating business rules for valid message"""
        message = GeospatialAnomalyIdentified(
            sender_agent="test_agent",
            anomaly_type="test",
            location={"lat": 0, "lon": 0},
            confidence=0.8,
            description="test",
            data_source="test",
        )

        result = MessageValidator.validate_business_rules(message)

        assert result is True

    def test_validate_business_rules_low_confidence(self):
        """Test validating business rules for low confidence message"""
        message = GeospatialAnomalyIdentified(
            sender_agent="test_agent",
            anomaly_type="test",
            location={"lat": 0, "lon": 0},
            confidence=0.05,  # Too low
            description="test",
            data_source="test",
        )

        result = MessageValidator.validate_business_rules(message)

        assert result is False


class TestA2AClient:
    """Test A2A client"""

    @pytest.fixture
    async def client(self):
        """Create test client"""
        client = A2AClient("http://test-server:8080", "test_agent")
        yield client
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_send_request_success(self, client):
        """Test sending a successful request"""
        # Mock the session
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "jsonrpc": "2.0",
                "result": {"status": "success"},
                "id": "test_1",
            }
        )

        with patch.object(client, "_send_request", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = mock_response.json.return_value

            anomaly = GeospatialAnomalyIdentified(
                sender_agent="test_agent",
                anomaly_type="test",
                location={"lat": 0, "lon": 0},
                confidence=0.8,
                description="test",
                data_source="test",
            )

            result = await client.send_request("GEOSPATIAL_ANOMALY_IDENTIFIED", anomaly)

            assert result["status"] == "success"
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_request_error(self, client):
        """Test sending a request that returns an error"""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": "Method not found"},
                "id": "test_1",
            }
        )

        with patch.object(client, "_send_request", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = mock_response.json.return_value

            anomaly = GeospatialAnomalyIdentified(
                sender_agent="test_agent",
                anomaly_type="test",
                location={"lat": 0, "lon": 0},
                confidence=0.8,
                description="test",
                data_source="test",
            )

            with pytest.raises(RuntimeError, match="Method not found"):
                await client.send_request("UNKNOWN_METHOD", anomaly)

    @pytest.mark.asyncio
    async def test_send_notification(self, client):
        """Test sending a notification"""
        with patch.object(client, "_send_request", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = None

            anomaly = GeospatialAnomalyIdentified(
                sender_agent="test_agent",
                anomaly_type="test",
                location={"lat": 0, "lon": 0},
                confidence=0.8,
                description="test",
                data_source="test",
            )

            await client.send_notification("GEOSPATIAL_ANOMALY_IDENTIFIED", anomaly)

            mock_send.assert_called_once()


class TestA2AServer:
    """Test A2A server"""

    @pytest.fixture
    def server(self):
        """Create test server"""
        server = A2AServer("localhost", 8080)
        return server

    def test_register_method(self, server):
        """Test registering a method"""

        async def test_handler(message):
            return {"result": "test"}

        server.register_method("TEST_METHOD", test_handler)

        assert "TEST_METHOD" in server.methods

    @pytest.mark.asyncio
    async def test_handle_health(self, server):
        """Test health check endpoint"""
        from aiohttp.test_utils import make_mocked_request

        request = make_mocked_request("GET", "/health")
        response = await server._handle_health(request)

        assert response.status == 200
        # Note: In real scenario, would check response body


class TestMessageTypeRegistry:
    """Test message type registry"""

    @pytest.fixture
    def registry(self):
        """Create test registry"""
        return MessageTypeRegistry()

    def test_register_message_type(self, registry):
        """Test registering a message type"""
        from pydantic import BaseModel

        class TestMessage(A2AMessage):
            test_field: str

        registry.register_message_type("TEST_MESSAGE", TestMessage)

        assert "TEST_MESSAGE" in registry._message_types
        assert registry.get_message_type("TEST_MESSAGE") == TestMessage

    def test_create_message(self, registry):
        """Test creating a message instance"""
        message = registry.create_message(
            "GEOSPATIAL_ANOMALY_IDENTIFIED",
            sender_agent="test",
            anomaly_type="test",
            location={"lat": 0, "lon": 0},
            confidence=0.8,
            description="test",
            data_source="test",
        )

        assert isinstance(message, GeospatialAnomalyIdentified)
        assert message.sender_agent == "test"

    def test_create_unknown_message_type(self, registry):
        """Test creating message with unknown type"""
        with pytest.raises(ValueError, match="Unknown message type"):
            registry.create_message("UNKNOWN_TYPE")

    def test_register_handler(self, registry):
        """Test registering a handler"""

        async def test_handler(message):
            return {"result": "test"}

        registry.register_handler("TEST_METHOD", test_handler)

        assert registry.get_handler("TEST_METHOD") == test_handler


# Integration test
@pytest.mark.asyncio
async def test_client_server_integration():
    """Test client-server integration"""
    # This would require starting a real server
    # For now, just test the setup
    server = A2AServer("localhost", 8081)

    async def test_handler(message):
        return {"processed": True, "message_id": message.message_id}

    server.register_method("GEOSPATIAL_ANOMALY_IDENTIFIED", test_handler)

    # In a real test, we would start the server and test with client
    # For this example, we just verify the setup
    assert "GEOSPATIAL_ANOMALY_IDENTIFIED" in server.methods


if __name__ == "__main__":
    pytest.main([__file__])
