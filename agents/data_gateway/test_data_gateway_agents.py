"""
Comprehensive test suite for Terra Constellata Data Gateway Agents
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import aiohttp

from .base_data_gateway_agent import DataGatewayAgent, DataGatewayAgentRegistry
from .secrets_manager import resolve_secret_placeholder


class TestDataGatewayAgent:
    """Unit tests for the base DataGatewayAgent class."""

    @pytest.fixture
    def mock_llm(self):
        """Mock language model for testing."""
        llm = Mock()
        llm.apredict = AsyncMock(return_value="Mock response")
        return llm

    @pytest.fixture
    def agent_config(self):
        """Sample agent configuration for testing."""
        return {
            "agent_name": "TEST_AGENT",
            "data_domain": "Test",
            "data_set_owner": {
                "name": "Test Organization",
                "ownerType": "Academic Institution",
                "officialContactUri": "https://test.org"
            },
            "capabilities": ["test_capability"],
            "base_url": "https://api.test.org",
            "authentication_methods": ["api_key"],
            "api_key": "test_key_123"
        }

    @pytest.fixture
    async def test_agent(self, mock_llm, agent_config):
        """Create a test agent instance."""
        agent = DataGatewayAgent(
            llm=mock_llm,
            tools=[],
            a2a_server_url="http://localhost:8080",
            **agent_config
        )
        yield agent
        # Cleanup
        if agent.session:
            await agent.session.close()

    @pytest.mark.asyncio
    async def test_agent_initialization(self, test_agent, agent_config):
        """Test agent initialization with correct properties."""
        assert test_agent.agent_name == agent_config["agent_name"]
        assert test_agent.data_domain == agent_config["data_domain"]
        assert test_agent.capabilities == agent_config["capabilities"]
        assert test_agent.base_url == agent_config["base_url"]
        assert test_agent.health_status == "initializing"

    @pytest.mark.asyncio
    async def test_a2a_connection(self, test_agent):
        """Test A2A protocol connection."""
        with patch.object(test_agent, 'connect_a2a', new_callable=AsyncMock) as mock_connect:
            await test_agent.connect_a2a()
            mock_connect.assert_called_once()
            assert test_agent.health_status == "connected"

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, test_agent):
        """Test health check when everything is working."""
        # Mock A2A client
        test_agent.a2a_client = Mock()

        # Mock API health check
        with patch.object(test_agent, '_check_external_api_health', new_callable=AsyncMock) as mock_api_check:
            mock_api_check.return_value = True

            health_data = await test_agent.health_check()

            assert health_data["status"] == "healthy"
            assert health_data["health_score"] == 1.0
            assert health_data["a2a_connected"] is True
            assert health_data["api_accessible"] is True
            assert "performance_metrics" in health_data

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, test_agent):
        """Test health check when services are down."""
        # No A2A client
        test_agent.a2a_client = None

        # Mock API health check failure
        with patch.object(test_agent, '_check_external_api_health', new_callable=AsyncMock) as mock_api_check:
            mock_api_check.return_value = False

            health_data = await test_agent.health_check()

            assert health_data["status"] == "unhealthy"
            assert health_data["health_score"] == 0.0
            assert health_data["a2a_connected"] is False
            assert health_data["api_accessible"] is False

    @pytest.mark.asyncio
    async def test_process_task_success(self, test_agent):
        """Test successful task processing."""
        with patch.object(test_agent, '_execute_capability', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"result": "success"}

            result = await test_agent.process_task("test_capability param=value")

            assert result == {"result": "success"}
            assert test_agent.request_count == 1
            assert test_agent.error_count == 0
            mock_execute.assert_called_once_with("test_capability", param="value")

    @pytest.mark.asyncio
    async def test_process_task_error(self, test_agent):
        """Test task processing with error."""
        with patch.object(test_agent, '_execute_capability', new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = Exception("API Error")

            with pytest.raises(Exception, match="API Error"):
                await test_agent.process_task("test_capability")

            assert test_agent.request_count == 1
            assert test_agent.error_count == 1
            assert test_agent.health_status == "error"

    @pytest.mark.asyncio
    async def test_api_request_success(self, test_agent):
        """Test successful API request."""
        # Initialize session
        await test_agent.connect_a2a()

        mock_response_data = {"data": "test"}

        with patch.object(test_agent.session, 'request') as mock_request:
            mock_response = AsyncMock()
            mock_response.__aenter__ = AsyncMock(return_value=Mock(
                status=200,
                raise_for_status=Mock(),
                json=AsyncMock(return_value=mock_response_data)
            ))
            mock_request.return_value = mock_response

            result = await test_agent._make_api_request("GET", "test/endpoint")

            assert result == mock_response_data
            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_api_request_retry(self, test_agent):
        """Test API request with retry on failure."""
        await test_agent.connect_a2a()

        with patch.object(test_agent.session, 'request') as mock_request:
            # First two calls fail, third succeeds
            mock_response_success = AsyncMock()
            mock_response_success.__aenter__ = AsyncMock(return_value=Mock(
                status=200,
                raise_for_status=Mock(),
                json=AsyncMock(return_value={"success": True})
            ))
            mock_response_success.__aexit__ = AsyncMock(return_value=None)

            mock_request.side_effect = [
                aiohttp.ClientError("Connection failed"),
                aiohttp.ClientError("Connection failed"),
                mock_response_success
            ]

            result = await test_agent._make_api_request("GET", "test/endpoint")

            assert result == {"success": True}
            assert mock_request.call_count == 3

    def test_get_agent_schema(self, test_agent):
        """Test agent schema generation."""
        schema = test_agent.get_agent_schema()

        assert schema["agentName"] == "TEST_AGENT"
        assert schema["dataDomain"] == "Test"
        assert schema["viaStatus"] == "AUTHENTICATED"
        assert len(schema["capabilities"]) == 1
        assert schema["capabilities"][0]["capabilityId"] == "test_capability"

    def test_performance_stats(self, test_agent):
        """Test performance statistics generation."""
        # Simulate some activity
        test_agent.request_count = 10
        test_agent.error_count = 1
        test_agent.response_times = [1.0, 2.0, 3.0]

        stats = test_agent.get_performance_stats()

        assert stats["total_requests"] == 10
        assert stats["error_count"] == 1
        assert stats["error_rate"] == 0.1
        assert stats["avg_response_time"] == 2.0
        assert stats["min_response_time"] == 1.0
        assert stats["max_response_time"] == 3.0


class TestSecretsManager:
    """Tests for secrets management functionality."""

    def test_resolve_secret_placeholder_resolved(self):
        """Test resolving a secret placeholder that exists."""
        with patch.dict('os.environ', {'TEST_SECRET': 'secret_value'}):
            result = resolve_secret_placeholder('{{SECRETS.TEST_SECRET}}')
            assert result == 'secret_value'

    def test_resolve_secret_placeholder_not_found(self):
        """Test resolving a secret placeholder that doesn't exist."""
        result = resolve_secret_placeholder('{{SECRETS.MISSING_SECRET}}')
        assert result == '{{SECRETS.MISSING_SECRET}}'  # Returns original if not found

    def test_resolve_secret_placeholder_no_placeholder(self):
        """Test resolving a value that's not a placeholder."""
        result = resolve_secret_placeholder('plain_value')
        assert result == 'plain_value'


class TestDataGatewayAgentRegistry:
    """Tests for the agent registry."""

    @pytest.fixture
    def registry(self):
        """Create a test registry."""
        return DataGatewayAgentRegistry()

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock()
        agent.agent_name = "TEST_AGENT"
        agent.data_domain = "Test"
        agent.capabilities = ["test_cap"]
        return agent

    def test_register_agent(self, registry, mock_agent):
        """Test agent registration."""
        registry.register_agent(mock_agent)
        assert registry.get_agent("TEST_AGENT") == mock_agent
        assert "TEST_AGENT" in registry.list_agents()

    def test_get_agents_by_domain(self, registry, mock_agent):
        """Test getting agents by domain."""
        registry.register_agent(mock_agent)
        agents = registry.get_agents_by_domain("Test")
        assert len(agents) == 1
        assert agents[0] == mock_agent

    def test_get_agents_by_capability(self, registry, mock_agent):
        """Test getting agents by capability."""
        registry.register_agent(mock_agent)
        agents = registry.get_agents_by_capability("test_cap")
        assert len(agents) == 1
        assert agents[0] == mock_agent

    @pytest.mark.asyncio
    async def test_broadcast_health_check(self, registry, mock_agent):
        """Test broadcasting health check to all agents."""
        mock_agent.health_check = AsyncMock(return_value={"status": "healthy"})
        registry.register_agent(mock_agent)

        results = await registry.broadcast_health_check()

        assert "TEST_AGENT" in results
        assert results["TEST_AGENT"]["status"] == "healthy"


class TestIntegration:
    """Integration tests for agent functionality."""

    @pytest.mark.asyncio
    async def test_agent_lifecycle(self):
        """Test complete agent lifecycle."""
        # This would be an integration test that starts a real agent
        # and tests its interaction with the A2A server
        pass

    @pytest.mark.asyncio
    async def test_inter_agent_communication(self):
        """Test communication between agents."""
        # Test A2A message passing between agents
        pass


# Mock API responses for testing
MOCK_API_RESPONSES = {
    "gebco": {
        "elevation": {"latitude": 40.0, "longitude": -74.0, "elevation": -100}
    },
    "nasa_landsat": {
        "images": [
            {"id": "LC08_L1GT_001002_20200101", "url": "https://landsat.example.com/image1.tif"}
        ]
    },
    "dpla": {
        "docs": [
            {"id": "123", "title": "Test Item", "description": "A test cultural item"}
        ]
    }
}


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for API testing."""
    with patch('aiohttp.ClientSession') as mock_session:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = Mock()
        mock_response.json = AsyncMock(return_value=MOCK_API_RESPONSES["gebco"])

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session.return_value.request.return_value = mock_context
        yield mock_session


if __name__ == "__main__":
    pytest.main([__file__, "-v"])