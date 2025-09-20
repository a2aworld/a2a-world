"""
Tests for A2A Protocol Integration
"""

import unittest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from ..manager import ProvingGroundManager
from ...a2a_protocol.schemas import CertificationRequest
from ...a2a_protocol.client import A2AClient


class TestA2AIntegration(unittest.TestCase):
    """Test cases for A2A protocol integration"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock LLM and tools
        self.mock_llm = Mock()
        self.mock_tools = [Mock()]

        # Create manager
        self.manager = ProvingGroundManager(
            name="TestProvingGroundManager",
            llm=self.mock_llm,
            tools=self.mock_tools,
            a2a_server_url="http://localhost:8080",
        )

    @patch("..manager.A2AClient")
    async def test_a2a_connection(self, mock_a2a_client):
        """Test A2A client connection"""
        mock_client_instance = Mock()
        mock_client_instance.connect = AsyncMock(return_value=True)
        mock_client_instance.disconnect = AsyncMock()
        mock_a2a_client.return_value = mock_client_instance

        # Connect to A2A
        await self.manager.connect_a2a()

        self.assertIsNotNone(self.manager.a2a_client)
        mock_client_instance.connect.assert_called_once()

        # Disconnect from A2A
        await self.manager.disconnect_a2a()

        self.assertIsNone(self.manager.a2a_client)
        mock_client_instance.disconnect.assert_called_once()

    async def test_send_message(self):
        """Test sending A2A messages"""
        # Mock A2A client
        mock_client = Mock()
        mock_client.send_request = AsyncMock(return_value={"status": "success"})
        self.manager.a2a_client = mock_client

        # Create test message
        message = CertificationRequest(
            sender_agent="test_agent",
            subject="test_subject",
            certification_type="agent_certification",
            evidence={"test": "data"},
            criteria=["communication"],
        )

        # Send message
        result = await self.manager.send_message(message, "target_agent")

        self.assertIsNotNone(result)
        mock_client.send_request.assert_called_once()

    async def test_send_notification(self):
        """Test sending A2A notifications"""
        # Mock A2A client
        mock_client = Mock()
        mock_client.send_notification = AsyncMock()
        self.manager.a2a_client = mock_client

        # Create test message
        message = CertificationRequest(
            sender_agent="test_agent",
            subject="test_subject",
            certification_type="agent_certification",
            evidence={"test": "data"},
            criteria=["communication"],
        )

        # Send notification
        await self.manager.send_notification(message, "target_agent")

        mock_client.send_notification.assert_called_once()

    async def test_request_inspiration(self):
        """Test requesting inspiration via A2A"""
        # Mock A2A client
        mock_client = Mock()
        mock_client.send_request = AsyncMock(return_value={"inspiration": "test"})
        self.manager.a2a_client = mock_client

        # Request inspiration
        result = await self.manager.request_inspiration(
            context="test context", domain="mythology", target_agent="inspiration_agent"
        )

        self.assertIsNotNone(result)
        mock_client.send_request.assert_called_once()

    async def test_provide_feedback(self):
        """Test providing feedback via A2A"""
        # Mock A2A client
        mock_client = Mock()
        mock_client.send_notification = AsyncMock()
        self.manager.a2a_client = mock_client

        # Provide feedback
        await self.manager.provide_feedback(
            original_request_id="req_123",
            feedback_type="positive",
            content="Great work!",
            target_agent="feedback_agent",
            rating=5,
        )

        mock_client.send_notification.assert_called_once()

    async def test_propose_tool(self):
        """Test proposing tools via A2A"""
        # Mock A2A client
        mock_client = Mock()
        mock_client.send_notification = AsyncMock()
        self.manager.a2a_client = mock_client

        # Propose tool
        await self.manager.propose_tool(
            tool_name="TestTool",
            description="A test tool",
            capabilities=["test"],
            target_agent="tool_agent",
            use_case="Testing",
        )

        mock_client.send_notification.assert_called_once()

    async def test_request_narrative(self):
        """Test requesting narrative via A2A"""
        # Mock A2A client
        mock_client = Mock()
        mock_client.send_request = AsyncMock(return_value={"narrative": "test story"})
        self.manager.a2a_client = mock_client

        # Request narrative
        result = await self.manager.request_narrative(
            theme="test theme",
            elements=["element1", "element2"],
            target_agent="narrative_agent",
        )

        self.assertIsNotNone(result)
        mock_client.send_request.assert_called_once()

    async def test_handle_certification_request_via_a2a(self):
        """Test handling certification requests received via A2A"""
        # Mock process_task
        self.manager.process_task = AsyncMock(
            return_value={"status": "certified", "certificate_vc": "mock_vc_jwt"}
        )

        # Create certification request
        request = CertificationRequest(
            sender_agent="requesting_agent",
            subject="agent_to_certify",
            certification_type="agent_certification",
            evidence={"performance": "good"},
            criteria=["communication", "security", "reliability"],
        )

        # Handle request
        result = await self.manager.handle_certification_request(request)

        self.assertIn("status", result)
        self.assertEqual(result["status"], "certified")
        self.manager.process_task.assert_called_once()

    async def test_autonomous_operation_loop(self):
        """Test autonomous operation loop"""
        # Mock the internal methods
        self.manager._perform_security_audit = AsyncMock()
        self.manager._cleanup_expired_credentials = AsyncMock()

        # Mock asyncio.sleep to avoid infinite loop
        with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
            # Start autonomous operation
            self.manager.is_active = True

            # Run loop for a short time
            loop_task = asyncio.create_task(self.manager._autonomous_loop())
            await asyncio.sleep(0.1)  # Let it run briefly
            loop_task.cancel()  # Cancel the loop

            # Check that security audit was called
            self.manager._perform_security_audit.assert_called()

    async def test_start_stop_autonomous_operation(self):
        """Test starting and stopping autonomous operation"""
        # Mock the autonomous loop
        self.manager._autonomous_loop = AsyncMock()

        # Start operation
        await self.manager.start_autonomous_operation()

        self.assertTrue(self.manager.is_active)
        self.manager._autonomous_loop.assert_called_once()

        # Stop operation
        await self.manager.stop_autonomous_operation()

        self.assertFalse(self.manager.is_active)

    def test_get_status(self):
        """Test getting agent status"""
        # Mock some internal state
        self.manager.is_active = True
        self.manager.a2a_client = Mock()

        status = self.manager.get_status()

        self.assertIn("name", status)
        self.assertIn("is_active", status)
        self.assertIn("a2a_connected", status)
        self.assertEqual(status["is_active"], True)
        self.assertEqual(status["a2a_connected"], True)


if __name__ == "__main__":
    unittest.main()
