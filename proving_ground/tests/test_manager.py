"""
Tests for ProvingGround Manager
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from ..manager import ProvingGroundManager
from ..did_manager import DIDManager
from ..vc_issuer import VCIssuer
from ..certification_tests import CertificationTestSuite


class TestProvingGroundManager(unittest.TestCase):
    """Test cases for ProvingGround Manager"""

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

    def test_initialization(self):
        """Test manager initialization"""
        self.assertEqual(self.manager.name, "TestProvingGroundManager")
        self.assertIsInstance(self.manager.did_manager, DIDManager)
        self.assertIsInstance(self.manager.vc_issuer, VCIssuer)
        self.assertIsInstance(self.manager.test_suite, CertificationTestSuite)
        self.assertIn("agent_certification", self.manager.capabilities)

    @patch("langchain.agents.initialize_agent")
    def test_create_agent_executor(self, mock_initialize_agent):
        """Test agent executor creation"""
        mock_agent = Mock()
        mock_initialize_agent.return_value = mock_agent

        executor = self.manager._create_agent_executor()

        self.assertEqual(executor, mock_agent)
        mock_initialize_agent.assert_called_once()

    async def test_process_certification_task(self):
        """Test processing certification task"""
        # Mock the test suite
        self.manager.test_suite.run_all_tests = AsyncMock(
            return_value={
                "overall_score": 0.9,
                "pass_rate": 0.9,
                "passed_tests": 9,
                "total_tests": 10,
                "results": [],
            }
        )

        # Mock VC issuance
        self.manager._issue_certification_vc = AsyncMock(return_value="mock_vc_jwt")

        result = await self.manager.process_task(
            "Certify agent test_agent",
            agent_id="test_agent_id",
            agent_name="Test Agent",
        )

        self.assertIn("status", result)
        self.assertEqual(result["status"], "certified")
        self.assertIn("certificate_vc", result)

    async def test_process_did_create_task(self):
        """Test processing DID creation task"""
        result = await self.manager.process_task("Create DID", method="terra")

        self.assertIn("operation", result)
        self.assertEqual(result["operation"], "create_did")
        self.assertTrue(result["did"].startswith("did:terra:"))

    async def test_process_did_resolve_task(self):
        """Test processing DID resolution task"""
        # Create a DID first
        create_result = await self.manager.process_task("Create DID", method="terra")
        did = create_result["did"]

        # Resolve the DID
        resolve_result = await self.manager.process_task("Resolve DID", did=did)

        self.assertIn("operation", resolve_result)
        self.assertEqual(resolve_result["operation"], "resolve_did")
        self.assertEqual(resolve_result["did"], did)
        self.assertIsNotNone(resolve_result["document"])

    async def test_process_vc_issue_task(self):
        """Test processing VC issuance task"""
        # Create DIDs first
        issuer_result = await self.manager.process_task("Create DID", method="terra")
        subject_result = await self.manager.process_task("Create DID", method="terra")

        issuer_did = issuer_result["did"]
        subject_did = subject_result["did"]

        # Issue VC
        vc_result = await self.manager.process_task(
            "Issue Verifiable Credential",
            issuer_did=issuer_did,
            subject_did=subject_did,
            credential_type=["TestCredential"],
            claims={"name": "Test Subject"},
        )

        self.assertIn("operation", vc_result)
        self.assertEqual(vc_result["operation"], "issue_vc")
        self.assertIsNotNone(vc_result["vc"])

    async def test_process_vc_verify_task(self):
        """Test processing VC verification task"""
        # Create DIDs and issue VC first
        issuer_result = await self.manager.process_task("Create DID", method="terra")
        subject_result = await self.manager.process_task("Create DID", method="terra")

        issuer_did = issuer_result["did"]
        subject_did = subject_result["did"]

        vc_result = await self.manager.process_task(
            "Issue Verifiable Credential",
            issuer_did=issuer_did,
            subject_did=subject_did,
            credential_type=["TestCredential"],
            claims={"name": "Test Subject"},
        )

        vc_jwt = vc_result["vc"]

        # Verify VC
        verify_result = await self.manager.process_task(
            "Verify Verifiable Credential", vc_jwt=vc_jwt
        )

        self.assertIn("operation", verify_result)
        self.assertEqual(verify_result["operation"], "verify_vc")
        self.assertTrue(verify_result["valid"])

    def test_get_certification_status(self):
        """Test getting certification status"""
        # Test non-existent agent
        status = self.manager.get_certification_status("nonexistent")
        self.assertIsNone(status)

        # Test pending certification
        self.manager.pending_certifications["test_agent"] = {
            "agent_name": "Test Agent",
            "start_time": "2023-01-01T00:00:00",
            "status": "running",
        }

        status = self.manager.get_certification_status("test_agent")
        self.assertIsNotNone(status)
        self.assertEqual(status["status"], "running")

    def test_get_statistics(self):
        """Test getting statistics"""
        stats = self.manager.get_statistics()

        self.assertIn("total_certifications", stats)
        self.assertIn("successful_certifications", stats)
        self.assertIn("pending_certifications", stats)
        self.assertIn("active_dids", stats)
        self.assertIn("issued_credentials", stats)

        # Should have some DIDs from tests
        self.assertGreaterEqual(stats["active_dids"], 0)

    async def test_issue_certification_vc(self):
        """Test issuing certification VC"""
        # Create DIDs
        issuer_did = self.manager.did_manager.create_did("terra")
        subject_did = "did:terra:agent:test_agent"

        test_results = {
            "overall_score": 0.85,
            "pass_rate": 0.85,
            "passed_tests": 17,
            "total_tests": 20,
        }

        vc_jwt = await self.manager._issue_certification_vc(
            "test_agent", "Test Agent", test_results
        )

        self.assertIsNotNone(vc_jwt)
        self.assertIsInstance(vc_jwt, str)

    async def test_handle_certification_request(self):
        """Test handling certification request"""
        from ...a2a_protocol.schemas import CertificationRequest

        request = CertificationRequest(
            sender_agent="test_agent",
            subject="test_subject",
            certification_type="agent_certification",
            evidence={"test": "data"},
            criteria=["communication", "security"],
        )

        # Mock the process_task method
        self.manager.process_task = AsyncMock(
            return_value={"status": "certified", "certificate_vc": "mock_vc"}
        )

        result = await self.manager.handle_certification_request(request)

        self.assertIn("status", result)
        self.assertEqual(result["status"], "certified")


if __name__ == "__main__":
    unittest.main()
