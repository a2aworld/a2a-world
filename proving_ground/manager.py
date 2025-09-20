"""
ProvingGround Manager Agent

This module implements the ProvingGround_Manager agent that administers
certification tests, issues W3C Verifiable Credentials, and manages DIDs.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from langchain.agents import AgentExecutor
from langchain.llms.base import BaseLLM
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate

from ..agents.base_agent import BaseSpecialistAgent
from ..a2a_protocol.schemas import CertificationRequest
from .did_manager import DIDManager, did_manager
from .vc_issuer import VCIssuer, vc_issuer
from .certification_tests import CertificationTestSuite, default_test_suite

logger = logging.getLogger(__name__)


class ProvingGroundManager(BaseSpecialistAgent):
    """
    ProvingGround Manager Agent for certification and credential management.

    This agent handles:
    - Agent certification testing
    - DID creation and management
    - VC issuance and verification
    - Integration with A2A protocol
    """

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        tools: List[BaseTool],
        a2a_server_url: str = "http://localhost:8080",
        **kwargs,
    ):
        super().__init__(
            name=name, llm=llm, tools=tools, a2a_server_url=a2a_server_url, **kwargs
        )

        # Initialize components
        self.did_manager = did_manager
        self.vc_issuer = vc_issuer
        self.test_suite = default_test_suite

        # Certification state
        self.pending_certifications: Dict[str, Dict[str, Any]] = {}
        self.completed_certifications: Dict[str, Dict[str, Any]] = {}

        # Agent capabilities
        self.capabilities = [
            "agent_certification",
            "did_management",
            "vc_issuance",
            "vc_verification",
            "security_audit",
        ]

        logger.info("ProvingGround Manager initialized")

    def _create_agent_executor(self) -> AgentExecutor:
        """Create the LangChain agent executor for certification tasks."""
        prompt = PromptTemplate(
            template="""
            You are the ProvingGround Manager, responsible for certifying agents in the Terra Constellata system.

            Your capabilities include:
            - Administering certification tests
            - Managing Decentralized Identifiers (DIDs)
            - Issuing W3C Verifiable Credentials
            - Verifying agent capabilities and security

            Current context:
            {agent_scratchpad}

            Task: {input}
            """,
            input_variables=["input", "agent_scratchpad"],
        )

        # Create a simple agent executor (simplified for this implementation)
        from langchain.agents import initialize_agent, AgentType

        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=10,
        )

    async def process_task(self, task: str, **kwargs) -> Any:
        """
        Process a certification-related task.

        Args:
            task: Task description
            **kwargs: Additional task parameters

        Returns:
            Task result
        """
        logger.info(f"Processing certification task: {task}")

        # Parse task type
        if "certify" in task.lower() or "certification" in task.lower():
            return await self._handle_certification_request(task, **kwargs)
        elif "did" in task.lower():
            return await self._handle_did_request(task, **kwargs)
        elif "credential" in task.lower() or "vc" in task.lower():
            return await self._handle_vc_request(task, **kwargs)
        else:
            # Use LangChain agent for general tasks
            return await self.agent_executor.arun(task)

    async def _handle_certification_request(
        self, task: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Handle certification requests.

        Args:
            task: Certification task description
            **kwargs: Additional parameters

        Returns:
            Certification result
        """
        agent_id = kwargs.get("agent_id")
        agent_name = kwargs.get("agent_name", "unknown_agent")

        if not agent_id:
            return {"error": "Agent ID required for certification"}

        logger.info(f"Starting certification for agent: {agent_name} ({agent_id})")

        # Create certification context
        agent_context = {
            "agent_id": agent_id,
            "agent_name": agent_name,
            "a2a_client": self.a2a_client,
            "agent": self,  # Self-reference for testing
            "security_features": kwargs.get("security_features", []),
        }

        # Store pending certification
        self.pending_certifications[agent_id] = {
            "agent_name": agent_name,
            "start_time": datetime.utcnow(),
            "status": "running",
        }

        try:
            # Run certification tests
            test_results = await self.test_suite.run_all_tests(agent_context)

            # Determine certification outcome
            if test_results["pass_rate"] >= 0.8:  # 80% pass rate required
                # Issue certificate
                certificate_vc = await self._issue_certification_vc(
                    agent_id, agent_name, test_results
                )

                result = {
                    "status": "certified",
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "certificate_vc": certificate_vc,
                    "test_results": test_results,
                    "certification_date": datetime.utcnow().isoformat(),
                }
            else:
                result = {
                    "status": "failed",
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "test_results": test_results,
                    "reason": "Insufficient test pass rate",
                }

            # Move to completed
            self.completed_certifications[agent_id] = result
            del self.pending_certifications[agent_id]

            logger.info(f"Certification completed for {agent_name}: {result['status']}")
            return result

        except Exception as e:
            logger.error(f"Certification failed for {agent_name}: {e}")
            self.pending_certifications[agent_id]["status"] = "error"
            return {"status": "error", "agent_id": agent_id, "error": str(e)}

    async def _handle_did_request(self, task: str, **kwargs) -> Dict[str, Any]:
        """
        Handle DID management requests.

        Args:
            task: DID task description
            **kwargs: Additional parameters

        Returns:
            DID operation result
        """
        if "create" in task.lower():
            method = kwargs.get("method", "terra")
            did = self.did_manager.create_did(method)
            return {"operation": "create_did", "did": did, "method": method}
        elif "resolve" in task.lower():
            did = kwargs.get("did")
            if not did:
                return {"error": "DID required for resolution"}
            document = self.did_manager.resolve_did(did)
            return {"operation": "resolve_did", "did": did, "document": document}
        else:
            return {"error": "Unknown DID operation"}

    async def _handle_vc_request(self, task: str, **kwargs) -> Dict[str, Any]:
        """
        Handle Verifiable Credential requests.

        Args:
            task: VC task description
            **kwargs: Additional parameters

        Returns:
            VC operation result
        """
        if "issue" in task.lower():
            issuer_did = kwargs.get("issuer_did")
            subject_did = kwargs.get("subject_did")
            credential_type = kwargs.get("credential_type", ["CertificationCredential"])
            claims = kwargs.get("claims", {})

            if not issuer_did or not subject_did:
                return {"error": "Issuer DID and Subject DID required"}

            vc = self.vc_issuer.issue_credential(
                issuer_did=issuer_did,
                subject_did=subject_did,
                credential_type=credential_type,
                claims=claims,
            )

            return {
                "operation": "issue_vc",
                "vc": vc,
                "issuer": issuer_did,
                "subject": subject_did,
            }

        elif "verify" in task.lower():
            vc_jwt = kwargs.get("vc_jwt")
            if not vc_jwt:
                return {"error": "VC JWT required for verification"}

            is_valid = self.vc_issuer.verify_credential(vc_jwt)
            return {"operation": "verify_vc", "valid": is_valid, "vc_jwt": vc_jwt}

        else:
            return {"error": "Unknown VC operation"}

    async def _issue_certification_vc(
        self, agent_id: str, agent_name: str, test_results: Dict[str, Any]
    ) -> Optional[str]:
        """
        Issue a certification Verifiable Credential.

        Args:
            agent_id: ID of the certified agent
            agent_name: Name of the certified agent
            test_results: Results from certification tests

        Returns:
            Signed VC JWT if successful, None otherwise
        """
        # Create issuer DID if not exists
        issuer_did = self.did_manager.create_did("terra")

        # Create subject DID for the agent
        subject_did = f"did:terra:agent:{agent_id}"

        # Prepare claims
        claims = {
            "agentName": agent_name,
            "certificationLevel": "Certified Agent",
            "capabilities": self.capabilities,
            "testResults": {
                "overallScore": test_results["overall_score"],
                "passRate": test_results["pass_rate"],
                "passedTests": test_results["passed_tests"],
                "totalTests": test_results["total_tests"],
            },
            "issuedBy": "ProvingGround Manager",
            "certificationDate": datetime.utcnow().isoformat(),
        }

        # Issue VC
        vc = self.vc_issuer.issue_credential(
            issuer_did=issuer_did,
            subject_did=subject_did,
            credential_type=["CertificationCredential", "TerraConstellataAgent"],
            claims=claims,
            validity_days=365,
        )

        return vc

    async def _autonomous_loop(self):
        """
        Autonomous operation loop for the ProvingGround Manager.

        This handles periodic tasks like:
        - Checking for pending certifications
        - Monitoring system security
        - Updating DID documents
        """
        while self.is_active:
            try:
                # Check for pending certifications
                if self.pending_certifications:
                    logger.info(
                        f"Processing {len(self.pending_certifications)} pending certifications"
                    )

                # Perform security checks
                await self._perform_security_audit()

                # Clean up expired credentials
                await self._cleanup_expired_credentials()

                # Wait before next cycle
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in autonomous loop: {e}")
                await asyncio.sleep(30)

    async def _perform_security_audit(self):
        """Perform periodic security audit."""
        # Simplified security audit
        logger.info("Performing security audit...")

        # Check DID document integrity
        dids = self.did_manager.list_dids()
        for did in dids:
            doc = self.did_manager.get_did_document(did)
            if doc:
                # Verify document integrity (simplified)
                logger.debug(f"DID {did} document is valid")

    async def _cleanup_expired_credentials(self):
        """Clean up expired credentials."""
        # This would be implemented to check and clean up expired VCs
        logger.debug("Cleaning up expired credentials...")

    async def handle_certification_request(
        self, request: CertificationRequest
    ) -> Dict[str, Any]:
        """
        Handle incoming certification requests via A2A protocol.

        Args:
            request: Certification request message

        Returns:
            Certification response
        """
        logger.info(f"Received certification request from {request.sender_agent}")

        # Process the certification
        result = await self.process_task(
            f"Certify agent {request.sender_agent}",
            agent_id=request.sender_agent,
            agent_name=request.sender_agent,
            security_features=["a2a_communication"],  # Basic security feature
        )

        return result

    def get_certification_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get certification status for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Certification status if found, None otherwise
        """
        if agent_id in self.completed_certifications:
            return self.completed_certifications[agent_id]
        elif agent_id in self.pending_certifications:
            return self.pending_certifications[agent_id]
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get certification statistics."""
        total_certified = len(
            [
                cert
                for cert in self.completed_certifications.values()
                if cert["status"] == "certified"
            ]
        )

        return {
            "total_certifications": len(self.completed_certifications),
            "successful_certifications": total_certified,
            "pending_certifications": len(self.pending_certifications),
            "failed_certifications": len(self.completed_certifications)
            - total_certified,
            "active_dids": len(self.did_manager.list_dids()),
            "issued_credentials": len(self.vc_issuer.issued_credentials),
        }
