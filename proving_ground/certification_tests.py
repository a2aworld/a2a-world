"""
Certification Test Suite

This module defines test suites for certifying agent capabilities,
including communication, task processing, and security compliance.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum

from ..a2a_protocol.schemas import CertificationRequest

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Status of a certification test"""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class CertificationTest:
    """Individual certification test"""

    def __init__(
        self,
        test_id: str,
        name: str,
        description: str,
        test_function: Callable,
        required_score: float = 0.8,
        timeout: int = 30,
    ):
        self.test_id = test_id
        self.name = name
        self.description = description
        self.test_function = test_function
        self.required_score = required_score
        self.timeout = timeout
        self.status = TestStatus.PENDING
        self.score = 0.0
        self.result = None
        self.error_message = None
        self.start_time = None
        self.end_time = None

    async def run(self, agent_context: Dict[str, Any]) -> bool:
        """
        Run the certification test.

        Args:
            agent_context: Context information about the agent being tested

        Returns:
            True if test passed, False otherwise
        """
        self.status = TestStatus.RUNNING
        self.start_time = datetime.utcnow()

        try:
            # Run test with timeout
            result = await asyncio.wait_for(
                self.test_function(agent_context), timeout=self.timeout
            )

            self.end_time = datetime.utcnow()
            self.result = result

            # Evaluate result
            if isinstance(result, dict) and "score" in result:
                self.score = result["score"]
            elif isinstance(result, bool):
                self.score = 1.0 if result else 0.0
            else:
                self.score = float(result) if result else 0.0

            if self.score >= self.required_score:
                self.status = TestStatus.PASSED
                return True
            else:
                self.status = TestStatus.FAILED
                return False

        except asyncio.TimeoutError:
            self.status = TestStatus.ERROR
            self.error_message = "Test timed out"
            return False
        except Exception as e:
            self.status = TestStatus.ERROR
            self.error_message = str(e)
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert test to dictionary"""
        return {
            "test_id": self.test_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "score": self.score,
            "required_score": self.required_score,
            "result": self.result,
            "error_message": self.error_message,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": (self.end_time - self.start_time).total_seconds()
            if self.start_time and self.end_time
            else None,
        }


class CertificationTestSuite:
    """Suite of certification tests"""

    def __init__(self, suite_name: str, description: str):
        self.suite_name = suite_name
        self.description = description
        self.tests: Dict[str, CertificationTest] = {}
        self.overall_score = 0.0
        self.passed_tests = 0
        self.total_tests = 0

    def add_test(self, test: CertificationTest):
        """Add a test to the suite"""
        self.tests[test.test_id] = test
        self.total_tests = len(self.tests)

    def remove_test(self, test_id: str):
        """Remove a test from the suite"""
        if test_id in self.tests:
            del self.tests[test_id]
            self.total_tests = len(self.tests)

    async def run_all_tests(self, agent_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all tests in the suite.

        Args:
            agent_context: Context information about the agent being tested

        Returns:
            Test results summary
        """
        logger.info(f"Running certification test suite: {self.suite_name}")

        results = []
        total_score = 0.0
        passed_count = 0

        for test in self.tests.values():
            logger.info(f"Running test: {test.name}")
            passed = await test.run(agent_context)
            results.append(test.to_dict())

            if passed:
                passed_count += 1
                total_score += test.score

        self.passed_tests = passed_count
        self.overall_score = total_score / len(self.tests) if self.tests else 0.0

        summary = {
            "suite_name": self.suite_name,
            "description": self.description,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.total_tests - self.passed_tests,
            "overall_score": self.overall_score,
            "pass_rate": self.passed_tests / self.total_tests
            if self.total_tests > 0
            else 0.0,
            "results": results,
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.info(f"Test suite completed. Pass rate: {summary['pass_rate']:.2%}")
        return summary

    def get_test_results(self) -> List[Dict[str, Any]]:
        """Get results of all tests"""
        return [test.to_dict() for test in self.tests.values()]

    def get_failed_tests(self) -> List[CertificationTest]:
        """Get list of failed tests"""
        return [
            test for test in self.tests.values() if test.status == TestStatus.FAILED
        ]

    def reset_tests(self):
        """Reset all tests to pending status"""
        for test in self.tests.values():
            test.status = TestStatus.PENDING
            test.score = 0.0
            test.result = None
            test.error_message = None
            test.start_time = None
            test.end_time = None


# Predefined test functions


async def test_a2a_communication(agent_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test A2A protocol communication capabilities.

    Args:
        agent_context: Agent context containing A2A client and other info

    Returns:
        Test result with score
    """
    try:
        a2a_client = agent_context.get("a2a_client")
        if not a2a_client:
            return {"score": 0.0, "error": "No A2A client available"}

        # Test basic connectivity
        connected = await a2a_client.connect()
        if not connected:
            return {"score": 0.0, "error": "Failed to connect to A2A server"}

        # Test message sending (simplified)
        test_message = {"type": "test", "content": "certification_test"}
        response = await a2a_client.send_request("TEST", test_message)

        await a2a_client.disconnect()

        if response:
            return {"score": 1.0, "message": "A2A communication successful"}
        else:
            return {"score": 0.5, "message": "Partial A2A communication success"}

    except Exception as e:
        return {"score": 0.0, "error": str(e)}


async def test_task_processing(agent_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test task processing capabilities.

    Args:
        agent_context: Agent context

    Returns:
        Test result with score
    """
    try:
        agent = agent_context.get("agent")
        if not agent:
            return {"score": 0.0, "error": "No agent available"}

        # Test basic task processing
        test_task = "Process this simple test task"
        result = await agent.process_task(test_task)

        if result and len(str(result)) > 0:
            return {"score": 1.0, "message": "Task processing successful"}
        else:
            return {"score": 0.0, "error": "Task processing failed"}

    except Exception as e:
        return {"score": 0.0, "error": str(e)}


async def test_security_compliance(agent_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test security compliance.

    Args:
        agent_context: Agent context

    Returns:
        Test result with score
    """
    try:
        # Check for required security features
        security_features = agent_context.get("security_features", [])

        required_features = ["encryption", "authentication", "authorization"]
        implemented_features = 0

        for feature in required_features:
            if feature in security_features:
                implemented_features += 1

        score = implemented_features / len(required_features)

        return {
            "score": score,
            "implemented_features": implemented_features,
            "total_features": len(required_features),
        }

    except Exception as e:
        return {"score": 0.0, "error": str(e)}


# Create default certification test suite
def create_default_test_suite() -> CertificationTestSuite:
    """Create a default certification test suite"""
    suite = CertificationTestSuite(
        "Terra Constellata Agent Certification",
        "Comprehensive certification tests for Terra Constellata agents",
    )

    # Add predefined tests
    suite.add_test(
        CertificationTest(
            "a2a_communication",
            "A2A Protocol Communication",
            "Test ability to communicate via A2A protocol",
            test_a2a_communication,
            required_score=0.8,
        )
    )

    suite.add_test(
        CertificationTest(
            "task_processing",
            "Task Processing",
            "Test ability to process tasks autonomously",
            test_task_processing,
            required_score=0.9,
        )
    )

    suite.add_test(
        CertificationTest(
            "security_compliance",
            "Security Compliance",
            "Test security features and compliance",
            test_security_compliance,
            required_score=0.7,
        )
    )

    return suite


# Global test suite instance
default_test_suite = create_default_test_suite()
