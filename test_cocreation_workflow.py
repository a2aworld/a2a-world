#!/usr/bin/env python3
"""
Test script for the Co-Creation Workflow integration.

This script tests the complete human-AI co-creation workflow:
doubt -> discovery -> art -> wisdom -> knowledge
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from workflow.cocreation_workflow import CoCreationWorkflow
from agents.sentinel.sentinel_orchestrator import SentinelOrchestrator
from agents.apprentice.apprentice_agent import ApprenticeAgent
from codex.codex_manager import CodexManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_cocreation_workflow():
    """Test the complete co-creation workflow."""
    logger.info("🧪 Starting Co-Creation Workflow Integration Test")

    try:
        # Initialize components
        logger.info("Initializing components...")

        # Initialize Codex Manager
        codex = CodexManager("./test_codex_data")

        # Initialize Sentinel Orchestrator (without LLM for testing)
        sentinel = SentinelOrchestrator(llm=None)

        # Initialize Apprentice Agent (without LLM for testing)
        apprentice = ApprenticeAgent(llm=None)

        # Initialize Co-Creation Workflow
        workflow = CoCreationWorkflow(
            sentinel=sentinel, apprentice=apprentice, codex=codex, chatbot=None
        )

        logger.info("✅ Components initialized successfully")

        # Test 1: Human-triggered workflow
        logger.info("🧪 Test 1: Human-triggered workflow")
        human_input = (
            "Explore the creative potential of ancient mythological landscapes"
        )

        workflow_result = await workflow.start_cocreation_workflow(
            trigger_source="human",
            human_input=human_input,
            workflow_id="test_workflow_001",
        )

        logger.info(f"Workflow started: {workflow_result['workflow_id']}")
        assert (
            workflow_result["success"] == False
        )  # Should be False initially as it's async

        # Check workflow status
        status = workflow.get_workflow_status("test_workflow_001")
        logger.info(f"Workflow status: {status}")
        assert status is not None
        assert status["workflow_id"] == "test_workflow_001"

        # Test 2: Autonomous workflow
        logger.info("🧪 Test 2: Autonomous workflow")
        autonomous_result = await workflow.start_cocreation_workflow(
            trigger_source="autonomous", workflow_id="test_autonomous_001"
        )

        logger.info(f"Autonomous workflow started: {autonomous_result['workflow_id']}")

        # Test 3: Workflow history
        logger.info("🧪 Test 3: Workflow history")
        history = workflow.get_workflow_history()
        logger.info(f"Workflow history: {len(history)} workflows")
        assert len(history) >= 2

        # Test 4: Philosophy endpoint (would need running server)
        logger.info("🧪 Test 4: Philosophy integration check")
        # This would be tested with a running server

        logger.info("✅ All workflow integration tests passed!")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_component_imports():
    """Test that all components can be imported."""
    logger.info("🧪 Testing component imports...")

    try:
        from workflow.cocreation_workflow import CoCreationWorkflow
        from agents.sentinel.sentinel_orchestrator import SentinelOrchestrator
        from agents.apprentice.apprentice_agent import ApprenticeAgent
        from codex.codex_manager import CodexManager
        from backend.api.workflow import router as workflow_router

        logger.info("✅ All components imported successfully")
        return True

    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def test_api_endpoints():
    """Test API endpoint definitions."""
    logger.info("🧪 Testing API endpoint definitions...")

    try:
        from backend.api.workflow import router as workflow_router

        # Check that expected routes exist
        routes = [route.path for route in workflow_router.routes]
        expected_routes = [
            "/start",
            "/status/{workflow_id}",
            "/feedback",
            "/history",
            "/trigger-autonomous",
            "/philosophy",
            "/stats",
        ]

        for route in expected_routes:
            assert route in routes, f"Missing route: {route}"

        logger.info("✅ All expected API routes found")
        return True

    except Exception as e:
        logger.error(f"❌ API test failed: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("🚀 Starting Terra Constellata Co-Creation Workflow Tests")
    logger.info("=" * 60)

    results = []

    # Test component imports
    results.append(("Component Imports", test_component_imports()))

    # Test API endpoints
    results.append(("API Endpoints", test_api_endpoints()))

    # Test workflow integration
    results.append(("Workflow Integration", await test_cocreation_workflow()))

    # Summary
    logger.info("=" * 60)
    logger.info("📊 Test Results Summary:")

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
        if success:
            passed += 1

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! Co-creation workflow is ready.")
        return 0
    else:
        logger.error("⚠️  Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
