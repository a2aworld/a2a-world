#!/usr/bin/env python3
"""
Test script for the enhanced Sentinel Orchestrator with LangGraph.
"""

import asyncio
import logging


# Using a simple mock instead of FakeLLM for testing
class FakeLLM:
    def __init__(self, responses=None):
        self.responses = responses or ["Mock response for testing"]
        self.response_index = 0

    def __call__(self, prompt):
        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1
        return response

    def invoke(self, inputs):
        return type("MockResult", (), {"content": self.responses[0]})()


from agents.sentinel.sentinel_orchestrator import SentinelOrchestrator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_enhanced_orchestrator():
    """Test the enhanced Sentinel Orchestrator with LangGraph."""

    # Create a fake LLM for testing
    fake_llm = FakeLLM(responses=["Mock response for testing"])

    # Create the enhanced orchestrator
    orchestrator = SentinelOrchestrator(llm=fake_llm)

    logger.info("Testing enhanced Sentinel Orchestrator...")

    try:
        # Test autonomous discovery
        result = await orchestrator.start_autonomous_discovery(max_iterations=2)

        logger.info(f"Discovery result: {result}")

        if result["status"] == "completed":
            logger.info("✅ Autonomous discovery completed successfully!")
            logger.info(f"Found {result['territories_found']} territories")
            logger.info(f"Generated {result['insights_generated']} insights")
            logger.info(f"Completed {result['iterations_completed']} iterations")

            if result.get("report"):
                logger.info("📋 Final Report:")
                logger.info(result["report"])
        else:
            logger.error(f"❌ Discovery failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_enhanced_orchestrator())
