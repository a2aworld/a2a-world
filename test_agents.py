#!/usr/bin/env python3
"""
Test script for Terra Constellata Specialist Agents

This script demonstrates the functionality of the specialist agent system,
showing how agents can work together autonomously using LangChain and the A2A protocol.
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import agents
from agents import (
    AtlasRelationalAnalyst,
    ComparativeMythologyAgent,
    LinguistAgent,
    SentinelOrchestrator,
    agent_registry,
)


# Mock LLM for testing (would be replaced with actual LLM like Llama or GPT-J)
class MockLLM:
    """Mock LLM for testing purposes."""

    def __call__(self, prompt: str) -> str:
        """Generate a mock response based on the prompt."""
        if "relational" in prompt.lower():
            return "I analyze relational data patterns and connections in the dataset."
        elif "myth" in prompt.lower():
            return "I compare mythological elements across different cultures and traditions."
        elif "language" in prompt.lower():
            return "I process and analyze linguistic patterns in text."
        elif "coordinate" in prompt.lower() or "orchestrate" in prompt.lower():
            return "I coordinate and manage the activities of all specialist agents."
        else:
            return "I am a specialist agent ready to assist with my designated tasks."


async def test_individual_agents():
    """Test individual agent functionality."""
    print("=== TESTING INDIVIDUAL AGENTS ===")

    # Create mock LLM
    llm = MockLLM()

    # Test Atlas Relational Analyst
    print("\n1. Testing Atlas Relational Analyst...")
    atlas = AtlasRelationalAnalyst(llm)
    result = await atlas.process_task("Analyze relationships in geospatial data")
    print(f"Atlas Result: {result}")

    # Test Comparative Mythology Agent
    print("\n2. Testing Comparative Mythology Agent...")
    myth_agent = ComparativeMythologyAgent(llm)
    result = await myth_agent.process_task("Compare creation myths across cultures")
    print(f"Mythology Agent Result: {result}")

    # Test Linguist Agent
    print("\n3. Testing Linguist Agent...")
    linguist = LinguistAgent(llm)
    result = await linguist.process_task("Analyze the linguistic patterns in this text")
    print(f"Linguist Agent Result: {result}")

    # Test Sentinel Orchestrator
    print("\n4. Testing Sentinel Orchestrator...")
    sentinel = SentinelOrchestrator(llm)
    result = await sentinel.process_task("Coordinate a research workflow")
    print(f"Sentinel Result: {result}")


async def test_agent_coordination():
    """Test coordination between agents."""
    print("\n=== TESTING AGENT COORDINATION ===")

    llm = MockLLM()

    # Create agents
    atlas = AtlasRelationalAnalyst(llm)
    myth_agent = ComparativeMythologyAgent(llm)
    linguist = LinguistAgent(llm)
    sentinel = SentinelOrchestrator(llm)

    # Register agents
    agent_registry.register_agent(atlas)
    agent_registry.register_agent(myth_agent)
    agent_registry.register_agent(linguist)
    agent_registry.register_agent(sentinel)

    print(f"\nRegistered agents: {agent_registry.list_agents()}")

    # Test coordination
    print("\nTesting coordination workflow...")
    coordination_result = await sentinel.coordinate_agents(
        "Create a comprehensive analysis of ancient civilizations combining "
        "geospatial data, mythological narratives, and linguistic patterns"
    )
    print(f"Coordination Result: {coordination_result}")

    # Test workflow management
    print("\nTesting workflow management...")
    workflow_result = await sentinel.manage_workflow(
        "research_analysis", {"topic": "Ancient Civilizations", "scope": "global"}
    )
    print(f"Workflow Result: {workflow_result}")


async def test_system_monitoring():
    """Test system monitoring capabilities."""
    print("\n=== TESTING SYSTEM MONITORING ===")

    llm = MockLLM()
    sentinel = SentinelOrchestrator(llm)

    # Test health monitoring
    print("\nTesting system health monitoring...")
    health_report = await sentinel.monitor_system_health()
    print(f"Health Report: {health_report}")

    # Test agent status monitoring
    print("\nTesting agent status monitoring...")
    status_result = await sentinel.process_task("Monitor the status of all agents")
    print(f"Status Result: {status_result}")


async def test_autonomous_operation():
    """Test autonomous operation capabilities."""
    print("\n=== TESTING AUTONOMOUS OPERATION ===")

    llm = MockLLM()

    # Create and register agents
    atlas = AtlasRelationalAnalyst(llm)
    myth_agent = ComparativeMythologyAgent(llm)
    linguist = LinguistAgent(llm)
    sentinel = SentinelOrchestrator(llm)

    agent_registry.register_agent(atlas)
    agent_registry.register_agent(myth_agent)
    agent_registry.register_agent(linguist)
    agent_registry.register_agent(sentinel)

    print("\nStarting autonomous operation for all agents...")

    # Start autonomous operation for a short period
    tasks = []

    # Start Sentinel (orchestrator) first
    sentinel_task = asyncio.create_task(sentinel.start_autonomous_operation())
    tasks.append(sentinel_task)

    # Start other agents
    atlas_task = asyncio.create_task(atlas.start_autonomous_operation())
    myth_task = asyncio.create_task(myth_agent.start_autonomous_operation())
    linguist_task = asyncio.create_task(linguist.start_autonomous_operation())

    tasks.extend([atlas_task, myth_task, linguist_task])

    # Let them run for a short time
    await asyncio.sleep(2)

    # Stop autonomous operation
    print("\nStopping autonomous operation...")
    await sentinel.stop_autonomous_operation()
    await atlas.stop_autonomous_operation()
    await myth_agent.stop_autonomous_operation()
    await linguist.stop_autonomous_operation()

    # Cancel tasks
    for task in tasks:
        task.cancel()

    print("Autonomous operation test completed.")


async def main():
    """Main test function."""
    print("🧠 TERRA CONSTELLATA SPECIALIST AGENTS TEST")
    print("=" * 50)

    try:
        # Test individual agents
        await test_individual_agents()

        # Test coordination
        await test_agent_coordination()

        # Test monitoring
        await test_system_monitoring()

        # Test autonomous operation
        await test_autonomous_operation()

        print("\n" + "=" * 50)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("\n📊 SUMMARY:")
        print(f"   - Agents Created: {len(agent_registry.list_agents())}")
        print("   - Agent Types: Atlas, Mythology, Linguist, Sentinel")
        print("   - Integration: A2A Protocol + LangChain")
        print("   - Status: Ready for autonomous operation")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n❌ TEST FAILED: {e}")


if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())
