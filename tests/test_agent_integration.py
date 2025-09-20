"""
Agent integration tests for Terra Constellata.

Tests the interaction between different agents, their coordination,
and integration with other system components.
"""

import pytest
import asyncio
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.asyncio
async def test_agent_registry_functionality(agent_registry, mock_llm):
    """Test agent registry functionality and agent discovery."""
    if not agent_registry:
        pytest.skip("Agent registry not available")

    # Test agent registration
    initial_count = len(agent_registry.list_agents())

    # Register a test agent
    from agents.base_agent import BaseAgent

    class TestAgent(BaseAgent):
        def __init__(self, llm):
            super().__init__("TestAgent", llm)

        async def process_task(self, task: str) -> str:
            return f"TestAgent processed: {task}"

    test_agent = TestAgent(mock_llm)
    agent_registry.register_agent(test_agent)

    # Verify registration
    assert len(agent_registry.list_agents()) == initial_count + 1
    assert "TestAgent" in agent_registry.list_agents()

    # Test agent retrieval
    retrieved_agent = agent_registry.get_agent("TestAgent")
    assert retrieved_agent is not None
    assert retrieved_agent.name == "TestAgent"

    # Test agent unregistration
    agent_registry.unregister_agent("TestAgent")
    assert len(agent_registry.list_agents()) == initial_count
    assert "TestAgent" not in agent_registry.list_agents()

    logger.info("Agent registry functionality test passed")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.asyncio
async def test_specialist_agent_coordination(agent_registry, mock_llm, sample_data):
    """Test coordination between specialist agents."""
    if not agent_registry:
        pytest.skip("Agent registry not available")

    # Get specialist agents
    atlas = agent_registry.get_agent("AtlasRelationalAnalyst")
    myth_agent = agent_registry.get_agent("ComparativeMythologyAgent")
    linguist = agent_registry.get_agent("LinguistAgent")

    if not all([atlas, myth_agent, linguist]):
        pytest.skip("Required specialist agents not available")

    # Test individual agent processing
    atlas_result = await atlas.process_task(
        "Analyze spatial patterns in geographical data"
    )
    myth_result = await myth_agent.process_task(
        "Compare creation myths across cultures"
    )
    linguist_result = await linguist.process_task(
        "Analyze linguistic patterns in narratives"
    )

    assert atlas_result is not None
    assert myth_result is not None
    assert linguist_result is not None

    # Test cross-agent data sharing
    # Atlas processes geospatial data
    geospatial_task = f"Process coordinates: {sample_data['geospatial']}"
    atlas_analysis = await atlas.process_task(geospatial_task)

    # Myth agent processes mythological data
    myth_task = f"Analyze myths: {sample_data['mythological']}"
    myth_analysis = await myth_agent.process_task(myth_task)

    # Linguist processes text from both
    combined_text = (
        f"Geospatial: {atlas_analysis[:100]}... Mythological: {myth_analysis[:100]}..."
    )
    linguist_analysis = await linguist.process_task(
        f"Analyze combined text: {combined_text}"
    )

    assert linguist_analysis is not None
    assert len(linguist_analysis) > 0

    logger.info("Specialist agent coordination test passed")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.asyncio
async def test_sentinel_orchestrator_integration(agent_registry, mock_llm, sample_data):
    """Test Sentinel orchestrator integration with other agents."""
    if not agent_registry:
        pytest.skip("Agent registry not available")

    sentinel = agent_registry.get_agent("SentinelOrchestrator")
    if not sentinel:
        pytest.skip("Sentinel orchestrator not available")

    # Test workflow creation
    workflow_config = {
        "name": "integrated_analysis",
        "agents": [
            "AtlasRelationalAnalyst",
            "ComparativeMythologyAgent",
            "LinguistAgent",
        ],
        "data": sample_data,
        "objectives": [
            "Analyze geospatial patterns",
            "Compare mythological narratives",
            "Extract linguistic features",
        ],
    }

    # Execute orchestrated workflow
    result = await sentinel.manage_workflow("integrated_analysis", workflow_config)

    assert result is not None
    assert isinstance(result, dict) or isinstance(result, str)

    # Test agent coordination
    coordination_result = await sentinel.coordinate_agents(
        "Create an integrated analysis combining geographical, mythological, and linguistic data"
    )

    assert coordination_result is not None
    assert len(coordination_result) > 0

    # Test system health monitoring
    health_report = await sentinel.monitor_system_health()
    assert health_report is not None

    logger.info("Sentinel orchestrator integration test passed")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.asyncio
async def test_apprentice_agent_learning(agent_registry, mock_llm, sample_data):
    """Test apprentice agent learning and adaptation capabilities."""
    if not agent_registry:
        pytest.skip("Agent registry not available")

    apprentice = agent_registry.get_agent("ApprenticeAgent")
    if not apprentice:
        pytest.skip("Apprentice agent not available")

    # Test learning from examples
    training_examples = [
        {
            "input": "Analyze the relationship between Paris and French culture",
            "output": "Paris serves as the cultural and historical heart of France, influencing art, cuisine, and philosophy.",
            "task_type": "cultural_analysis",
        },
        {
            "input": "Compare Greek and Roman mythology",
            "output": "Greek and Roman mythologies share many deities and stories, with Romans adapting Greek gods to their own pantheon.",
            "task_type": "comparative_mythology",
        },
    ]

    # Train apprentice
    for example in training_examples:
        await apprentice.learn_from_example(
            example["input"], example["output"], example["task_type"]
        )

    # Test apprentice performance on similar tasks
    test_task = "Analyze the connection between London and British literature"
    apprentice_result = await apprentice.process_task(test_task)

    assert apprentice_result is not None
    assert len(apprentice_result) > 0

    # Test adaptation to new task types
    new_task = "Examine linguistic patterns in geographical names"
    adapted_result = await apprentice.process_task(new_task)

    assert adapted_result is not None

    logger.info("Apprentice agent learning test passed")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.asyncio
async def test_agent_communication_protocol(agent_registry, a2a_server, mock_llm):
    """Test agent communication through A2A protocol."""
    if not all([agent_registry, a2a_server]):
        pytest.skip("Agent registry or A2A server not available")

    # Test agent-to-agent messaging
    atlas = agent_registry.get_agent("AtlasRelationalAnalyst")
    myth_agent = agent_registry.get_agent("ComparativeMythologyAgent")

    if not all([atlas, myth_agent]):
        pytest.skip("Required agents not available")

    # Create A2A client for testing
    from a2a_protocol.client import A2AClient

    client = A2AClient(a2a_server.host, a2a_server.port)

    # Test message exchange between agents
    message = {
        "from": "AtlasRelationalAnalyst",
        "to": "ComparativeMythologyAgent",
        "type": "collaboration_request",
        "payload": {
            "task": "correlate_geographical_sites_with_mythological_narratives",
            "data": {
                "locations": ["Delphi", "Olympus", "Troy"],
                "myth_types": ["heroic", "divine", "foundational"],
            },
        },
    }

    # Send message through A2A protocol
    response = await client.send_message(message)
    assert response is not None

    # Test agent response processing
    if hasattr(myth_agent, "process_a2a_message"):
        agent_response = await myth_agent.process_a2a_message(message)
        assert agent_response is not None

    logger.info("Agent communication protocol test passed")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.performance
async def test_agent_performance_under_load(agent_registry, mock_llm, benchmark):
    """Test agent performance under concurrent load."""
    if not agent_registry:
        pytest.skip("Agent registry not available")

    async def benchmark_agent_processing():
        """Benchmark concurrent agent processing."""
        tasks = []
        agents = agent_registry.list_agents()

        # Create concurrent tasks for all available agents
        for agent_name in agents[:3]:  # Limit to first 3 agents
            agent = agent_registry.get_agent(agent_name)
            if agent:
                task = asyncio.create_task(
                    agent.process_task(f"Load test task for {agent_name}")
                )
                tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        return results

    # Benchmark the concurrent processing
    def sync_benchmark():
        return asyncio.run(benchmark_agent_processing())
    results = benchmark(sync_benchmark)

    assert len(results) > 0
    for result in results:
        assert result is not None
        assert len(result) > 0

    logger.info("Agent performance under load test completed")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.asyncio
async def test_agent_error_handling_and_recovery(agent_registry, mock_llm):
    """Test agent error handling and recovery mechanisms."""
    if not agent_registry:
        pytest.skip("Agent registry not available")

    # Test with a mock agent that can simulate failures
    from agents.base_agent import BaseAgent

    class FaultyAgent(BaseAgent):
        def __init__(self, llm, fail_count=2):
            super().__init__("FaultyAgent", llm)
            self.fail_count = fail_count
            self.call_count = 0

        async def process_task(self, task: str) -> str:
            self.call_count += 1
            if self.call_count <= self.fail_count:
                raise Exception(f"Simulated failure #{self.call_count}")
            return f"Task completed after {self.call_count} attempts: {task}"

    # Register faulty agent
    faulty_agent = FaultyAgent(mock_llm)
    agent_registry.register_agent(faulty_agent)

    # Test error handling
    result = None
    for i in range(5):
        try:
            result = await faulty_agent.process_task("Test error handling")
            break
        except Exception as e:
            if i == 4:
                assert "Simulated failure" in str(e)
    if result:
        assert "completed after" in result
        assert faulty_agent.call_count > faulty_agent.fail_count

    # Test recovery and continued operation
    result2 = await faulty_agent.process_task("Test recovery")
    assert result2 is not None

    # Cleanup
    agent_registry.unregister_agent("FaultyAgent")

    logger.info("Agent error handling and recovery test passed")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.asyncio
async def test_agent_state_persistence(agent_registry, mock_llm, temp_directories):
    """Test agent state persistence across sessions."""
    if not agent_registry:
        pytest.skip("Agent registry not available")

    apprentice = agent_registry.get_agent("ApprenticeAgent")
    if not apprentice:
        pytest.skip("Apprentice agent not available")

    # Test learning and state accumulation
    learning_data = [
        ("pattern1", "response1", "task_type1"),
        ("pattern2", "response2", "task_type2"),
        ("pattern3", "response3", "task_type1"),
    ]

    # Learn patterns
    for pattern, response, task_type in learning_data:
        await apprentice.learn_from_example(pattern, response, task_type)

    # Test state persistence (if implemented)
    if hasattr(apprentice, "save_state"):
        state_file = f"{temp_directories['data']}/apprentice_state.json"
        await apprentice.save_state(state_file)

        # Verify state file was created
        import os

        assert os.path.exists(state_file)

    # Test state retrieval (if implemented)
    if hasattr(apprentice, "load_state"):
        new_apprentice = type(apprentice)(mock_llm)
        await new_apprentice.load_state(state_file)

        # Test that learned patterns are retained
        test_result = await new_apprentice.process_task("pattern1")
        assert test_result is not None

    logger.info("Agent state persistence test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
