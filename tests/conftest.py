"""
Pytest configuration and fixtures for Terra Constellata integration tests.

This module provides shared fixtures and configuration for all integration tests,
including database connections, service clients, and test data setup.
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from typing import Dict, Any, AsyncGenerator, Generator
from pathlib import Path
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_CONFIG = {
    "databases": {
        "postgis": {
            "host": os.getenv("TEST_POSTGIS_HOST", "localhost"),
            "port": int(os.getenv("TEST_POSTGIS_PORT", "5432")),
            "database": os.getenv("TEST_POSTGIS_DB", "terra_constellata_test"),
            "user": os.getenv("TEST_POSTGIS_USER", "postgres"),
            "password": os.getenv("TEST_POSTGIS_PASSWORD", "postgres"),
        },
        "arangodb": {
            "host": os.getenv("TEST_ARANGODB_HOST", "localhost"),
            "port": int(os.getenv("TEST_ARANGODB_PORT", "8529")),
            "database": os.getenv("TEST_ARANGODB_DB", "terra_constellata_test"),
            "user": os.getenv("TEST_ARANGODB_USER", "root"),
            "password": os.getenv("TEST_ARANGODB_PASSWORD", ""),
        },
    },
    "services": {
        "backend": {
            "host": os.getenv("TEST_BACKEND_HOST", "localhost"),
            "port": int(os.getenv("TEST_BACKEND_PORT", "8000")),
        },
        "a2a_server": {
            "host": os.getenv("TEST_A2A_HOST", "localhost"),
            "port": int(os.getenv("TEST_A2A_PORT", "8080")),
        },
    },
    "temp_dirs": {"data": None, "logs": None, "cache": None},
}


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return TEST_CONFIG


@pytest.fixture(scope="session")
def temp_directories():
    """Create temporary directories for test data."""
    temp_base = tempfile.mkdtemp(prefix="terra_constellata_test_")

    TEST_CONFIG["temp_dirs"]["data"] = os.path.join(temp_base, "data")
    TEST_CONFIG["temp_dirs"]["logs"] = os.path.join(temp_base, "logs")
    TEST_CONFIG["temp_dirs"]["cache"] = os.path.join(temp_base, "cache")

    # Create directories
    for dir_path in TEST_CONFIG["temp_dirs"].values():
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    yield TEST_CONFIG["temp_dirs"]

    # Cleanup
    shutil.rmtree(temp_base, ignore_errors=True)


@pytest.fixture(scope="session")
async def postgis_connection(test_config):
    """Provide PostGIS database connection for tests."""
    try:
        from data.postgis.connection import PostGISConnection

        config = test_config["databases"]["postgis"]
        connection = PostGISConnection(
            host=config["host"],
            port=config["port"],
            database=config["database"],
            user=config["user"],
            password=config["password"],
        )

        # Initialize connection
        await connection.connect()
        await connection.initialize_schema()

        yield connection

        # Cleanup
        await connection.disconnect()

    except Exception as e:
        logger.warning(f"PostGIS connection failed: {e}")
        yield None


@pytest.fixture(scope="session")
async def ckg_connection(test_config):
    """Provide CKG (ArangoDB) connection for tests."""
    try:
        from data.ckg.connection import CKGConnection

        config = test_config["databases"]["arangodb"]
        connection = CKGConnection(
            host=config["host"],
            port=config["port"],
            database=config["database"],
            username=config["user"],
            password=config["password"],
        )

        # Initialize connection
        await connection.connect()
        await connection.initialize_schema()

        yield connection

        # Cleanup
        await connection.disconnect()

    except Exception as e:
        logger.warning(f"CKG connection failed: {e}")
        yield None


@pytest.fixture(scope="session")
async def a2a_server(test_config):
    """Start A2A protocol server for tests."""
    try:
        from a2a_protocol.server import A2AServer

        config = test_config["services"]["a2a_server"]
        server = A2AServer(host=config["host"], port=config["port"])

        # Start server
        await server.start()

        yield server

        # Stop server
        await server.stop()

    except Exception as e:
        logger.warning(f"A2A server startup failed: {e}")
        yield None


@pytest.fixture(scope="session")
async def backend_app(test_config, temp_directories):
    """Start FastAPI backend for tests."""
    try:
        from backend.main import app
        from fastapi.testclient import TestClient

        # Set test environment
        os.environ["TESTING"] = "true"

        # Override database paths to use temp directories
        if temp_directories["data"]:
            os.environ["DATA_DIR"] = temp_directories["data"]
        if temp_directories["logs"]:
            os.environ["LOG_DIR"] = temp_directories["logs"]

        client = TestClient(app)

        yield client

    except Exception as e:
        logger.warning(f"Backend app startup failed: {e}")
        yield None


@pytest.fixture
async def mock_llm():
    """Provide mock LLM for testing agents without external dependencies."""

    class MockLLM:
        def __call__(self, prompt: str) -> str:
            """Generate mock responses based on prompt content."""
            if "relational" in prompt.lower() or "atlas" in prompt.lower():
                return (
                    "I analyze relational patterns and geospatial connections in data."
                )
            elif "myth" in prompt.lower() or "comparative" in prompt.lower():
                return "I compare mythological narratives across different cultures and traditions."
            elif "language" in prompt.lower() or "linguist" in prompt.lower():
                return "I process and analyze linguistic patterns in text data."
            elif "coordinate" in prompt.lower() or "orchestrate" in prompt.lower():
                return (
                    "I coordinate and manage the activities of all specialist agents."
                )
            elif "apprentice" in prompt.lower() or "learn" in prompt.lower():
                return "I learn from examples and improve through iterative training."
            else:
                return (
                    "I am a specialist agent ready to assist with my designated tasks."
                )

        async def agenerate(self, prompt: str) -> str:
            """Async version of generate."""
            return self(prompt)

    return MockLLM()


@pytest.fixture
async def agent_registry(mock_llm):
    """Provide initialized agent registry for tests."""
    try:
        from agents import agent_registry
        from agents.atlas.atlas_relational_analyst import AtlasRelationalAnalyst
        from agents.myth.comparative_mythology_agent import ComparativeMythologyAgent
        from agents.lang.linguist_agent import LinguistAgent
        from agents.sentinel.sentinel_orchestrator import SentinelOrchestrator
        from agents.apprentice.apprentice_agent import ApprenticeAgent

        # Clear existing registry
        agent_registry.clear()

        # Create and register agents
        atlas = AtlasRelationalAnalyst(mock_llm)
        myth_agent = ComparativeMythologyAgent(mock_llm)
        linguist = LinguistAgent(mock_llm)
        sentinel = SentinelOrchestrator(mock_llm)
        apprentice = ApprenticeAgent(mock_llm)

        agent_registry.register_agent(atlas)
        agent_registry.register_agent(myth_agent)
        agent_registry.register_agent(linguist)
        agent_registry.register_agent(sentinel)
        agent_registry.register_agent(apprentice)

        yield agent_registry

        # Cleanup
        agent_registry.clear()

    except Exception as e:
        logger.warning(f"Agent registry setup failed: {e}")
        yield None


@pytest.fixture
def sample_data():
    """Provide sample test data for various components."""
    return {
        "geospatial": [
            {
                "name": "New York City",
                "entity": "city",
                "latitude": 40.7128,
                "longitude": -74.0060,
                "description": "Major metropolitan area",
            },
            {
                "name": "Central Park",
                "entity": "park",
                "latitude": 40.7829,
                "longitude": -73.9654,
                "description": "Urban park in Manhattan",
            },
            {
                "name": "Eiffel Tower",
                "entity": "landmark",
                "latitude": 48.8584,
                "longitude": 2.2945,
                "description": "Iconic iron tower",
            },
        ],
        "mythological": [
            {
                "culture": "Greek",
                "myth": "Creation of the World",
                "narrative": "In the beginning, there was Chaos...",
            },
            {
                "culture": "Norse",
                "myth": "Ragnarok",
                "narrative": "The end times when gods and giants clash...",
            },
        ],
        "linguistic": [
            {
                "text": "The quick brown fox jumps over the lazy dog",
                "language": "English",
                "patterns": ["subject-verb-object"],
            }
        ],
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "database: mark test as requiring database")
    config.addinivalue_line("markers", "agent: mark test as requiring agents")
    config.addinivalue_line(
        "markers", "performance: mark test as performance benchmark"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Auto-mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Auto-mark database tests
        if any(
            keyword in str(item.fspath)
            for keyword in ["database", "db", "postgis", "ckg"]
        ):
            item.add_marker(pytest.mark.database)

        # Auto-mark agent tests
        if "agent" in str(item.fspath):
            item.add_marker(pytest.mark.agent)

        # Auto-mark performance tests
        if any(
            keyword in str(item.fspath)
            for keyword in ["performance", "benchmark", "load"]
        ):
            item.add_marker(pytest.mark.performance)
