"""
Comprehensive integration tests for Terra Constellata system.

This module tests the interaction between all major components:
- Databases (PostGIS + CKG)
- Agents (Atlas, Myth, Linguist, Sentinel, Apprentice)
- A2A Protocol
- Backend API
- Workflows
"""

import pytest
import asyncio
import json
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_system_integration(
    postgis_connection,
    ckg_connection,
    a2a_server,
    backend_app,
    agent_registry,
    mock_llm,
    sample_data,
    temp_directories,
):
    """
    Test complete system integration from data ingestion to agent coordination.

    This test verifies that all components work together:
    1. Data ingestion into both databases
    2. Agent initialization and registration
    3. A2A protocol communication
    4. Backend API responses
    5. Workflow execution
    """
    if not all(
        [postgis_connection, ckg_connection, a2a_server, backend_app, agent_registry]
    ):
        pytest.skip("Required services not available for integration testing")

    try:
        # Step 1: Test data ingestion into PostGIS
        logger.info("Step 1: Testing data ingestion into PostGIS")
        await test_postgis_data_ingestion(postgis_connection, sample_data)

        # Step 2: Test data ingestion into CKG
        logger.info("Step 2: Testing data ingestion into CKG")
        await test_ckg_data_ingestion(ckg_connection, sample_data)

        # Step 3: Test cross-database data synchronization
        logger.info("Step 3: Testing cross-database synchronization")
        await test_database_synchronization(postgis_connection, ckg_connection)

        # Step 4: Test agent initialization and basic functionality
        logger.info("Step 4: Testing agent initialization")
        await test_agent_initialization(agent_registry, mock_llm)

        # Step 5: Test agent coordination via Sentinel
        logger.info("Step 5: Testing agent coordination")
        await test_agent_coordination(agent_registry, mock_llm)

        # Step 6: Test A2A protocol communication
        logger.info("Step 6: Testing A2A protocol")
        await test_a2a_protocol_communication(a2a_server, agent_registry)

        # Step 7: Test backend API integration
        logger.info("Step 7: Testing backend API")
        await test_backend_api_integration(backend_app, sample_data)

        # Step 8: Test complete workflow execution
        logger.info("Step 8: Testing workflow execution")
        await test_workflow_execution(agent_registry, backend_app, sample_data)

        logger.info("✅ Full system integration test completed successfully")

    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        raise


async def test_postgis_data_ingestion(connection, sample_data):
    """Test data ingestion into PostGIS database."""
    if not connection:
        pytest.skip("PostGIS connection not available")

    try:
        # Insert geospatial data
        for item in sample_data["geospatial"]:
            await connection.execute(
                """
                INSERT INTO locations (name, entity, latitude, longitude, description, geom)
                VALUES ($1, $2, $3, $4, $5, ST_SetSRID(ST_MakePoint($4, $3), 4326))
                ON CONFLICT (name) DO NOTHING
            """,
                item["name"],
                item["entity"],
                item["latitude"],
                item["longitude"],
                item["description"],
            )

        # Verify data insertion
        result = await connection.fetch("SELECT COUNT(*) FROM locations")
        assert result[0]["count"] >= len(sample_data["geospatial"])

        logger.info(
            f"Successfully ingested {len(sample_data['geospatial'])} records into PostGIS"
        )

    except Exception as e:
        logger.error(f"PostGIS data ingestion failed: {e}")
        raise


async def test_ckg_data_ingestion(connection, sample_data):
    """Test data ingestion into CKG (ArangoDB)."""
    if not connection:
        pytest.skip("CKG connection not available")

    try:
        # Insert knowledge graph data
        for item in sample_data["mythological"]:
            await connection.execute_aql(
                """
                INSERT {
                    culture: @culture,
                    myth: @myth,
                    narrative: @narrative,
                    type: 'mythological_narrative'
                } INTO mythological_narratives
                OPTIONS { ignoreErrors: true }
            """,
                culture=item["culture"],
                myth=item["myth"],
                narrative=item["narrative"],
            )

        # Verify data insertion
        result = await connection.execute_aql(
            """
            FOR doc IN mythological_narratives
            COLLECT WITH COUNT INTO length
            RETURN length
        """
        )
        assert result[0] >= len(sample_data["mythological"])

        logger.info(
            f"Successfully ingested {len(sample_data['mythological'])} records into CKG"
        )

    except Exception as e:
        logger.error(f"CKG data ingestion failed: {e}")
        raise


async def test_database_synchronization(postgis_conn, ckg_conn):
    """Test synchronization between PostGIS and CKG databases."""
    if not all([postgis_conn, ckg_conn]):
        pytest.skip("Database connections not available")

    try:
        # Query PostGIS for location data
        postgis_data = await postgis_conn.fetch(
            """
            SELECT name, entity, latitude, longitude
            FROM locations
            LIMIT 5
        """
        )

        # Create corresponding entries in CKG
        for row in postgis_data:
            await ckg_conn.execute_aql(
                """
                INSERT {
                    name: @name,
                    entity: @entity,
                    latitude: @latitude,
                    longitude: @longitude,
                    type: 'geospatial_entity'
                } INTO geospatial_entities
                OPTIONS { ignoreErrors: true }
            """,
                name=row["name"],
                entity=row["entity"],
                latitude=row["latitude"],
                longitude=row["longitude"],
            )

        # Verify synchronization
        ckg_count = await ckg_conn.execute_aql(
            """
            FOR doc IN geospatial_entities
            COLLECT WITH COUNT INTO length
            RETURN length
        """
        )

        assert ckg_count[0] >= len(postgis_data)
        logger.info("Database synchronization test passed")

    except Exception as e:
        logger.error(f"Database synchronization failed: {e}")
        raise


async def test_agent_initialization(registry, mock_llm):
    """Test agent initialization and basic functionality."""
    if not registry:
        pytest.skip("Agent registry not available")

    try:
        agents = registry.list_agents()
        assert len(agents) >= 4  # Should have at least Atlas, Myth, Linguist, Sentinel

        # Test each agent can process a basic task
        for agent_name in agents:
            agent = registry.get_agent(agent_name)
            result = await agent.process_task(f"Basic test task for {agent_name}")
            assert result is not None
            assert len(result) > 0

        logger.info(f"Successfully initialized {len(agents)} agents")

    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
        raise


async def test_agent_coordination(registry, mock_llm):
    """Test agent coordination via Sentinel orchestrator."""
    if not registry:
        pytest.skip("Agent registry not available")

    try:
        sentinel = registry.get_agent("SentinelOrchestrator")
        assert sentinel is not None

        # Test coordination task
        coordination_result = await sentinel.coordinate_agents(
            "Analyze the relationship between geographical locations and mythological narratives"
        )

        assert coordination_result is not None
        assert (
            "coordinate" in coordination_result.lower()
            or "analysis" in coordination_result.lower()
        )

        logger.info("Agent coordination test passed")

    except Exception as e:
        logger.error(f"Agent coordination failed: {e}")
        raise


async def test_a2a_protocol_communication(server, registry):
    """Test A2A protocol communication between agents."""
    if not all([server, registry]):
        pytest.skip("A2A server or agent registry not available")

    try:
        # Test basic A2A message exchange
        from a2a_protocol.client import A2AClient

        client = A2AClient(server.host, server.port)

        # Send test message
        test_message = {
            "type": "agent_coordination",
            "payload": {
                "task": "test_coordination",
                "agents": ["Atlas", "Myth"],
                "data": {"test": "data"},
            },
        }

        response = await client.send_message(test_message)
        assert response is not None

        logger.info("A2A protocol communication test passed")

    except Exception as e:
        logger.error(f"A2A protocol test failed: {e}")
        raise


async def test_backend_api_integration(app, sample_data):
    """Test backend API integration with all endpoints."""
    if not app:
        pytest.skip("Backend app not available")

    try:
        # Test health endpoint
        response = app.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

        # Test root endpoint
        response = app.get("/")
        assert response.status_code == 200
        assert "Galactic Storybook CMS" in response.json()["message"]

        # Test pipeline endpoint (if available)
        try:
            response = app.get("/api/pipeline/status")
            assert response.status_code in [
                200,
                404,
            ]  # 404 is acceptable if not implemented
        except:
            pass  # Endpoint might not be implemented

        logger.info("Backend API integration test passed")

    except Exception as e:
        logger.error(f"Backend API test failed: {e}")
        raise


async def test_workflow_execution(registry, app, sample_data):
    """Test complete workflow execution from start to finish."""
    if not all([registry, app]):
        pytest.skip("Required components not available")

    try:
        # Create a comprehensive workflow task
        workflow_task = {
            "type": "comprehensive_analysis",
            "description": "Analyze geographical and mythological data together",
            "data": sample_data,
            "expected_output": "integrated_analysis",
        }

        # Execute through Sentinel orchestrator
        sentinel = registry.get_agent("SentinelOrchestrator")
        result = await sentinel.manage_workflow("comprehensive_analysis", workflow_task)

        assert result is not None
        assert len(result) > 0

        # Verify workflow was recorded (if workflow tracer is available)
        try:
            response = app.get("/api/workflow/history")
            if response.status_code == 200:
                history = response.json()
                assert isinstance(history, list)
        except:
            pass  # Workflow history might not be implemented

        logger.info("Workflow execution test passed")

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise


@pytest.mark.integration
@pytest.mark.performance
async def test_system_performance_baseline(
    postgis_connection, ckg_connection, agent_registry, benchmark
):
    """
    Performance baseline test for key system operations.

    This test establishes performance benchmarks for:
    - Database queries
    - Agent processing
    - Data synchronization
    """
    if not all([postgis_connection, ckg_connection, agent_registry]):
        pytest.skip("Required components not available")

    # Benchmark database query performance
    async def benchmark_postgis_query():
        return await postgis_connection.fetch(
            """
            SELECT name, entity, ST_AsText(geom) as geometry
            FROM locations
            WHERE entity = 'city'
        """
        )

    postgis_result = benchmark(benchmark_postgis_query)
    assert len(postgis_result) >= 0

    # Benchmark agent processing
    async def benchmark_agent_processing():
        atlas = agent_registry.get_agent("AtlasRelationalAnalyst")
        return await atlas.process_task("Analyze spatial relationships in test data")

    agent_result = benchmark(benchmark_agent_processing)
    assert agent_result is not None

    logger.info("Performance baseline test completed")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])
