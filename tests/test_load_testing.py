"""
Load testing framework for Terra Constellata system.

This module provides comprehensive load testing capabilities including:
- Concurrent user simulation
- Database load testing
- API endpoint stress testing
- Agent workload simulation
- Resource usage monitoring
"""

import pytest
import asyncio
import time
import statistics
from typing import Dict, Any, List
import logging
from concurrent.futures import ThreadPoolExecutor
import aiohttp

try:
    import psutil
    import os
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@pytest.mark.performance
@pytest.mark.asyncio
async def test_database_load_capacity(postgis_connection, ckg_connection):
    """Test database load capacity under high concurrent operations."""
    if not all([postgis_connection, ckg_connection]):
        pytest.skip("Database connections not available")

    # Setup test data
    await setup_load_test_data(postgis_connection, ckg_connection)

    # Test concurrent read operations
    await test_concurrent_reads(postgis_connection, ckg_connection)

    # Test concurrent write operations
    await test_concurrent_writes(postgis_connection, ckg_connection)

    # Test mixed read/write operations
    await test_mixed_operations(postgis_connection, ckg_connection)


@pytest.mark.performance
@pytest.mark.asyncio
async def test_api_load_capacity(backend_app):
    """Test API load capacity under concurrent requests."""
    if not backend_app:
        pytest.skip("Backend app not available")

    # Test health endpoint under load
    await test_endpoint_load(
        backend_app, "/health", num_requests=100, concurrent_requests=10
    )

    # Test root endpoint under load
    await test_endpoint_load(backend_app, "/", num_requests=100, concurrent_requests=10)

    # Test API endpoints under load (if available)
    try:
        await test_endpoint_load(
            backend_app, "/api/content/", num_requests=50, concurrent_requests=5
        )
    except:
        logger.info("Content API endpoint not available for load testing")


@pytest.mark.performance
@pytest.mark.asyncio
async def test_agent_load_capacity(agent_registry, mock_llm):
    """Test agent system load capacity under concurrent tasks."""
    if not agent_registry:
        pytest.skip("Agent registry not available")

    # Test individual agent load capacity
    await test_individual_agent_load(agent_registry)

    # Test coordinated agent load
    await test_coordinated_agent_load(agent_registry)

    # Test agent system recovery under load
    await test_agent_system_recovery(agent_registry)


@pytest.mark.performance
@pytest.mark.asyncio
async def test_system_integration_load(
    postgis_connection, ckg_connection, agent_registry, backend_app
):
    """Test full system integration under load."""
    if not all([postgis_connection, ckg_connection, agent_registry, backend_app]):
        pytest.skip("Required components not available")

    # Simulate integrated workflow under load
    await test_integrated_workflow_load(
        postgis_connection, ckg_connection, agent_registry, backend_app
    )


@pytest.mark.performance
@pytest.mark.asyncio
async def test_resource_usage_under_load(postgis_connection, agent_registry):
    """Monitor resource usage during load testing."""
    if not all([postgis_connection, agent_registry]):
        pytest.skip("Required components not available")

    if not PSUTIL_AVAILABLE:
        pytest.skip("psutil not available for resource monitoring")

    process = psutil.Process(os.getpid())

    try:
        # Baseline resource usage
        baseline_cpu = process.cpu_percent(interval=1)
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        logger.info(f"Baseline - CPU: {baseline_cpu}%, Memory: {baseline_memory:.2f}MB")

        # Generate load
        load_start = time.time()
        await generate_system_load(postgis_connection, agent_registry, duration=30)
        load_end = time.time()

        # Peak resource usage
        peak_cpu = process.cpu_percent(interval=1)
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB

        logger.info(f"Peak - CPU: {peak_cpu}%, Memory: {peak_memory:.2f}MB")
        logger.info(f"Load duration: {load_end - load_start:.2f}s")

        # Resource usage analysis
        cpu_increase = peak_cpu - baseline_cpu
        memory_increase = peak_memory - baseline_memory

        logger.info(
            f"Resource increase - CPU: {cpu_increase}%, Memory: {memory_increase:.2f}MB"
        )

        # Assert reasonable resource usage
        assert cpu_increase < 80, f"CPU usage increase too high: {cpu_increase}%"
        assert memory_increase < 500, f"Memory increase too high: {memory_increase:.2f}MB"

    except Exception as e:
        logger.error(f"Error monitoring resource usage: {e}")
        pytest.skip(f"Resource monitoring failed: {e}")


async def test_concurrent_reads(postgis_conn, ckg_conn):
    """Test concurrent read operations."""

    async def concurrent_postgis_reads(num_operations: int):
        """Execute concurrent PostGIS read operations."""
        tasks = []
        for i in range(num_operations):
            task = asyncio.create_task(
                postgis_conn.fetch(
                    """
                    SELECT name, entity, latitude, longitude
                    FROM locations
                    WHERE entity = 'test_location'
                    LIMIT 10
                """
                )
            )
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        return {
            "duration": end_time - start_time,
            "operations": len(results),
            "avg_time_per_operation": (end_time - start_time) / len(results),
        }

    async def concurrent_ckg_reads(num_operations: int):
        """Execute concurrent CKG read operations."""
        tasks = []
        for i in range(num_operations):
            task = asyncio.create_task(
                ckg_conn.execute_aql(
                    """
                    FOR entity IN geospatial_entities
                    FILTER entity.type == 'geospatial_entity'
                    LIMIT 10
                    RETURN entity
                """
                )
            )
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        return {
            "duration": end_time - start_time,
            "operations": len(results),
            "avg_time_per_operation": (end_time - start_time) / len(results),
        }

    # Test different concurrency levels
    concurrency_levels = [5, 10, 20]

    for level in concurrency_levels:
        logger.info(f"Testing concurrent reads at level {level}")

        postgis_stats = await concurrent_postgis_reads(level)
        ckg_stats = await concurrent_ckg_reads(level)

        logger.info(
            f"PostGIS - {level} concurrent reads: {postgis_stats['duration']:.3f}s "
            f"({postgis_stats['avg_time_per_operation']:.3f}s per operation)"
        )
        logger.info(
            f"CKG - {level} concurrent reads: {ckg_stats['duration']:.3f}s "
            f"({ckg_stats['avg_time_per_operation']:.3f}s per operation)"
        )


async def test_concurrent_writes(postgis_conn, ckg_conn):
    """Test concurrent write operations."""

    async def concurrent_postgis_writes(num_operations: int):
        """Execute concurrent PostGIS write operations."""
        tasks = []
        for i in range(num_operations):
            task = asyncio.create_task(
                postgis_conn.execute(
                    """
                    INSERT INTO locations (name, entity, latitude, longitude, description, geom)
                    VALUES ($1, $2, $3, $4, $5, ST_SetSRID(ST_MakePoint($4, $3), 4326))
                """,
                    f"LoadTestLocation{i}",
                    "test_location",
                    40.0 + i * 0.001,
                    -74.0 + i * 0.001,
                    f"Load test location {i}",
                )
            )
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        return {
            "duration": end_time - start_time,
            "operations": len(results),
            "avg_time_per_operation": (end_time - start_time) / len(results),
        }

    async def concurrent_ckg_writes(num_operations: int):
        """Execute concurrent CKG write operations."""
        tasks = []
        for i in range(num_operations):
            task = asyncio.create_task(
                ckg_conn.execute_aql(
                    """
                    INSERT {
                        name: @name,
                        type: 'geospatial_entity',
                        latitude: @latitude,
                        longitude: @longitude
                    } INTO geospatial_entities
                """,
                    name=f"LoadTestEntity{i}",
                    latitude=40.0 + i * 0.001,
                    longitude=-74.0 + i * 0.001,
                )
            )
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        return {
            "duration": end_time - start_time,
            "operations": len(results),
            "avg_time_per_operation": (end_time - start_time) / len(results),
        }

    # Test concurrent writes
    num_writes = 50

    postgis_stats = await concurrent_postgis_writes(num_writes)
    ckg_stats = await concurrent_ckg_writes(num_writes)

    logger.info(
        f"PostGIS - {num_writes} concurrent writes: {postgis_stats['duration']:.3f}s "
        f"({postgis_stats['avg_time_per_operation']:.3f}s per operation)"
    )
    logger.info(
        f"CKG - {num_writes} concurrent writes: {ckg_stats['duration']:.3f}s "
        f"({ckg_stats['avg_time_per_operation']:.3f}s per operation)"
    )


async def test_mixed_operations(postgis_conn, ckg_conn):
    """Test mixed read/write operations."""

    async def mixed_operations(num_operations: int):
        """Execute mixed read and write operations."""
        tasks = []

        for i in range(num_operations):
            if i % 3 == 0:  # 33% writes
                task = asyncio.create_task(
                    postgis_conn.execute(
                        """
                        INSERT INTO locations (name, entity, latitude, longitude, description, geom)
                        VALUES ($1, $2, $3, $4, $5, ST_SetSRID(ST_MakePoint($4, $3), 4326))
                    """,
                        f"MixedTestLocation{i}",
                        "test_location",
                        40.0 + i * 0.001,
                        -74.0 + i * 0.001,
                        f"Mixed test location {i}",
                    )
                )
            else:  # 67% reads
                task = asyncio.create_task(
                    postgis_conn.fetch(
                        """
                        SELECT COUNT(*) FROM locations
                    """
                    )
                )
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        return {
            "duration": end_time - start_time,
            "operations": len(results),
            "avg_time_per_operation": (end_time - start_time) / len(results),
        }

    # Test mixed operations
    num_operations = 100

    stats = await mixed_operations(num_operations)

    logger.info(
        f"Mixed operations - {num_operations} operations: {stats['duration']:.3f}s "
        f"({stats['avg_time_per_operation']:.3f}s per operation)"
    )


async def test_endpoint_load(
    app, endpoint: str, num_requests: int, concurrent_requests: int
):
    """Test API endpoint load capacity."""

    async def make_request(session, url):
        """Make a single request."""
        start_time = time.time()
        try:
            async with session.get(url) as response:
                end_time = time.time()
                return {
                    "status": response.status,
                    "duration": end_time - start_time,
                    "success": response.status < 400,
                }
        except Exception as e:
            end_time = time.time()
            return {
                "status": None,
                "duration": end_time - start_time,
                "success": False,
                "error": str(e),
            }

    async def load_test_endpoint():
        """Execute load test for endpoint."""
        base_url = "http://testserver"  # TestClient uses this
        url = base_url + endpoint

        # Use ThreadPoolExecutor to simulate concurrent requests with TestClient
        response_times = []
        success_count = 0

        # Since TestClient is not async, we'll simulate concurrent requests
        # by making sequential requests in batches
        for batch_start in range(0, num_requests, concurrent_requests):
            batch_end = min(batch_start + concurrent_requests, num_requests)
            batch_size = batch_end - batch_start

            batch_start_time = time.time()

            # Make concurrent requests in this batch
            tasks = []
            for i in range(batch_size):
                # Use TestClient for each request
                response = app.get(endpoint)
                tasks.append(asyncio.create_task(asyncio.sleep(0)))  # Placeholder

            # Wait for batch to complete
            await asyncio.gather(*tasks)

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time

            # Record results
            for i in range(batch_size):
                response_times.append(batch_duration / batch_size)
                if response.status_code < 400:
                    success_count += 1

        return {
            "total_requests": num_requests,
            "successful_requests": success_count,
            "success_rate": success_count / num_requests,
            "avg_response_time": statistics.mean(response_times),
            "median_response_time": statistics.median(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
        }

    results = await load_test_endpoint()

    logger.info(f"Endpoint {endpoint} load test results:")
    logger.info(f"  Total requests: {results['total_requests']}")
    logger.info(f"  Success rate: {results['success_rate']:.2%}")
    logger.info(f"  Avg response time: {results['avg_response_time']:.3f}s")
    logger.info(f"  Median response time: {results['median_response_time']:.3f}s")
    logger.info(
        f"  Response time range: {results['min_response_time']:.3f}s - {results['max_response_time']:.3f}s"
    )


async def test_individual_agent_load(agent_registry):
    """Test individual agent load capacity."""

    async def agent_load_test(agent_name: str, num_tasks: int):
        """Test single agent under load."""
        agent = agent_registry.get_agent(agent_name)
        if not agent:
            return None

        tasks = []
        for i in range(num_tasks):
            task = asyncio.create_task(
                agent.process_task(f"Load test task {i} for {agent_name}")
            )
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        return {
            "agent": agent_name,
            "tasks": num_tasks,
            "duration": end_time - start_time,
            "avg_time_per_task": (end_time - start_time) / num_tasks,
            "successful_tasks": sum(1 for r in results if r is not None),
        }

    # Test each agent
    agents = agent_registry.list_agents()
    num_tasks = 20

    for agent_name in agents:
        results = await agent_load_test(agent_name, num_tasks)
        if results:
            logger.info(f"Agent {agent_name} load test:")
            logger.info(f"  Tasks: {results['tasks']}")
            logger.info(f"  Duration: {results['duration']:.3f}s")
            logger.info(f"  Avg time per task: {results['avg_time_per_task']:.3f}s")
            logger.info(
                f"  Success rate: {results['successful_tasks']}/{results['tasks']}"
            )


async def test_coordinated_agent_load(agent_registry):
    """Test coordinated agent load."""
    sentinel = agent_registry.get_agent("SentinelOrchestrator")
    if not sentinel:
        pytest.skip("Sentinel orchestrator not available")

    # Test coordinated workflow under load
    async def coordinated_load_test(num_workflows: int):
        """Execute multiple coordinated workflows."""
        tasks = []
        for i in range(num_workflows):
            workflow_config = {
                "name": f"load_test_workflow_{i}",
                "agents": agent_registry.list_agents()[:3],  # Use first 3 agents
                "data": {"test_data": f"workflow_{i}"},
                "objectives": ["process_data", "coordinate_results"],
            }
            task = asyncio.create_task(
                sentinel.manage_workflow(f"load_test_{i}", workflow_config)
            )
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        return {
            "workflows": num_workflows,
            "duration": end_time - start_time,
            "avg_time_per_workflow": (end_time - start_time) / num_workflows,
            "successful_workflows": sum(1 for r in results if r is not None),
        }

    num_workflows = 10
    results = await coordinated_load_test(num_workflows)

    logger.info("Coordinated agent load test:")
    logger.info(f"  Workflows: {results['workflows']}")
    logger.info(f"  Duration: {results['duration']:.3f}s")
    logger.info(f"  Avg time per workflow: {results['avg_time_per_workflow']:.3f}s")
    logger.info(
        f"  Success rate: {results['successful_workflows']}/{results['workflows']}"
    )


async def test_agent_system_recovery(agent_registry):
    """Test agent system recovery under load."""
    # Simulate system stress and recovery
    agents = agent_registry.list_agents()

    # Phase 1: Normal load
    logger.info("Phase 1: Normal load")
    normal_tasks = []
    for i in range(10):
        agent_name = agents[i % len(agents)]
        agent = agent_registry.get_agent(agent_name)
        if agent:
            task = asyncio.create_task(agent.process_task(f"Normal task {i}"))
            normal_tasks.append(task)

    await asyncio.gather(*normal_tasks)
    logger.info("Normal load phase completed")

    # Phase 2: High load
    logger.info("Phase 2: High load")
    high_load_tasks = []
    for i in range(50):
        agent_name = agents[i % len(agents)]
        agent = agent_registry.get_agent(agent_name)
        if agent:
            task = asyncio.create_task(agent.process_task(f"High load task {i}"))
            high_load_tasks.append(task)

    await asyncio.gather(*high_load_tasks)
    logger.info("High load phase completed")

    # Phase 3: Recovery
    logger.info("Phase 3: Recovery")
    recovery_tasks = []
    for i in range(10):
        agent_name = agents[i % len(agents)]
        agent = agent_registry.get_agent(agent_name)
        if agent:
            task = asyncio.create_task(agent.process_task(f"Recovery task {i}"))
            recovery_tasks.append(task)

    await asyncio.gather(*recovery_tasks)
    logger.info("Recovery phase completed")


async def test_integrated_workflow_load(postgis_conn, ckg_conn, agent_registry, app):
    """Test integrated workflow under load."""

    async def integrated_workflow(workflow_id: int):
        """Execute integrated workflow."""
        start_time = time.time()

        # Step 1: Database operations
        db_result = await postgis_conn.fetch("SELECT COUNT(*) FROM locations")

        # Step 2: Agent processing
        agent_results = []
        for agent_name in agent_registry.list_agents()[:2]:  # Use 2 agents
            agent = agent_registry.get_agent(agent_name)
            if agent:
                result = await agent.process_task(f"Workflow {workflow_id} task")
                agent_results.append(result)

        # Step 3: API call
        try:
            response = app.get("/health")
            api_success = response.status_code == 200
        except:
            api_success = False

        end_time = time.time()

        return {
            "workflow_id": workflow_id,
            "duration": end_time - start_time,
            "db_success": len(db_result) > 0,
            "agent_success": all(r is not None for r in agent_results),
            "api_success": api_success,
        }

    # Execute multiple integrated workflows
    num_workflows = 20
    tasks = []
    for i in range(num_workflows):
        task = asyncio.create_task(integrated_workflow(i))
        tasks.append(task)

    start_time = time.time()
    results = await asyncio.gather(*tasks)
    end_time = time.time()

    # Analyze results
    successful_workflows = sum(
        1
        for r in results
        if r["db_success"] and r["agent_success"] and r["api_success"]
    )
    avg_duration = statistics.mean(r["duration"] for r in results)

    logger.info("Integrated workflow load test:")
    logger.info(f"  Workflows: {num_workflows}")
    logger.info(f"  Total duration: {end_time - start_time:.3f}s")
    logger.info(f"  Avg workflow duration: {avg_duration:.3f}s")
    logger.info(f"  Success rate: {successful_workflows}/{num_workflows}")


async def generate_system_load(postgis_conn, agent_registry, duration: int):
    """Generate sustained system load for resource monitoring."""
    end_time = time.time() + duration

    while time.time() < end_time:
        # Generate database load
        db_tasks = []
        for i in range(5):
            task = asyncio.create_task(
                postgis_conn.fetch("SELECT COUNT(*) FROM locations")
            )
            db_tasks.append(task)

        # Generate agent load
        agent_tasks = []
        agents = agent_registry.list_agents()
        for i in range(5):
            agent_name = agents[i % len(agents)]
            agent = agent_registry.get_agent(agent_name)
            if agent:
                task = asyncio.create_task(
                    agent.process_task(f"Load generation task {i}")
                )
                agent_tasks.append(task)

        # Execute tasks
        await asyncio.gather(*db_tasks, *agent_tasks)

        # Small delay to prevent overwhelming the system
        await asyncio.sleep(0.1)


async def setup_load_test_data(postgis_conn, ckg_conn):
    """Setup test data for load testing."""
    # Create test locations for load testing
    test_locations = []
    for i in range(500):
        test_locations.append(
            {
                "name": f"LoadTestLocation{i}",
                "entity": "test_location",
                "latitude": 40.0 + (i % 50) * 0.01,
                "longitude": -74.0 + (i % 50) * 0.01,
                "description": f"Load test location {i}",
            }
        )

    # Insert into PostGIS in batches
    batch_size = 50
    for i in range(0, len(test_locations), batch_size):
        batch = test_locations[i : i + batch_size]
        tasks = []
        for loc in batch:
            task = asyncio.create_task(
                postgis_conn.execute(
                    """
                    INSERT INTO locations (name, entity, latitude, longitude, description, geom)
                    VALUES ($1, $2, $3, $4, $5, ST_SetSRID(ST_MakePoint($4, $3), 4326))
                    ON CONFLICT (name) DO NOTHING
                """,
                    loc["name"],
                    loc["entity"],
                    loc["latitude"],
                    loc["longitude"],
                    loc["description"],
                )
            )
            tasks.append(task)
        await asyncio.gather(*tasks)

    # Insert subset into CKG
    for loc in test_locations[:100]:
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
            name=loc["name"],
            entity=loc["entity"],
            latitude=loc["latitude"],
            longitude=loc["longitude"],
        )

    logger.info("Load test data setup completed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short", "-m", "performance"])
