"""
Performance benchmarks for Terra Constellata system.

This module provides comprehensive performance testing including:
- Database query performance
- Agent processing speed
- API response times
- Memory usage profiling
- Load testing scenarios
"""

import pytest
import asyncio
import time
import psutil
import os
from typing import Dict, Any, List
import logging
from statistics import mean, median, stdev

logger = logging.getLogger(__name__)


@pytest.mark.performance
@pytest.mark.asyncio
async def test_database_query_performance(
    postgis_connection, ckg_connection, benchmark
):
    """Benchmark database query performance."""
    if not all([postgis_connection, ckg_connection]):
        pytest.skip("Database connections not available")

    # Setup test data
    await setup_performance_test_data(postgis_connection, ckg_connection)

    # Benchmark PostGIS spatial queries
    async def benchmark_postgis_spatial():
        return await postgis_connection.fetch(
            """
            SELECT name, entity, ST_AsText(geom) as geometry,
                   ST_Distance(geom, ST_SetSRID(ST_MakePoint(0, 0), 4326)) as distance
            FROM locations
            ORDER BY distance
            LIMIT 100
        """
        )

    postgis_result = benchmark(benchmark_postgis_spatial)
    assert len(postgis_result) > 0
    logger.info(f"PostGIS spatial query: {len(postgis_result)} results")

    # Benchmark CKG graph queries
    async def benchmark_ckg_graph():
        return await ckg_connection.execute_aql(
            """
            FOR entity IN geospatial_entities
            FOR relation IN location_myth_relations
                FILTER relation._from == CONCAT('geospatial_entities/', entity._key)
                FOR myth IN mythological_narratives
                    FILTER relation._to == CONCAT('mythological_narratives/', myth._key)
                    RETURN {
                        location: entity.name,
                        myth: myth.myth,
                        culture: myth.culture
                    }
        """
        )

    ckg_result = benchmark(benchmark_ckg_graph)
    assert len(ckg_result) > 0
    logger.info(f"CKG graph query: {len(ckg_result)} results")


@pytest.mark.performance
@pytest.mark.asyncio
async def test_agent_processing_performance(agent_registry, mock_llm, benchmark):
    """Benchmark agent processing performance."""
    if not agent_registry:
        pytest.skip("Agent registry not available")

    # Prepare test tasks
    test_tasks = [
        "Analyze spatial relationships in geographical data",
        "Compare mythological narratives across cultures",
        "Process linguistic patterns in text data",
        "Coordinate multi-agent workflow execution",
        "Learn from example data patterns",
    ]

    # Benchmark individual agent processing
    async def benchmark_individual_agent_processing():
        results = []
        for agent_name in agent_registry.list_agents():
            agent = agent_registry.get_agent(agent_name)
            if agent:
                for task in test_tasks:
                    start_time = time.time()
                    result = await agent.process_task(task)
                    end_time = time.time()
                    results.append(
                        {
                            "agent": agent_name,
                            "task": task,
                            "duration": end_time - start_time,
                            "result_length": len(result) if result else 0,
                        }
                    )
        return results

    results = benchmark(benchmark_individual_agent_processing)
    assert len(results) > 0

    # Analyze performance statistics
    durations = [r["duration"] for r in results]
    logger.info(
        f"Agent processing stats - Mean: {mean(durations):.3f}s, "
        f"Median: {median(durations):.3f}s, "
        f"StdDev: {stdev(durations):.3f}s"
    )


@pytest.mark.performance
@pytest.mark.asyncio
async def test_api_response_performance(backend_app, benchmark):
    """Benchmark API response performance."""
    if not backend_app:
        pytest.skip("Backend app not available")

    # Benchmark health endpoint
    def benchmark_health_endpoint():
        response = backend_app.get("/health")
        return response.status_code, response.json()

    result = benchmark(benchmark_health_endpoint)
    assert result[0] == 200
    assert result[1]["status"] == "healthy"

    # Benchmark root endpoint
    def benchmark_root_endpoint():
        response = backend_app.get("/")
        return response.status_code, response.json()

    result = benchmark(benchmark_root_endpoint)
    assert result[0] == 200

    logger.info("API response performance test completed")


@pytest.mark.performance
@pytest.mark.asyncio
async def test_concurrent_load_performance(
    postgis_connection, agent_registry, benchmark
):
    """Test system performance under concurrent load."""
    if not all([postgis_connection, agent_registry]):
        pytest.skip("Required components not available")

    async def concurrent_database_queries(num_queries: int):
        """Execute concurrent database queries."""
        tasks = []
        for i in range(num_queries):
            task = asyncio.create_task(
                postgis_connection.fetch("SELECT COUNT(*) FROM locations")
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

    async def concurrent_agent_processing(num_tasks: int):
        """Execute concurrent agent processing tasks."""
        tasks = []
        agents = agent_registry.list_agents()

        for i in range(num_tasks):
            agent_name = agents[i % len(agents)]
            agent = agent_registry.get_agent(agent_name)
            if agent:
                task = asyncio.create_task(agent.process_task(f"Concurrent task {i}"))
                tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

    # Benchmark concurrent operations
    async def benchmark_concurrent_operations():
        # Concurrent database queries
        db_results = await concurrent_database_queries(10)

        # Concurrent agent processing
        agent_results = await concurrent_agent_processing(10)

        return db_results, agent_results

    results = benchmark(benchmark_concurrent_operations)
    db_results, agent_results = results

    assert len(db_results) == 10
    assert len(agent_results) == 10

    logger.info("Concurrent load performance test completed")


@pytest.mark.performance
async def test_memory_usage_profiling(postgis_connection, agent_registry):
    """Profile memory usage during system operations."""
    if not all([postgis_connection, agent_registry]):
        pytest.skip("Required components not available")

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Perform memory-intensive operations
    memory_usage_points = []

    # 1. Load data into memory
    locations = await postgis_connection.fetch(
        """
        SELECT name, entity, latitude, longitude, description
        FROM locations
        LIMIT 1000
    """
    )
    memory_usage_points.append(
        ("after_data_load", process.memory_info().rss / 1024 / 1024)
    )

    # 2. Process data with agents
    for agent_name in agent_registry.list_agents()[:2]:  # Limit to 2 agents
        agent = agent_registry.get_agent(agent_name)
        if agent:
            for location in locations[:10]:  # Process subset
                await agent.process_task(f"Analyze {location['name']}")
            memory_usage_points.append(
                (f"after_{agent_name}", process.memory_info().rss / 1024 / 1024)
            )

    # 3. Clean up
    del locations
    memory_usage_points.append(
        ("after_cleanup", process.memory_info().rss / 1024 / 1024)
    )

    # Analyze memory usage
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = final_memory - initial_memory

    logger.info(
        f"Memory profiling - Initial: {initial_memory:.2f}MB, "
        f"Final: {final_memory:.2f}MB, "
        f"Increase: {memory_increase:.2f}MB"
    )

    # Log memory usage points
    for point_name, memory_mb in memory_usage_points:
        logger.info(f"Memory at {point_name}: {memory_mb:.2f}MB")

    # Assert memory usage is reasonable (less than 500MB increase)
    assert memory_increase < 500, f"Memory increase too high: {memory_increase:.2f}MB"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_end_to_end_workflow_performance(
    agent_registry, backend_app, sample_data, benchmark
):
    """Test end-to-end workflow performance."""
    if not all([agent_registry, backend_app]):
        pytest.skip("Required components not available")

    async def benchmark_full_workflow():
        """Benchmark complete workflow from data to results."""
        start_time = time.time()

        # Step 1: Data processing
        data_processing_start = time.time()
        processed_data = sample_data.copy()
        data_processing_time = time.time() - data_processing_start

        # Step 2: Agent processing
        agent_processing_start = time.time()
        agent_results = {}
        for agent_name in agent_registry.list_agents():
            agent = agent_registry.get_agent(agent_name)
            if agent:
                result = await agent.process_task(
                    f"Process data for {agent_name}: {processed_data}"
                )
                agent_results[agent_name] = result
        agent_processing_time = time.time() - agent_processing_start

        # Step 3: API integration
        api_start = time.time()
        try:
            response = backend_app.get("/health")
            api_success = response.status_code == 200
        except:
            api_success = False
        api_time = time.time() - api_start

        total_time = time.time() - start_time

        return {
            "total_time": total_time,
            "data_processing_time": data_processing_time,
            "agent_processing_time": agent_processing_time,
            "api_time": api_time,
            "api_success": api_success,
            "agent_results_count": len(agent_results),
        }

    result = benchmark(benchmark_full_workflow)

    assert result["total_time"] > 0
    assert result["agent_results_count"] > 0

    logger.info(f"End-to-end workflow performance: {result['total_time']:.3f}s")
    logger.info(f"  - Data processing: {result['data_processing_time']:.3f}s")
    logger.info(f"  - Agent processing: {result['agent_processing_time']:.3f}s")
    logger.info(f"  - API calls: {result['api_time']:.3f}s")


@pytest.mark.performance
@pytest.mark.asyncio
async def test_scalability_under_load(postgis_connection, agent_registry):
    """Test system scalability under increasing load."""
    if not all([postgis_connection, agent_registry]):
        pytest.skip("Required components not available")

    load_levels = [1, 5, 10, 20]  # Concurrent operations
    performance_results = []

    for load_level in load_levels:
        logger.info(f"Testing scalability at load level: {load_level}")

        start_time = time.time()

        # Create concurrent database queries
        db_tasks = []
        for i in range(load_level):
            task = asyncio.create_task(
                postgis_connection.fetch("SELECT COUNT(*) FROM locations")
            )
            db_tasks.append(task)

        # Create concurrent agent tasks
        agent_tasks = []
        agents = agent_registry.list_agents()
        for i in range(load_level):
            agent_name = agents[i % len(agents)]
            agent = agent_registry.get_agent(agent_name)
            if agent:
                task = asyncio.create_task(agent.process_task(f"Load test task {i}"))
                agent_tasks.append(task)

        # Execute all tasks
        db_results, agent_results = await asyncio.gather(
            asyncio.gather(*db_tasks), asyncio.gather(*agent_tasks)
        )

        total_time = time.time() - start_time

        performance_results.append(
            {
                "load_level": load_level,
                "total_time": total_time,
                "avg_time_per_operation": total_time
                / (len(db_tasks) + len(agent_tasks)),
                "db_operations": len(db_results),
                "agent_operations": len(agent_results),
            }
        )

        logger.info(
            f"Load level {load_level}: {total_time:.3f}s total, "
            f"{total_time/(len(db_tasks) + len(agent_tasks)):.3f}s per operation"
        )

    # Analyze scalability
    for i in range(1, len(performance_results)):
        prev = performance_results[i - 1]
        curr = performance_results[i]

        # Calculate scaling efficiency
        expected_time = prev["avg_time_per_operation"] * (
            curr["load_level"] / prev["load_level"]
        )
        actual_time = curr["avg_time_per_operation"]
        scaling_efficiency = expected_time / actual_time

        logger.info(
            f"Scaling from {prev['load_level']} to {curr['load_level']}: "
            f"Efficiency = {scaling_efficiency:.2f}"
        )


async def setup_performance_test_data(postgis_connection, ckg_connection):
    """Setup test data for performance benchmarks."""
    # Create test locations
    test_locations = []
    for i in range(1000):
        test_locations.append(
            {
                "name": f"PerfTestLocation{i}",
                "entity": "test_location",
                "latitude": 40.0 + (i % 100) * 0.01,
                "longitude": -74.0 + (i % 100) * 0.01,
                "description": f"Performance test location {i}",
            }
        )

    # Insert into PostGIS
    for loc in test_locations:
        await postgis_connection.execute(
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

    # Insert subset into CKG
    for loc in test_locations[:100]:
        await ckg_connection.execute_aql(
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

    # Create relationships in CKG
    for i in range(10):
        await ckg_connection.execute_aql(
            """
            INSERT {
                culture: @culture,
                myth: @myth,
                narrative: @narrative,
                type: 'mythological_narrative'
            } INTO mythological_narratives
            OPTIONS { ignoreErrors: true }
        """,
            culture=f"TestCulture{i}",
            myth=f"TestMyth{i}",
            narrative=f"Test narrative {i}",
        )

    logger.info("Performance test data setup completed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short", "-m", "performance"])
