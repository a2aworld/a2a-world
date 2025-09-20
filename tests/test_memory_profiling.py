"""
Memory profiling and optimization tests for Terra Constellata.

This module provides comprehensive memory profiling including:
- Memory usage tracking during operations
- Memory leak detection
- Garbage collection analysis
- Memory optimization recommendations
"""

import pytest
import asyncio
import time
import gc
import psutil
import os
import sys
from typing import Dict, Any, List
import logging
import tracemalloc
from memory_profiler import profile as memory_profile

logger = logging.getLogger(__name__)


@pytest.mark.performance
async def test_memory_usage_baseline(postgis_connection, agent_registry):
    """Establish memory usage baseline for system components."""
    if not all([postgis_connection, agent_registry]):
        pytest.skip("Required components not available")

    process = psutil.Process(os.getpid())

    # Start memory tracing
    tracemalloc.start()

    # Baseline memory measurement
    gc.collect()  # Force garbage collection
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    baseline_tracemalloc = tracemalloc.get_traced_memory()

    logger.info(
        f"Baseline memory - RSS: {baseline_memory:.2f}MB, "
        f"Tracemalloc: {baseline_tracemalloc[0]/1024/1024:.2f}MB"
    )

    # Test database operations memory usage
    await test_database_memory_usage(postgis_connection, baseline_memory)

    # Test agent operations memory usage
    await test_agent_memory_usage(agent_registry, baseline_memory)

    # Test combined operations memory usage
    await test_combined_operations_memory(
        postgis_connection, agent_registry, baseline_memory
    )

    # Stop memory tracing
    tracemalloc.stop()


@pytest.mark.performance
async def test_memory_leak_detection(postgis_connection, agent_registry):
    """Test for memory leaks during repeated operations."""
    if not all([postgis_connection, agent_registry]):
        pytest.skip("Required components not available")

    process = psutil.Process(os.getpid())
    tracemalloc.start()

    # Record initial memory
    initial_memory = process.memory_info().rss / 1024 / 1024

    # Perform repeated operations to detect leaks
    memory_snapshots = []

    for iteration in range(10):
        # Force garbage collection
        gc.collect()

        # Record memory before operations
        snapshot = tracemalloc.take_snapshot()
        memory_snapshots.append(snapshot)

        # Perform database operations
        for i in range(50):
            await postgis_connection.fetch("SELECT COUNT(*) FROM locations")

        # Perform agent operations
        for agent_name in agent_registry.list_agents()[:2]:
            agent = agent_registry.get_agent(agent_name)
            if agent:
                await agent.process_task(f"Leak test task {iteration}-{agent_name}")

        # Small delay
        await asyncio.sleep(0.1)

    # Analyze memory snapshots for leaks
    if len(memory_snapshots) >= 2:
        stats = memory_snapshots[-1].compare_to(memory_snapshots[0], "lineno")

        # Log top memory differences
        for stat in stats[:10]:
            if stat.size_diff > 0:  # Only show increases
                logger.info(
                    f"Memory increase: +{stat.size_diff/1024:.1f}KB at {stat.traceback.format()[-1]}"
                )

    # Check for significant memory growth
    final_memory = process.memory_info().rss / 1024 / 1024
    initial_memory_value = initial_memory

    # Calculate memory growth rate
    memory_growth = (final_memory - initial_memory_value) / len(memory_snapshots)
    logger.info(f"Memory growth per iteration: {memory_growth:.2f}MB")

    # Assert reasonable memory growth (less than 10MB per iteration)
    assert (
        memory_growth < 10
    ), f"Potential memory leak detected: {memory_growth:.2f}MB per iteration"

    tracemalloc.stop()


@pytest.mark.performance
async def test_garbage_collection_efficiency(postgis_connection, agent_registry):
    """Test garbage collection efficiency and object lifecycle."""
    if not all([postgis_connection, agent_registry]):
        pytest.skip("Required components not available")

    # Enable garbage collection debugging
    gc.set_debug(gc.DEBUG_STATS)

    # Track object counts
    initial_objects = len(gc.get_objects())

    # Perform operations that create objects
    await perform_memory_intensive_operations(postgis_connection, agent_registry)

    # Force garbage collection and measure
    collected = gc.collect()
    final_objects = len(gc.get_objects())

    logger.info(
        f"Garbage collection - Collected: {collected} objects, "
        f"Objects before: {initial_objects}, after: {final_objects}"
    )

    # Check for object accumulation
    object_growth = final_objects - initial_objects
    if object_growth > 1000:  # Allow some growth
        logger.warning(f"Significant object accumulation: {object_growth}")

    # Disable GC debugging
    gc.set_debug(0)


@pytest.mark.performance
async def test_memory_optimization_recommendations(postgis_connection, agent_registry):
    """Generate memory optimization recommendations based on profiling."""
    if not all([postgis_connection, agent_registry]):
        pytest.skip("Required components not available")

    recommendations = []

    # Analyze database connection pooling
    if hasattr(postgis_connection, "_pool"):
        pool_size = getattr(postgis_connection._pool, "size", "unknown")
        recommendations.append(f"Database connection pool size: {pool_size}")

    # Analyze agent memory usage patterns
    agent_memory_usage = {}
    for agent_name in agent_registry.list_agents():
        agent = agent_registry.get_agent(agent_name)
        if agent:
            # Get agent object size (approximate)
            agent_size = sys.getsizeof(agent)
            agent_memory_usage[agent_name] = agent_size

    if agent_memory_usage:
        largest_agent = max(agent_memory_usage.items(), key=lambda x: x[1])
        recommendations.append(
            f"Largest agent by memory: {largest_agent[0]} ({largest_agent[1]} bytes)"
        )

    # Check for large data structures
    large_objects = []
    for obj in gc.get_objects():
        size = sys.getsizeof(obj)
        if size > 1000000:  # Objects larger than 1MB
            large_objects.append((type(obj).__name__, size))

    if large_objects:
        recommendations.append(f"Large objects detected: {len(large_objects)}")
        for obj_type, size in large_objects[:5]:  # Show top 5
            recommendations.append(f"  - {obj_type}: {size/1024/1024:.2f}MB")

    # Log recommendations
    logger.info("Memory optimization recommendations:")
    for rec in recommendations:
        logger.info(f"  {rec}")

    return recommendations


@memory_profile
async def test_database_memory_usage(postgis_connection, baseline_memory):
    """Profile memory usage during database operations."""
    process = psutil.Process(os.getpid())

    # Test various database operations
    operations = [
        (
            "Simple query",
            lambda: postgis_connection.fetch("SELECT COUNT(*) FROM locations"),
        ),
        (
            "Spatial query",
            lambda: postgis_connection.fetch(
                """
            SELECT name, ST_AsText(geom) FROM locations LIMIT 100
        """
            ),
        ),
        (
            "Complex query",
            lambda: postgis_connection.fetch(
                """
            SELECT l1.name, l2.name, ST_Distance(l1.geom, l2.geom) as distance
            FROM locations l1, locations l2
            WHERE ST_DWithin(l1.geom, l2.geom, 1000) AND l1.name != l2.name
            LIMIT 50
        """
            ),
        ),
    ]

    for op_name, op_func in operations:
        start_memory = process.memory_info().rss / 1024 / 1024
        start_time = time.time()

        await op_func()

        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024

        memory_delta = end_memory - start_memory
        duration = end_time - start_time

        logger.info(f"{op_name} - Memory: {memory_delta:+.2f}MB, Time: {duration:.3f}s")


@memory_profile
async def test_agent_memory_usage(agent_registry, baseline_memory):
    """Profile memory usage during agent operations."""
    process = psutil.Process(os.getpid())

    for agent_name in agent_registry.list_agents():
        agent = agent_registry.get_agent(agent_name)
        if agent:
            start_memory = process.memory_info().rss / 1024 / 1024
            start_time = time.time()

            # Perform agent task
            result = await agent.process_task(f"Memory profiling task for {agent_name}")

            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024

            memory_delta = end_memory - start_memory
            duration = end_time - start_time

            logger.info(
                f"Agent {agent_name} - Memory: {memory_delta:+.2f}MB, "
                f"Time: {duration:.3f}s, Result size: {len(str(result))}"
            )


@memory_profile
async def test_combined_operations_memory(
    postgis_connection, agent_registry, baseline_memory
):
    """Profile memory usage during combined operations."""
    process = psutil.Process(os.getpid())

    start_memory = process.memory_info().rss / 1024 / 1024
    start_time = time.time()

    # Perform combined operations
    # 1. Database query
    db_results = await postgis_connection.fetch(
        """
        SELECT name, entity, latitude, longitude FROM locations LIMIT 50
    """
    )

    # 2. Agent processing on results
    agent_results = []
    for agent_name in agent_registry.list_agents()[:2]:  # Limit to 2 agents
        agent = agent_registry.get_agent(agent_name)
        if agent:
            result = await agent.process_task(
                f"Process {len(db_results)} database results"
            )
            agent_results.append(result)

    # 3. Cross-agent coordination (if Sentinel available)
    sentinel = agent_registry.get_agent("SentinelOrchestrator")
    if sentinel:
        coord_result = await sentinel.coordinate_agents(
            "Analyze combined database and agent results"
        )

    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024

    memory_delta = end_memory - start_memory
    duration = end_time - start_time

    logger.info(
        f"Combined operations - Memory: {memory_delta:+.2f}MB, "
        f"Time: {duration:.3f}s, DB results: {len(db_results)}, "
        f"Agent results: {len(agent_results)}"
    )


async def perform_memory_intensive_operations(postgis_connection, agent_registry):
    """Perform operations designed to stress memory usage."""
    # Create many database connections/results
    db_tasks = []
    for i in range(20):
        task = asyncio.create_task(
            postgis_connection.fetch(
                """
                SELECT name, entity, latitude, longitude, description
                FROM locations LIMIT 100
            """
            )
        )
        db_tasks.append(task)

    db_results = await asyncio.gather(*db_tasks)

    # Process results with agents
    agent_tasks = []
    for agent_name in agent_registry.list_agents():
        agent = agent_registry.get_agent(agent_name)
        if agent:
            for i in range(5):
                task = asyncio.create_task(
                    agent.process_task(
                        f"Process batch {i} of {len(db_results)} results"
                    )
                )
                agent_tasks.append(task)

    await asyncio.gather(*agent_tasks)

    # Create some temporary objects
    temp_objects = []
    for i in range(1000):
        temp_objects.append(
            {
                "id": i,
                "data": "x" * 1000,  # 1KB per object
                "results": db_results[i % len(db_results)] if db_results else None,
            }
        )

    # Use objects briefly then discard
    total_size = sum(len(str(obj)) for obj in temp_objects)
    logger.info(
        f"Created {len(temp_objects)} temporary objects, total size: {total_size/1024:.1f}KB"
    )

    # Objects go out of scope here


@pytest.mark.performance
async def test_memory_cleanup_efficiency(postgis_connection, agent_registry):
    """Test efficiency of memory cleanup after operations."""
    if not all([postgis_connection, agent_registry]):
        pytest.skip("Required components not available")

    process = psutil.Process(os.getpid())

    # Perform memory-intensive operations
    await perform_memory_intensive_operations(postgis_connection, agent_registry)

    # Measure memory before cleanup
    before_cleanup = process.memory_info().rss / 1024 / 1024

    # Force garbage collection
    collected = gc.collect()

    # Measure memory after cleanup
    after_cleanup = process.memory_info().rss / 1024 / 1024

    memory_freed = before_cleanup - after_cleanup

    logger.info(
        f"Memory cleanup - Before: {before_cleanup:.2f}MB, "
        f"After: {after_cleanup:.2f}MB, Freed: {memory_freed:.2f}MB, "
        f"Objects collected: {collected}"
    )

    # Assert reasonable cleanup efficiency
    cleanup_efficiency = memory_freed / before_cleanup if before_cleanup > 0 else 0
    assert (
        cleanup_efficiency > 0.1
    ), f"Poor cleanup efficiency: {cleanup_efficiency:.2%}"


@pytest.mark.performance
async def test_large_dataset_memory_handling(postgis_connection):
    """Test memory handling with large datasets."""
    if not postgis_connection:
        pytest.skip("PostGIS connection not available")

    process = psutil.Process(os.getpid())

    # Test with increasingly large result sets
    sizes = [100, 500, 1000, 5000]

    for size in sizes:
        start_memory = process.memory_info().rss / 1024 / 1024
        start_time = time.time()

        # Query large dataset
        results = await postgis_connection.fetch(
            f"""
            SELECT name, entity, latitude, longitude, description
            FROM locations
            LIMIT {size}
        """
        )

        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024

        memory_delta = end_memory - start_memory
        duration = end_time - start_time

        memory_per_record = memory_delta / len(results) if results else 0

        logger.info(
            f"Dataset size {size} - Memory: {memory_delta:+.2f}MB, "
            f"Time: {duration:.3f}s, Memory per record: {memory_per_record:.4f}MB"
        )

        # Force cleanup between tests
        del results
        gc.collect()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short", "-m", "performance"])
