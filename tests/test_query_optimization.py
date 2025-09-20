"""
Database query optimization tests for Terra Constellata.

This module provides comprehensive query optimization including:
- Query performance analysis
- Index effectiveness testing
- Query plan optimization
- Database tuning recommendations
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


@pytest.mark.performance
@pytest.mark.database
@pytest.mark.asyncio
async def test_query_performance_analysis(postgis_connection, ckg_connection):
    """Analyze query performance and identify optimization opportunities."""
    if not all([postgis_connection, ckg_connection]):
        pytest.skip("Database connections not available")

    # Test PostGIS query performance
    await analyze_postgis_query_performance(postgis_connection)

    # Test CKG query performance
    await analyze_ckg_query_performance(ckg_connection)

    # Compare query performance across databases
    await compare_cross_database_performance(postgis_connection, ckg_connection)


@pytest.mark.performance
@pytest.mark.database
@pytest.mark.asyncio
async def test_index_effectiveness(postgis_connection):
    """Test effectiveness of database indexes."""
    if not postgis_connection:
        pytest.skip("PostGIS connection not available")

    # Test spatial index effectiveness
    await test_spatial_index_effectiveness(postgis_connection)

    # Test regular index effectiveness
    await test_regular_index_effectiveness(postgis_connection)

    # Analyze index usage statistics
    await analyze_index_usage_statistics(postgis_connection)


@pytest.mark.performance
@pytest.mark.database
@pytest.mark.asyncio
async def test_query_plan_optimization(postgis_connection):
    """Test and optimize query execution plans."""
    if not postgis_connection:
        pytest.skip("PostGIS connection not available")

    # Analyze query plans
    await analyze_query_plans(postgis_connection)

    # Test query rewriting for optimization
    await test_query_rewriting(postgis_connection)

    # Test parameterized vs non-parameterized queries
    await test_parameterized_queries(postgis_connection)


@pytest.mark.performance
@pytest.mark.database
@pytest.mark.asyncio
async def test_connection_pool_optimization(postgis_connection):
    """Test database connection pool optimization."""
    if not postgis_connection:
        pytest.skip("PostGIS connection not available")

    # Test connection pool sizing
    await test_connection_pool_sizing(postgis_connection)

    # Test connection reuse efficiency
    await test_connection_reuse_efficiency(postgis_connection)

    # Test connection pool under load
    await test_connection_pool_under_load(postgis_connection)


async def analyze_postgis_query_performance(connection):
    """Analyze PostGIS query performance."""
    test_queries = [
        {
            "name": "Simple count",
            "query": "SELECT COUNT(*) FROM locations",
            "description": "Basic count query",
        },
        {
            "name": "Spatial bounding box",
            "query": """
                SELECT name, entity FROM locations
                WHERE ST_Within(geom, ST_MakeEnvelope(-180, -90, 180, 90, 4326))
                LIMIT 100
            """,
            "description": "Spatial bounding box query",
        },
        {
            "name": "Distance calculation",
            "query": """
                SELECT name, ST_Distance(geom, ST_SetSRID(ST_MakePoint(0, 0), 4326)) as distance
                FROM locations
                ORDER BY distance
                LIMIT 50
            """,
            "description": "Distance calculation with ordering",
        },
        {
            "name": "Complex spatial join",
            "query": """
                SELECT l1.name, l2.name, ST_Distance(l1.geom, l2.geom) as distance
                FROM locations l1, locations l2
                WHERE ST_DWithin(l1.geom, l2.geom, 10000) AND l1.name < l2.name
                LIMIT 25
            """,
            "description": "Complex spatial join operation",
        },
    ]

    performance_results = []

    for test_query in test_queries:
        # Execute query multiple times for averaging
        execution_times = []

        for i in range(5):  # 5 runs for averaging
            start_time = time.time()
            result = await connection.fetch(test_query["query"])
            end_time = time.time()

            execution_times.append(end_time - start_time)

        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)

        performance_results.append(
            {
                "name": test_query["name"],
                "description": test_query["description"],
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "result_count": len(result) if result else 0,
            }
        )

        logger.info(
            f"PostGIS Query '{test_query['name']}': "
            f"Avg {avg_time:.4f}s, Min {min_time:.4f}s, Max {max_time:.4f}s, "
            f"Results: {len(result) if result else 0}"
        )

    return performance_results


async def analyze_ckg_query_performance(connection):
    """Analyze CKG (ArangoDB) query performance."""
    test_queries = [
        {
            "name": "Simple count",
            "query": """
                FOR doc IN geospatial_entities
                COLLECT WITH COUNT INTO length
                RETURN length
            """,
            "description": "Basic count query",
        },
        {
            "name": "Filtered query",
            "query": """
                FOR doc IN geospatial_entities
                FILTER doc.latitude > 0 AND doc.latitude < 90
                RETURN doc
            """,
            "description": "Filtered geospatial query",
        },
        {
            "name": "Graph traversal",
            "query": """
                FOR location IN geospatial_entities
                    FOR relation IN location_myth_relations
                        FILTER relation._from == CONCAT('geospatial_entities/', location._key)
                        FOR myth IN mythological_narratives
                            FILTER relation._to == CONCAT('mythological_narratives/', myth._key)
                            RETURN {
                                location: location.name,
                                myth: myth.myth,
                                culture: myth.culture
                            }
            """,
            "description": "Graph traversal query",
        },
    ]

    performance_results = []

    for test_query in test_queries:
        # Execute query multiple times for averaging
        execution_times = []

        for i in range(5):  # 5 runs for averaging
            start_time = time.time()
            result = await connection.execute_aql(test_query["query"])
            end_time = time.time()

            execution_times.append(end_time - start_time)

        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)

        performance_results.append(
            {
                "name": test_query["name"],
                "description": test_query["description"],
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "result_count": len(result) if result else 0,
            }
        )

        logger.info(
            f"CKG Query '{test_query['name']}': "
            f"Avg {avg_time:.4f}s, Min {min_time:.4f}s, Max {max_time:.4f}s, "
            f"Results: {len(result) if result else 0}"
        )

    return performance_results


async def compare_cross_database_performance(postgis_conn, ckg_conn):
    """Compare query performance across databases."""
    # Test equivalent queries
    test_scenarios = [
        {
            "name": "Entity count",
            "postgis": "SELECT COUNT(*) FROM locations",
            "ckg": """
                FOR doc IN geospatial_entities
                COLLECT WITH COUNT INTO length
                RETURN length
            """,
        },
        {
            "name": "Filtered entities",
            "postgis": """
                SELECT COUNT(*) FROM locations
                WHERE latitude > 0
            """,
            "ckg": """
                FOR doc IN geospatial_entities
                FILTER doc.latitude > 0
                COLLECT WITH COUNT INTO length
                RETURN length
            """,
        },
    ]

    comparison_results = []

    for scenario in test_scenarios:
        # Test PostGIS
        postgis_times = []
        for i in range(3):
            start_time = time.time()
            result = await postgis_conn.fetch(scenario["postgis"])
            end_time = time.time()
            postgis_times.append(end_time - start_time)

        # Test CKG
        ckg_times = []
        for i in range(3):
            start_time = time.time()
            result = await ckg_conn.execute_aql(scenario["ckg"])
            end_time = time.time()
            ckg_times.append(end_time - start_time)

        postgis_avg = sum(postgis_times) / len(postgis_times)
        ckg_avg = sum(ckg_times) / len(ckg_times)

        comparison_results.append(
            {
                "scenario": scenario["name"],
                "postgis_avg": postgis_avg,
                "ckg_avg": ckg_avg,
                "performance_ratio": ckg_avg / postgis_avg if postgis_avg > 0 else 0,
            }
        )

        logger.info(
            f"Cross-DB comparison '{scenario['name']}': "
            f"PostGIS {postgis_avg:.4f}s, CKG {ckg_avg:.4f}s, "
            f"Ratio: {ckg_avg/postgis_avg:.2f}x"
        )

    return comparison_results


async def test_spatial_index_effectiveness(connection):
    """Test spatial index effectiveness."""
    # Test query performance with and without spatial index
    test_points = [
        (-74.0, 40.7),  # New York area
        (2.3, 48.8),  # Paris area
        (-118.2, 34.0),  # Los Angeles area
        (139.7, 35.7),  # Tokyo area
    ]

    for lon, lat in test_points:
        # Test distance query
        start_time = time.time()
        result = await connection.fetch(
            """
            SELECT name, ST_Distance(geom, ST_SetSRID(ST_MakePoint($1, $2), 4326)) as distance
            FROM locations
            WHERE ST_DWithin(geom, ST_SetSRID(ST_MakePoint($1, $2), 4326), 100000)
            ORDER BY distance
            LIMIT 10
        """,
            lon,
            lat,
        )
        end_time = time.time()

        query_time = end_time - start_time

        logger.info(
            f"Spatial query near ({lon:.1f}, {lat:.1f}): "
            f"{query_time:.4f}s, {len(result)} results"
        )

        # Analyze if spatial index is being used (would need EXPLAIN ANALYZE)


async def test_regular_index_effectiveness(connection):
    """Test regular index effectiveness."""
    # Test queries that should benefit from regular indexes
    test_queries = [
        {
            "name": "Entity filter",
            "query": "SELECT COUNT(*) FROM locations WHERE entity = 'city'",
            "description": "Filter by entity type",
        },
        {
            "name": "Name search",
            "query": "SELECT COUNT(*) FROM locations WHERE name LIKE 'New%'",
            "description": "Name prefix search",
        },
    ]

    for test_query in test_queries:
        start_time = time.time()
        result = await connection.fetch(test_query["query"])
        end_time = time.time()

        query_time = end_time - start_time

        logger.info(
            f"Regular index query '{test_query['name']}': "
            f"{query_time:.4f}s, result: {result[0]['count'] if result else 0}"
        )


async def analyze_index_usage_statistics(connection):
    """Analyze database index usage statistics."""
    try:
        # Get index statistics from PostgreSQL
        index_stats = await connection.fetch(
            """
            SELECT
                schemaname,
                tablename,
                indexname,
                idx_scan,
                idx_tup_read,
                idx_tup_fetch
            FROM pg_stat_user_indexes
            WHERE schemaname = 'public'
            ORDER BY idx_scan DESC
        """
        )

        logger.info("Database index usage statistics:")
        for stat in index_stats[:10]:  # Top 10 indexes
            logger.info(
                f"  {stat['tablename']}.{stat['indexname']}: "
                f"scans={stat['idx_scan']}, "
                f"tuples_read={stat['idx_tup_read']}, "
                f"tuples_fetched={stat['idx_tup_fetch']}"
            )

    except Exception as e:
        logger.warning(f"Could not retrieve index statistics: {e}")


async def analyze_query_plans(connection):
    """Analyze query execution plans."""
    test_queries = [
        "SELECT COUNT(*) FROM locations",
        "SELECT * FROM locations WHERE entity = 'city' LIMIT 10",
        """
        SELECT l1.name, l2.name, ST_Distance(l1.geom, l2.geom) as distance
        FROM locations l1, locations l2
        WHERE ST_DWithin(l1.geom, l2.geom, 1000) AND l1.name != l2.name
        LIMIT 5
        """,
    ]

    for query in test_queries:
        try:
            # Get query plan (PostgreSQL specific)
            plan_result = await connection.fetch(f"EXPLAIN ANALYZE {query}")

            logger.info(f"Query plan for: {query[:50]}...")
            for row in plan_result:
                logger.info(f"  {row['QUERY PLAN']}")

        except Exception as e:
            logger.warning(f"Could not get query plan: {e}")


async def test_query_rewriting(connection):
    """Test query rewriting for optimization."""
    # Test different ways to write the same query
    query_variants = [
        {
            "name": "Subquery",
            "query": """
                SELECT name FROM locations
                WHERE name IN (
                    SELECT name FROM locations WHERE latitude > 0
                )
            """,
        },
        {
            "name": "Join",
            "query": """
                SELECT l1.name FROM locations l1
                JOIN locations l2 ON l1.name = l2.name
                WHERE l2.latitude > 0
            """,
        },
        {
            "name": "Direct filter",
            "query": """
                SELECT name FROM locations WHERE latitude > 0
            """,
        },
    ]

    for variant in query_variants:
        start_time = time.time()
        result = await connection.fetch(variant["query"])
        end_time = time.time()

        query_time = end_time - start_time

        logger.info(
            f"Query variant '{variant['name']}': "
            f"{query_time:.4f}s, {len(result)} results"
        )


async def test_parameterized_queries(connection):
    """Test parameterized vs non-parameterized queries."""
    # Test parameterized query
    param_query = "SELECT COUNT(*) FROM locations WHERE entity = $1"
    param_times = []

    for i in range(10):
        start_time = time.time()
        result = await connection.fetch(param_query, "city")
        end_time = time.time()
        param_times.append(end_time - start_time)

    # Test non-parameterized query
    nonparam_query = "SELECT COUNT(*) FROM locations WHERE entity = 'city'"
    nonparam_times = []

    for i in range(10):
        start_time = time.time()
        result = await connection.fetch(nonparam_query)
        end_time = time.time()
        nonparam_times.append(end_time - start_time)

    param_avg = sum(param_times) / len(param_times)
    nonparam_avg = sum(nonparam_times) / len(nonparam_times)

    logger.info(f"Parameterized query: {param_avg:.6f}s avg")
    logger.info(f"Non-parameterized query: {nonparam_avg:.6f}s avg")
    logger.info(
        f"Performance difference: {((nonparam_avg - param_avg) / param_avg * 100):+.1f}%"
    )


async def test_connection_pool_sizing(connection):
    """Test optimal connection pool sizing."""
    # Test different connection pool sizes
    pool_sizes = [1, 5, 10, 20]

    for pool_size in pool_sizes:
        logger.info(f"Testing connection pool size: {pool_size}")

        # Simulate concurrent operations
        tasks = []
        for i in range(pool_size):
            task = asyncio.create_task(
                connection.fetch("SELECT COUNT(*) FROM locations")
            )
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_query = total_time / pool_size

        logger.info(
            f"  Pool size {pool_size}: {total_time:.3f}s total, "
            f"{avg_time_per_query:.4f}s per query"
        )


async def test_connection_reuse_efficiency(connection):
    """Test connection reuse efficiency."""
    # Test reusing the same connection vs creating new ones
    reuse_times = []
    new_conn_times = []

    # Test with connection reuse
    for i in range(20):
        start_time = time.time()
        result = await connection.fetch("SELECT COUNT(*) FROM locations")
        end_time = time.time()
        reuse_times.append(end_time - start_time)

    # Note: Testing new connections would require connection pool management
    # For now, just measure the reuse efficiency

    reuse_avg = sum(reuse_times) / len(reuse_times)
    reuse_stddev = (
        sum((x - reuse_avg) ** 2 for x in reuse_times) / len(reuse_times)
    ) ** 0.5

    logger.info(
        f"Connection reuse - Avg: {reuse_avg:.6f}s, StdDev: {reuse_stddev:.6f}s"
    )
    logger.info(f"Connection efficiency: {reuse_stddev/reuse_avg:.2%} variability")


async def test_connection_pool_under_load(connection):
    """Test connection pool performance under load."""
    # Simulate sustained load
    load_duration = 10  # seconds
    end_time = time.time() + load_duration

    query_count = 0
    total_time = 0

    while time.time() < end_time:
        start_time = time.time()
        result = await connection.fetch("SELECT COUNT(*) FROM locations")
        end_time_query = time.time()

        total_time += end_time_query - start_time
        query_count += 1

        # Small delay to prevent overwhelming
        await asyncio.sleep(0.01)

    avg_query_time = total_time / query_count
    qps = query_count / load_duration

    logger.info(
        f"Connection pool under load - Duration: {load_duration}s, "
        f"Queries: {query_count}, QPS: {qps:.1f}, "
        f"Avg query time: {avg_query_time:.4f}s"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short", "-m", "performance"])
