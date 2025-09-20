"""
Database integration tests for Terra Constellata.

Tests the interaction between PostGIS and ArangoDB (CKG) databases,
including data synchronization, cross-database queries, and performance.
"""

import pytest
import asyncio
import time
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.asyncio
async def test_database_connections(postgis_connection, ckg_connection):
    """Test that both databases can be connected to simultaneously."""
    if not all([postgis_connection, ckg_connection]):
        pytest.skip("Database connections not available")

    # Test PostGIS connection
    postgis_result = await postgis_connection.fetch("SELECT version()")
    assert postgis_result
    assert "PostgreSQL" in postgis_result[0]["version"]

    # Test CKG connection
    ckg_result = await ckg_connection.execute_aql("RETURN 1")
    assert ckg_result == [1]

    logger.info("Database connections test passed")


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.asyncio
async def test_cross_database_data_flow(
    postgis_connection, ckg_connection, sample_data
):
    """Test data flow between PostGIS and CKG databases."""
    if not all([postgis_connection, ckg_connection]):
        pytest.skip("Database connections not available")

    # Insert test data into PostGIS
    geospatial_data = sample_data["geospatial"][0]
    await postgis_connection.execute(
        """
        INSERT INTO locations (name, entity, latitude, longitude, description, geom)
        VALUES ($1, $2, $3, $4, $5, ST_SetSRID(ST_MakePoint($4, $3), 4326))
    """,
        geospatial_data["name"],
        geospatial_data["entity"],
        geospatial_data["latitude"],
        geospatial_data["longitude"],
        geospatial_data["description"],
    )

    # Query the data back from PostGIS
    postgis_result = await postgis_connection.fetch(
        """
        SELECT name, entity, latitude, longitude, description
        FROM locations WHERE name = $1
    """,
        geospatial_data["name"],
    )

    assert len(postgis_result) == 1
    postgis_record = postgis_result[0]

    # Sync to CKG
    await ckg_connection.execute_aql(
        """
        INSERT {
            _key: @name,
            name: @name,
            entity: @entity,
            latitude: @latitude,
            longitude: @longitude,
            description: @description,
            source: 'postgis_sync',
            synced_at: @timestamp
        } INTO geospatial_entities
        OPTIONS { overwrite: true }
    """,
        name=postgis_record["name"],
        entity=postgis_record["entity"],
        latitude=postgis_record["latitude"],
        longitude=postgis_record["longitude"],
        description=postgis_record["description"],
        timestamp=int(time.time()),
    )

    # Verify data in CKG
    ckg_result = await ckg_connection.execute_aql(
        """
        FOR doc IN geospatial_entities
        FILTER doc.name == @name
        RETURN doc
    """,
        name=geospatial_data["name"],
    )

    assert len(ckg_result) == 1
    ckg_record = ckg_result[0]

    # Verify data consistency
    assert ckg_record["name"] == postgis_record["name"]
    assert ckg_record["entity"] == postgis_record["entity"]
    assert abs(ckg_record["latitude"] - postgis_record["latitude"]) < 0.0001
    assert abs(ckg_record["longitude"] - postgis_record["longitude"]) < 0.0001

    logger.info("Cross-database data flow test passed")


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.asyncio
async def test_geospatial_knowledge_graph_integration(
    postgis_connection, ckg_connection, sample_data
):
    """Test integration between geospatial data and knowledge graph."""
    if not all([postgis_connection, ckg_connection]):
        pytest.skip("Database connections not available")

    # Insert geospatial entities
    for item in sample_data["geospatial"]:
        await postgis_connection.execute(
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

    # Create knowledge graph relationships
    for myth in sample_data["mythological"]:
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
            culture=myth["culture"],
            myth=myth["myth"],
            narrative=myth["narrative"],
        )

    # Create relationships between locations and myths
    location_myth_relations = [
        ("Eiffel Tower", "Greek", "Creation of the World"),
        ("New York City", "Norse", "Ragnarok"),
    ]

    for location_name, culture, myth in location_myth_relations:
        # Create relationship in CKG
        await ckg_connection.execute_aql(
            """
            LET location = FIRST(
                FOR loc IN geospatial_entities
                FILTER loc.name == @location_name
                RETURN loc
            )
            LET myth_doc = FIRST(
                FOR m IN mythological_narratives
                FILTER m.culture == @culture AND m.myth == @myth
                RETURN m
            )
            INSERT {
                _from: CONCAT('geospatial_entities/', location._key),
                _to: CONCAT('mythological_narratives/', myth_doc._key),
                type: 'cultural_association',
                strength: 0.8
            } INTO location_myth_relations
            OPTIONS { ignoreErrors: true }
        """,
            location_name=location_name,
            culture=culture,
            myth=myth,
        )

    # Query integrated data
    integrated_results = await ckg_connection.execute_aql(
        """
        FOR location IN geospatial_entities
            FOR edge IN location_myth_relations
                FILTER edge._from == CONCAT('geospatial_entities/', location._key)
                FOR myth IN mythological_narratives
                    FILTER edge._to == CONCAT('mythological_narratives/', myth._key)
                    RETURN {
                        location: location.name,
                        myth: myth.myth,
                        culture: myth.culture,
                        relationship_strength: edge.strength
                    }
    """
    )

    assert len(integrated_results) > 0

    # Verify PostGIS spatial query integration
    for result in integrated_results:
        # Get location coordinates from PostGIS
        coords = await postgis_connection.fetch(
            """
            SELECT ST_X(geom) as longitude, ST_Y(geom) as latitude
            FROM locations WHERE name = $1
        """,
            result["location"],
        )

        if coords:
            assert "latitude" in coords[0]
            assert "longitude" in coords[0]

    logger.info("Geospatial-knowledge graph integration test passed")


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.performance
async def test_database_query_performance(
    postgis_connection, ckg_connection, benchmark
):
    """Test database query performance for common operations."""
    if not all([postgis_connection, ckg_connection]):
        pytest.skip("Database connections not available")

    # Setup test data
    test_locations = []
    for i in range(100):
        test_locations.append(
            {
                "name": f"TestLocation{i}",
                "entity": "test_entity",
                "latitude": 40.0 + i * 0.01,
                "longitude": -74.0 + i * 0.01,
                "description": f"Test location {i}",
            }
        )

    # Insert test data into PostGIS
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

    # Benchmark PostGIS spatial query
    async def benchmark_postgis_spatial_query():
        return await postgis_connection.fetch(
            """
            SELECT name, entity,
                   ST_Distance(geom, ST_SetSRID(ST_MakePoint(-74.0, 40.0), 4326)) as distance
            FROM locations
            WHERE ST_DWithin(geom, ST_SetSRID(ST_MakePoint(-74.0, 40.0), 4326), 1000)
            ORDER BY distance
            LIMIT 10
        """
        )

    postgis_result = benchmark(benchmark_postgis_spatial_query)
    assert len(postgis_result) > 0

    # Insert corresponding data into CKG
    for loc in test_locations[:50]:  # Insert subset for CKG
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

    # Benchmark CKG graph query
    async def benchmark_ckg_graph_query():
        return await ckg_connection.execute_aql(
            """
            FOR entity IN geospatial_entities
            FILTER entity.latitude > 40.0 AND entity.latitude < 41.0
            SORT entity.latitude
            LIMIT 10
            RETURN entity
        """
        )

    ckg_result = benchmark(benchmark_ckg_graph_query)
    assert len(ckg_result) > 0

    logger.info("Database query performance test completed")


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.asyncio
async def test_database_transaction_integrity(postgis_connection, ckg_connection):
    """Test transaction integrity across database operations."""
    if not all([postgis_connection, ckg_connection]):
        pytest.skip("Database connections not available")

    # Test PostGIS transaction
    async with postgis_connection.transaction():
        # Insert test data
        await postgis_connection.execute(
            """
            INSERT INTO locations (name, entity, latitude, longitude, description, geom)
            VALUES ($1, $2, $3, $4, $5, ST_SetSRID(ST_MakePoint($4, $3), 4326))
        """,
            "TransactionTest",
            "test",
            40.0,
            -74.0,
            "Transaction test location",
        )

        # Verify insertion within transaction
        result = await postgis_connection.fetch(
            """
            SELECT COUNT(*) FROM locations WHERE name = $1
        """,
            "TransactionTest",
        )
        assert result[0]["count"] == 1

    # Verify data persists after transaction
    result = await postgis_connection.fetch(
        """
        SELECT COUNT(*) FROM locations WHERE name = $1
    """,
        "TransactionTest",
    )
    assert result[0]["count"] == 1

    # Test CKG transaction-like behavior (ArangoDB doesn't have traditional transactions)
    await ckg_connection.execute_aql(
        """
        INSERT {
            name: @name,
            type: 'transaction_test'
        } INTO test_entities
        OPTIONS { overwrite: true }
    """,
        name="TransactionTest",
    )

    # Verify insertion
    result = await ckg_connection.execute_aql(
        """
        FOR doc IN test_entities
        FILTER doc.name == @name
        RETURN doc
    """,
        name="TransactionTest",
    )
    assert len(result) == 1

    logger.info("Database transaction integrity test passed")


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.asyncio
async def test_database_connection_pooling(postgis_connection, ckg_connection):
    """Test database connection pooling and concurrent operations."""
    if not all([postgis_connection, ckg_connection]):
        pytest.skip("Database connections not available")

    async def concurrent_postgis_queries(query_id: int):
        """Execute concurrent PostGIS queries."""
        results = []
        for i in range(5):
            result = await postgis_connection.fetch(
                """
                SELECT COUNT(*) as count FROM locations
            """
            )
            results.append(result[0]["count"])
        return results

    async def concurrent_ckg_queries(query_id: int):
        """Execute concurrent CKG queries."""
        results = []
        for i in range(5):
            result = await ckg_connection.execute_aql(
                """
                FOR doc IN geospatial_entities
                COLLECT WITH COUNT INTO length
                RETURN length
            """
            )
            results.append(result[0] if result else 0)
        return results

    # Execute concurrent operations
    tasks = []
    for i in range(10):
        tasks.append(asyncio.create_task(concurrent_postgis_queries(i)))
        tasks.append(asyncio.create_task(concurrent_ckg_queries(i)))

    results = await asyncio.gather(*tasks)

    # Verify all operations completed successfully
    assert len(results) == 20  # 10 PostGIS + 10 CKG tasks
    for result in results:
        assert isinstance(result, list)
        assert len(result) == 5  # Each task should return 5 results

    logger.info("Database connection pooling test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
