#!/usr/bin/env python3
"""
Geospatial Query Module for AI Puzzle Pieces Data Pipeline

This module contains functions for performing geospatial queries and operations
on the puzzle pieces data stored in PostgreSQL/PostGIS.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from .connection import PostGISConnection

logger = logging.getLogger(__name__)


class GeospatialQueries:
    """
    Handles geospatial queries for puzzle pieces data.
    """

    def __init__(self, db: PostGISConnection):
        """
        Initialize the query handler.

        Args:
            db: PostGISConnection instance
        """
        self.db = db

    def get_puzzle_pieces_in_bbox(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Get puzzle pieces within a bounding box.

        Args:
            min_lon: Minimum longitude
            min_lat: Minimum latitude
            max_lon: Maximum longitude
            max_lat: Maximum latitude
            limit: Maximum number of results

        Returns:
            List of puzzle piece records
        """
        query = """
        SELECT id, row_number, name, entity, sub_entity, description,
               source_url, latitude, longitude,
               ST_AsGeoJSON(geom) as geometry
        FROM puzzle_pieces
        WHERE geom && ST_MakeEnvelope(%s, %s, %s, %s, 4326)
        ORDER BY row_number
        LIMIT %s
        """

        try:
            results = self.db.execute_query(
                query, (min_lon, min_lat, max_lon, max_lat, limit)
            )
            return [dict(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to query bounding box: {e}")
            return []

    def get_puzzle_pieces_near_point(
        self, lon: float, lat: float, distance_meters: float = 1000, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get puzzle pieces within a certain distance from a point.

        Args:
            lon: Longitude of center point
            lat: Latitude of center point
            distance_meters: Search radius in meters
            limit: Maximum number of results

        Returns:
            List of puzzle piece records with distance
        """
        query = """
        SELECT id, row_number, name, entity, sub_entity, description,
               source_url, latitude, longitude,
               ST_AsGeoJSON(geom) as geometry,
               ST_Distance(geom::geography, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography) as distance_meters
        FROM puzzle_pieces
        WHERE ST_DWithin(geom::geography, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s)
        ORDER BY distance_meters
        LIMIT %s
        """

        try:
            results = self.db.execute_query(
                query, (lon, lat, lon, lat, distance_meters, limit)
            )
            return [dict(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to query nearby points: {e}")
            return []

    def get_puzzle_pieces_by_entity(
        self, entity: str, limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get puzzle pieces by entity type.

        Args:
            entity: Entity type to filter by
            limit: Maximum number of results

        Returns:
            List of puzzle piece records
        """
        query = """
        SELECT id, row_number, name, entity, sub_entity, description,
               source_url, latitude, longitude,
               ST_AsGeoJSON(geom) as geometry
        FROM puzzle_pieces
        WHERE LOWER(entity) = LOWER(%s)
        ORDER BY row_number
        LIMIT %s
        """

        try:
            results = self.db.execute_query(query, (entity, limit))
            return [dict(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to query by entity: {e}")
            return []

    def get_puzzle_pieces_by_sub_entity(
        self, sub_entity: str, limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get puzzle pieces by sub-entity type.

        Args:
            sub_entity: Sub-entity type to filter by
            limit: Maximum number of results

        Returns:
            List of puzzle piece records
        """
        query = """
        SELECT id, row_number, name, entity, sub_entity, description,
               source_url, latitude, longitude,
               ST_AsGeoJSON(geom) as geometry
        FROM puzzle_pieces
        WHERE LOWER(sub_entity) = LOWER(%s)
        ORDER BY row_number
        LIMIT %s
        """

        try:
            results = self.db.execute_query(query, (sub_entity, limit))
            return [dict(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to query by sub-entity: {e}")
            return []

    def search_puzzle_pieces_by_name(
        self, search_term: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search puzzle pieces by name (case-insensitive partial match).

        Args:
            search_term: Term to search for
            limit: Maximum number of results

        Returns:
            List of puzzle piece records
        """
        query = """
        SELECT id, row_number, name, entity, sub_entity, description,
               source_url, latitude, longitude,
               ST_AsGeoJSON(geom) as geometry
        FROM puzzle_pieces
        WHERE LOWER(name) LIKE LOWER(%s)
        ORDER BY row_number
        LIMIT %s
        """

        try:
            search_pattern = f"%{search_term}%"
            results = self.db.execute_query(query, (search_pattern, limit))
            return [dict(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to search by name: {e}")
            return []

    def get_puzzle_pieces_within_polygon(
        self, polygon_wkt: str, limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get puzzle pieces within a polygon defined by WKT.

        Args:
            polygon_wkt: Well-Known Text representation of polygon
            limit: Maximum number of results

        Returns:
            List of puzzle piece records
        """
        query = """
        SELECT id, row_number, name, entity, sub_entity, description,
               source_url, latitude, longitude,
               ST_AsGeoJSON(geom) as geometry
        FROM puzzle_pieces
        WHERE ST_Within(geom, ST_GeomFromText(%s, 4326))
        ORDER BY row_number
        LIMIT %s
        """

        try:
            results = self.db.execute_query(query, (polygon_wkt, limit))
            return [dict(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to query within polygon: {e}")
            return []

    def get_cluster_analysis(
        self, cluster_distance: float = 1000
    ) -> List[Dict[str, Any]]:
        """
        Perform cluster analysis on puzzle pieces.

        Args:
            cluster_distance: Distance in meters to define clusters

        Returns:
            List of cluster information
        """
        query = """
        WITH clusters AS (
            SELECT
                ST_ClusterDBSCAN(geom, eps := %s, minpoints := 2) OVER () as cluster_id,
                id, row_number, name, entity, latitude, longitude, geom
            FROM puzzle_pieces
            WHERE geom IS NOT NULL
        ),
        cluster_stats AS (
            SELECT
                cluster_id,
                COUNT(*) as point_count,
                ST_Centroid(ST_Collect(geom)) as centroid,
                ST_AsGeoJSON(ST_Centroid(ST_Collect(geom))) as centroid_geojson,
                array_agg(name) as names,
                array_agg(entity) as entities
            FROM clusters
            WHERE cluster_id IS NOT NULL
            GROUP BY cluster_id
        )
        SELECT
            cluster_id,
            point_count,
            ST_X(centroid) as centroid_lon,
            ST_Y(centroid) as centroid_lat,
            centroid_geojson,
            names,
            entities
        FROM cluster_stats
        ORDER BY point_count DESC
        """

        try:
            results = self.db.execute_query(query, (cluster_distance,))
            return [dict(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to perform cluster analysis: {e}")
            return []

    def get_spatial_statistics(self) -> Dict[str, Any]:
        """
        Get spatial statistics for the puzzle pieces dataset.

        Returns:
            Dictionary with spatial statistics
        """
        stats = {}

        # Total count
        count_query = "SELECT COUNT(*) as total_count FROM puzzle_pieces"
        try:
            result = self.db.execute_query(count_query)
            stats["total_count"] = result[0]["total_count"] if result else 0
        except Exception as e:
            logger.error(f"Failed to get total count: {e}")
            stats["total_count"] = 0

        # Bounding box
        bbox_query = """
        SELECT
            ST_XMin(ST_Extent(geom)) as min_lon,
            ST_YMin(ST_Extent(geom)) as min_lat,
            ST_XMax(ST_Extent(geom)) as max_lon,
            ST_YMax(ST_Extent(geom)) as max_lat
        FROM puzzle_pieces
        WHERE geom IS NOT NULL
        """
        try:
            result = self.db.execute_query(bbox_query)
            if result and result[0]["min_lon"] is not None:
                stats["bounding_box"] = {
                    "min_lon": result[0]["min_lon"],
                    "min_lat": result[0]["min_lat"],
                    "max_lon": result[0]["max_lon"],
                    "max_lat": result[0]["max_lat"],
                }
            else:
                stats["bounding_box"] = None
        except Exception as e:
            logger.error(f"Failed to get bounding box: {e}")
            stats["bounding_box"] = None

        # Entity distribution
        entity_query = """
        SELECT entity, COUNT(*) as count
        FROM puzzle_pieces
        WHERE entity IS NOT NULL
        GROUP BY entity
        ORDER BY count DESC
        """
        try:
            results = self.db.execute_query(entity_query)
            stats["entity_distribution"] = (
                [dict(row) for row in results] if results else []
            )
        except Exception as e:
            logger.error(f"Failed to get entity distribution: {e}")
            stats["entity_distribution"] = []

        # Coordinate validity
        coord_query = """
        SELECT
            COUNT(*) as total_with_coords,
            COUNT(CASE WHEN latitude IS NULL OR longitude IS NULL THEN 1 END) as missing_coords,
            COUNT(CASE WHEN latitude = 0 AND longitude = 0 THEN 1 END) as zero_coords
        FROM puzzle_pieces
        """
        try:
            result = self.db.execute_query(coord_query)
            if result:
                stats["coordinate_stats"] = dict(result[0])
        except Exception as e:
            logger.error(f"Failed to get coordinate stats: {e}")
            stats["coordinate_stats"] = {}

        return stats

    def find_nearest_neighbors(
        self, lon: float, lat: float, k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find k nearest neighbors to a point.

        Args:
            lon: Longitude of query point
            lat: Latitude of query point
            k: Number of nearest neighbors to find

        Returns:
            List of nearest puzzle pieces with distances
        """
        query = """
        SELECT id, row_number, name, entity, sub_entity, description,
               source_url, latitude, longitude,
               ST_AsGeoJSON(geom) as geometry,
               ST_Distance(geom::geography, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography) as distance_meters
        FROM puzzle_pieces
        WHERE geom IS NOT NULL
        ORDER BY geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 4326)
        LIMIT %s
        """

        try:
            results = self.db.execute_query(query, (lon, lat, lon, lat, k))
            return [dict(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to find nearest neighbors: {e}")
            return []

    def get_puzzle_pieces_in_buffer(
        self, lon: float, lat: float, buffer_distance: float = 5000
    ) -> List[Dict[str, Any]]:
        """
        Get puzzle pieces within a buffer around a point.

        Args:
            lon: Longitude of center point
            lat: Latitude of center point
            buffer_distance: Buffer distance in meters

        Returns:
            List of puzzle pieces within buffer
        """
        query = """
        SELECT id, row_number, name, entity, sub_entity, description,
               source_url, latitude, longitude,
               ST_AsGeoJSON(geom) as geometry,
               ST_AsGeoJSON(ST_Buffer(ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s)::geometry) as buffer_geometry
        FROM puzzle_pieces
        WHERE ST_DWithin(geom::geography, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s)
        ORDER BY row_number
        """

        try:
            results = self.db.execute_query(
                query, (lon, lat, buffer_distance, lon, lat, buffer_distance)
            )
            return [dict(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to query buffer: {e}")
            return []


def perform_geospatial_query(query_type: str, db: PostGISConnection, **kwargs) -> Any:
    """
    Convenience function to perform geospatial queries.

    Args:
        query_type: Type of query to perform
        db: Database connection
        **kwargs: Query parameters

    Returns:
        Query results
    """
    queries = GeospatialQueries(db)

    if query_type == "bbox":
        return queries.get_puzzle_pieces_in_bbox(**kwargs)
    elif query_type == "nearby":
        return queries.get_puzzle_pieces_near_point(**kwargs)
    elif query_type == "by_entity":
        return queries.get_puzzle_pieces_by_entity(**kwargs)
    elif query_type == "by_sub_entity":
        return queries.get_puzzle_pieces_by_sub_entity(**kwargs)
    elif query_type == "search_name":
        return queries.search_puzzle_pieces_by_name(**kwargs)
    elif query_type == "within_polygon":
        return queries.get_puzzle_pieces_within_polygon(**kwargs)
    elif query_type == "clusters":
        return queries.get_cluster_analysis(**kwargs)
    elif query_type == "statistics":
        return queries.get_spatial_statistics()
    elif query_type == "nearest":
        return queries.find_nearest_neighbors(**kwargs)
    elif query_type == "buffer":
        return queries.get_puzzle_pieces_in_buffer(**kwargs)
    else:
        raise ValueError(f"Unknown query type: {query_type}")


if __name__ == "__main__":
    # Test geospatial queries
    logging.basicConfig(level=logging.INFO)

    db = PostGISConnection()
    if db.connect():
        queries = GeospatialQueries(db)

        # Get spatial statistics
        stats = queries.get_spatial_statistics()
        print(f"Spatial Statistics: {stats}")

        # Example bbox query
        results = queries.get_puzzle_pieces_in_bbox(-180, -90, 180, 90, limit=5)
        print(f"Bbox query results: {len(results)} records")

        db.disconnect()
    else:
        print("Failed to connect to database")
