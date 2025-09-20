"""
Data Ingestion Module for Inspiration Engine

This module handles data ingestion from multiple sources:
- Cultural Knowledge Graph (CKG) - ArangoDB
- PostGIS geospatial database
- Integration with existing data pipelines
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime, timedelta
import json

# Import existing connection modules
from ..data.ckg.connection import get_db_connection as get_ckg_connection
from ..data.postgis.connection import PostGISConnection

logger = logging.getLogger(__name__)


class DataIngestor:
    """
    Handles data ingestion from CKG and PostGIS databases for novelty detection
    """

    def __init__(
        self,
        ckg_host: str = "http://localhost:8529",
        ckg_db: str = "ckg_db",
        ckg_user: str = "root",
        ckg_password: str = "",
        postgis_host: str = "localhost",
        postgis_port: int = 5432,
        postgis_db: str = "terra_constellata",
        postgis_user: str = "postgres",
        postgis_password: str = "",
    ):
        """
        Initialize data ingestor with database connection parameters

        Args:
            ckg_host: ArangoDB host URL
            ckg_db: CKG database name
            ckg_user: CKG username
            ckg_password: CKG password
            postgis_host: PostGIS host
            postgis_port: PostGIS port
            postgis_db: PostGIS database name
            postgis_user: PostGIS username
            postgis_password: PostGIS password
        """
        self.ckg_config = {
            "host": ckg_host,
            "database": ckg_db,
            "username": ckg_user,
            "password": ckg_password,
        }

        self.postgis_config = {
            "host": postgis_host,
            "port": postgis_port,
            "database": postgis_db,
            "user": postgis_user,
            "password": postgis_password,
        }

        self.ckg_db = None
        self.postgis_db = None

    def connect_databases(self) -> bool:
        """
        Establish connections to both databases

        Returns:
            bool: True if both connections successful
        """
        try:
            # Connect to CKG
            self.ckg_db = get_ckg_connection(
                host=self.ckg_config["host"],
                username=self.ckg_config["username"],
                password=self.ckg_config["password"],
                database=self.ckg_config["database"],
            )
            logger.info("Connected to CKG database")

            # Connect to PostGIS
            self.postgis_db = PostGISConnection(
                host=self.postgis_config["host"],
                port=self.postgis_config["port"],
                database=self.postgis_config["database"],
                user=self.postgis_config["user"],
                password=self.postgis_config["password"],
            )

            if self.postgis_db.connect():
                logger.info("Connected to PostGIS database")
                return True
            else:
                logger.error("Failed to connect to PostGIS database")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to databases: {e}")
            return False

    def disconnect_databases(self):
        """Close database connections"""
        if self.postgis_db:
            self.postgis_db.disconnect()
        # CKG connection doesn't need explicit disconnect
        logger.info("Database connections closed")

    def get_ckg_data(
        self,
        collections: List[str] = None,
        limit: int = 1000,
        filters: Dict[str, Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Retrieve data from CKG collections

        Args:
            collections: List of collection names to query (default: all vertex collections)
            limit: Maximum number of records per collection
            filters: Optional filters to apply

        Returns:
            Dictionary mapping collection names to DataFrames
        """
        if not self.ckg_db:
            logger.error("CKG database not connected")
            return {}

        if collections is None:
            # Get all vertex collections
            collections = [
                "MythologicalEntity",
                "GeographicFeature",
                "CulturalConcept",
                "TextSource",
                "GeospatialPoint",
            ]

        data_frames = {}

        for collection_name in collections:
            try:
                if not self.ckg_db.has_collection(collection_name):
                    logger.warning(f"Collection {collection_name} does not exist")
                    continue

                collection = self.ckg_db.collection(collection_name)

                # Build AQL query
                query = f"""
                FOR doc IN {collection_name}
                LIMIT {limit}
                RETURN doc
                """

                cursor = self.ckg_db.aql.execute(query)
                documents = list(cursor)

                if documents:
                    df = pd.DataFrame(documents)
                    data_frames[collection_name] = df
                    logger.info(f"Retrieved {len(df)} documents from {collection_name}")
                else:
                    logger.info(f"No data found in {collection_name}")

            except Exception as e:
                logger.error(f"Failed to retrieve data from {collection_name}: {e}")

        return data_frames

    def get_postgis_data(
        self,
        table_name: str = "puzzle_pieces",
        columns: List[str] = None,
        limit: int = 1000,
        filters: Dict[str, Any] = None,
        spatial_filters: Dict[str, Any] = None,
    ) -> pd.DataFrame:
        """
        Retrieve geospatial data from PostGIS

        Args:
            table_name: Name of the table to query
            columns: List of columns to retrieve (default: all)
            limit: Maximum number of records
            filters: SQL WHERE filters
            spatial_filters: Spatial filters (e.g., bounding box)

        Returns:
            DataFrame with geospatial data
        """
        if not self.postgis_db:
            logger.error("PostGIS database not connected")
            return pd.DataFrame()

        try:
            # Build SELECT query
            if columns:
                column_str = ", ".join(columns)
            else:
                column_str = "*"

            query = f"SELECT {column_str} FROM {table_name}"

            # Add spatial filters
            where_conditions = []
            params = []

            if spatial_filters:
                if "bbox" in spatial_filters:
                    # Bounding box filter
                    bbox = spatial_filters[
                        "bbox"
                    ]  # [min_lon, min_lat, max_lon, max_lat]
                    where_conditions.append(
                        "ST_Within(geom, ST_MakeEnvelope(%s, %s, %s, %s, 4326))"
                    )
                    params.extend(bbox)

                if "radius" in spatial_filters:
                    # Radius filter around a point
                    center = spatial_filters["radius"]["center"]  # [lon, lat]
                    radius_km = spatial_filters["radius"]["radius_km"]
                    where_conditions.append(
                        "ST_DWithin(geom::geography, ST_Point(%s, %s)::geography, %s)"
                    )
                    params.extend(
                        [center[0], center[1], radius_km * 1000]
                    )  # Convert to meters

            # Add regular filters
            if filters:
                for key, value in filters.items():
                    where_conditions.append(f"{key} = %s")
                    params.append(value)

            if where_conditions:
                query += " WHERE " + " AND ".join(where_conditions)

            query += f" LIMIT {limit}"

            # Execute query
            results = self.postgis_db.execute_query(
                query, tuple(params) if params else None
            )

            if results:
                df = pd.DataFrame(results)
                logger.info(f"Retrieved {len(df)} records from {table_name}")
                return df
            else:
                logger.info(f"No data found in {table_name}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to retrieve data from PostGIS: {e}")
            return pd.DataFrame()

    def get_recent_data(self, hours: int = 24) -> Dict[str, pd.DataFrame]:
        """
        Retrieve recently added/modified data from both databases

        Args:
            hours: Number of hours to look back

        Returns:
            Dictionary with recent data from both sources
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        # Get recent PostGIS data
        postgis_filters = {"updated_at": f"> '{cutoff_time.isoformat()}'"}
        postgis_data = self.get_postgis_data(filters=postgis_filters)

        # Get recent CKG data (assuming documents have timestamp fields)
        ckg_data = {}
        collections = ["MythologicalEntity", "GeographicFeature", "CulturalConcept"]

        for collection in collections:
            try:
                query = f"""
                FOR doc IN {collection}
                FILTER doc.created_at >= '{cutoff_time.isoformat()}'
                OR doc.updated_at >= '{cutoff_time.isoformat()}'
                LIMIT 500
                RETURN doc
                """

                cursor = self.ckg_db.aql.execute(query)
                documents = list(cursor)

                if documents:
                    df = pd.DataFrame(documents)
                    ckg_data[collection] = df

            except Exception as e:
                logger.error(f"Failed to get recent CKG data from {collection}: {e}")

        return {"postgis_recent": postgis_data, "ckg_recent": ckg_data}

    def get_spatial_clusters(
        self, table_name: str = "puzzle_pieces", cluster_distance: float = 0.1
    ) -> pd.DataFrame:
        """
        Identify spatial clusters in geospatial data

        Args:
            table_name: PostGIS table name
            cluster_distance: Clustering distance in degrees

        Returns:
            DataFrame with cluster information
        """
        if not self.postgis_db:
            logger.error("PostGIS database not connected")
            return pd.DataFrame()

        try:
            query = f"""
            SELECT
                ST_ClusterDBSCAN(geom, eps := %s, minpoints := 2) OVER () AS cluster_id,
                COUNT(*) OVER (PARTITION BY ST_ClusterDBSCAN(geom, eps := %s, minpoints := 2) OVER ()) AS cluster_size,
                *
            FROM {table_name}
            WHERE geom IS NOT NULL
            ORDER BY cluster_id, cluster_size DESC
            """

            results = self.postgis_db.execute_query(
                query, (cluster_distance, cluster_distance)
            )

            if results:
                df = pd.DataFrame(results)
                logger.info(f"Identified {df['cluster_id'].nunique()} spatial clusters")
                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to identify spatial clusters: {e}")
            return pd.DataFrame()

    def get_semantic_relationships(
        self, entity_name: str, relationship_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve semantic relationships for a given entity from CKG

        Args:
            entity_name: Name of the entity to analyze
            relationship_types: Types of relationships to include

        Returns:
            Dictionary with entity relationships and metadata
        """
        if not self.ckg_db:
            logger.error("CKG database not connected")
            return {}

        if relationship_types is None:
            relationship_types = ["DEPICTS", "LOCATED_AT", "MENTIONED_IN", "RELATED_TO"]

        relationships = {}

        try:
            # Find the entity
            entity_query = """
            FOR entity IN MythologicalEntity
            FILTER entity.name == @name
            RETURN entity
            """

            cursor = self.ckg_db.aql.execute(
                entity_query, bind_vars={"name": entity_name}
            )
            entities = list(cursor)

            if not entities:
                logger.warning(f"Entity '{entity_name}' not found in CKG")
                return {}

            entity = entities[0]

            # Get relationships
            for rel_type in relationship_types:
                if not self.ckg_db.has_collection(rel_type):
                    continue

                rel_query = f"""
                FOR edge IN {rel_type}
                FILTER edge._from == @entity_id OR edge._to == @entity_id
                FOR vertex IN MythologicalEntity
                FILTER vertex._id == (edge._from == @entity_id ? edge._to : edge._from)
                RETURN {{
                    relationship: edge,
                    connected_entity: vertex,
                    direction: edge._from == @entity_id ? 'outgoing' : 'incoming'
                }}
                """

                cursor = self.ckg_db.aql.execute(
                    rel_query, bind_vars={"entity_id": entity["_id"]}
                )
                rel_data = list(cursor)

                if rel_data:
                    relationships[rel_type] = rel_data

            return {
                "entity": entity,
                "relationships": relationships,
                "relationship_count": sum(len(rels) for rels in relationships.values()),
            }

        except Exception as e:
            logger.error(f"Failed to get semantic relationships: {e}")
            return {}

    def prepare_novelty_data(
        self,
        include_spatial: bool = True,
        include_semantic: bool = True,
        time_window_hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Prepare comprehensive data package for novelty detection

        Args:
            include_spatial: Whether to include geospatial data
            include_semantic: Whether to include semantic/graph data
            time_window_hours: Time window for recent data

        Returns:
            Dictionary with prepared data for novelty analysis
        """
        data_package = {
            "timestamp": datetime.utcnow(),
            "spatial_data": {},
            "semantic_data": {},
            "recent_changes": {},
            "metadata": {},
        }

        try:
            # Get spatial data
            if include_spatial:
                data_package["spatial_data"] = {
                    "puzzle_pieces": self.get_postgis_data(),
                    "spatial_clusters": self.get_spatial_clusters(),
                }

            # Get semantic data
            if include_semantic:
                data_package["semantic_data"] = self.get_ckg_data()

            # Get recent changes
            data_package["recent_changes"] = self.get_recent_data(time_window_hours)

            # Add metadata
            data_package["metadata"] = {
                "spatial_records": sum(
                    len(df)
                    for df in data_package["spatial_data"].values()
                    if isinstance(df, pd.DataFrame)
                ),
                "semantic_records": sum(
                    len(df)
                    for df in data_package["semantic_data"].values()
                    if isinstance(df, pd.DataFrame)
                ),
                "recent_changes_count": sum(
                    len(df)
                    for df in data_package["recent_changes"]["ckg_recent"].values()
                    if isinstance(df, pd.DataFrame)
                )
                + (
                    len(data_package["recent_changes"]["postgis_recent"])
                    if isinstance(
                        data_package["recent_changes"]["postgis_recent"], pd.DataFrame
                    )
                    else 0
                ),
                "data_sources": ["CKG", "PostGIS"],
                "time_window_hours": time_window_hours,
            }

            logger.info(
                f"Prepared data package with {data_package['metadata']['spatial_records']} spatial and "
                f"{data_package['metadata']['semantic_records']} semantic records"
            )

        except Exception as e:
            logger.error(f"Failed to prepare novelty data: {e}")

        return data_package
