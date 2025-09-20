#!/usr/bin/env python3
"""
Schema definitions for the AI Puzzle Pieces Data Pipeline

This module contains SQL schema definitions and table creation functions
for the PostgreSQL/PostGIS database used in the Terra Constellata project.
"""

import logging
from .connection import PostGISConnection

logger = logging.getLogger(__name__)

# Table schema for puzzle_pieces
PUZZLE_PIECES_SCHEMA = """
CREATE TABLE IF NOT EXISTS puzzle_pieces (
    id SERIAL PRIMARY KEY,
    row_number INTEGER NOT NULL,
    name VARCHAR(255) NOT NULL,
    entity VARCHAR(255),
    sub_entity VARCHAR(255),
    description TEXT,
    source_url TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    geom GEOMETRY(POINT, 4326),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT valid_latitude CHECK (latitude >= -90 AND latitude <= 90),
    CONSTRAINT valid_longitude CHECK (longitude >= -180 AND longitude <= 180),
    CONSTRAINT unique_row_number UNIQUE (row_number),

    -- Indexes for performance
    INDEX idx_puzzle_pieces_geom ON puzzle_pieces USING GIST (geom),
    INDEX idx_puzzle_pieces_entity ON puzzle_pieces (entity),
    INDEX idx_puzzle_pieces_sub_entity ON puzzle_pieces (sub_entity),
    INDEX idx_puzzle_pieces_name ON puzzle_pieces (name)
);

-- Create spatial index for geospatial queries
CREATE INDEX IF NOT EXISTS idx_puzzle_pieces_geom_gist
ON puzzle_pieces USING GIST (geom);

-- Create indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_puzzle_pieces_entity_sub
ON puzzle_pieces (entity, sub_entity);

CREATE INDEX IF NOT EXISTS idx_puzzle_pieces_created_at
ON puzzle_pieces (created_at);

-- Trigger to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_puzzle_pieces_updated_at
    BEFORE UPDATE ON puzzle_pieces
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
"""

# Additional tables for data quality and processing metadata
DATA_QUALITY_SCHEMA = """
CREATE TABLE IF NOT EXISTS data_quality_log (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(255) NOT NULL,
    record_id INTEGER,
    issue_type VARCHAR(100) NOT NULL,
    issue_description TEXT,
    severity VARCHAR(20) DEFAULT 'WARNING',
    resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_data_quality_table ON data_quality_log (table_name),
    INDEX idx_data_quality_severity ON data_quality_log (severity),
    INDEX idx_data_quality_resolved ON data_quality_log (resolved)
);

CREATE TABLE IF NOT EXISTS processing_log (
    id SERIAL PRIMARY KEY,
    operation VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    records_processed INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,

    INDEX idx_processing_operation ON processing_log (operation),
    INDEX idx_processing_status ON processing_log (status)
);
"""


def create_puzzle_pieces_table(db: PostGISConnection) -> bool:
    """
    Create the puzzle_pieces table with PostGIS geometry column.

    Args:
        db: PostGISConnection instance

    Returns:
        bool: True if successful
    """
    try:
        logger.info("Creating puzzle_pieces table...")

        # Execute the schema creation
        db.execute_command(PUZZLE_PIECES_SCHEMA)

        logger.info("puzzle_pieces table created successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to create puzzle_pieces table: {e}")
        return False


def create_data_quality_tables(db: PostGISConnection) -> bool:
    """
    Create data quality and processing log tables.

    Args:
        db: PostGISConnection instance

    Returns:
        bool: True if successful
    """
    try:
        logger.info("Creating data quality tables...")

        # Execute the data quality schema
        db.execute_command(DATA_QUALITY_SCHEMA)

        logger.info("Data quality tables created successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to create data quality tables: {e}")
        return False


def initialize_database(db: PostGISConnection) -> bool:
    """
    Initialize the database with all required tables and extensions.

    Args:
        db: PostGISConnection instance

    Returns:
        bool: True if successful
    """
    try:
        logger.info("Initializing database...")

        # Enable PostGIS extension
        if not db.enable_postgis():
            logger.error("Failed to enable PostGIS extension")
            return False

        # Create main table
        if not create_puzzle_pieces_table(db):
            logger.error("Failed to create puzzle_pieces table")
            return False

        # Create auxiliary tables
        if not create_data_quality_tables(db):
            logger.error("Failed to create data quality tables")
            return False

        logger.info("Database initialization completed successfully")
        return True

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


def drop_tables(db: PostGISConnection) -> bool:
    """
    Drop all tables (useful for testing or resetting).

    Args:
        db: PostGISConnection instance

    Returns:
        bool: True if successful
    """
    try:
        logger.info("Dropping tables...")

        drop_queries = [
            "DROP TABLE IF EXISTS data_quality_log CASCADE;",
            "DROP TABLE IF EXISTS processing_log CASCADE;",
            "DROP TABLE IF EXISTS puzzle_pieces CASCADE;",
            "DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;",
        ]

        for query in drop_queries:
            db.execute_command(query)

        logger.info("Tables dropped successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to drop tables: {e}")
        return False


if __name__ == "__main__":
    # Test schema creation
    logging.basicConfig(level=logging.INFO)

    db = PostGISConnection()
    if db.connect():
        initialize_database(db)
        db.disconnect()
    else:
        print("Failed to connect to database")
