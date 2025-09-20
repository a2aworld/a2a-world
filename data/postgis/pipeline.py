#!/usr/bin/env python3
"""
AI Puzzle Pieces Data Pipeline Main Script

This script orchestrates the complete data pipeline for ingesting,
processing, and querying puzzle pieces data in PostgreSQL/PostGIS.
"""

import logging
import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from .connection import PostGISConnection, get_db_connection
from .schema import initialize_database
from .ingestion import ingest_puzzle_pieces_csv
from .queries import perform_geospatial_query
from .data_processing import clean_and_validate_data

logger = logging.getLogger(__name__)


class PuzzlePiecesPipeline:
    """
    Main pipeline class for AI Puzzle Pieces data processing.
    """

    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline.

        Args:
            db_config: Database configuration
        """
        self.db_config = db_config or {
            "host": "localhost",
            "port": 5432,
            "database": "terra_constellata",
            "user": "postgres",
            "password": "",
        }
        self.db = None

    def initialize(self) -> bool:
        """
        Initialize the database connection and schema.

        Returns:
            bool: True if successful
        """
        try:
            logger.info("Initializing pipeline...")

            # Create database connection
            self.db = PostGISConnection(**self.db_config)

            # Connect and initialize database
            if not self.db.connect():
                logger.error("Failed to connect to database")
                return False

            if not initialize_database(self.db):
                logger.error("Failed to initialize database schema")
                return False

            logger.info("Pipeline initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            return False

    def ingest_csv(self, csv_file: str, batch_size: int = 1000) -> Dict[str, Any]:
        """
        Ingest data from CSV file.

        Args:
            csv_file: Path to CSV file
            batch_size: Batch size for insertion

        Returns:
            Dictionary with ingestion results
        """
        if not self.db:
            return {"success": False, "error": "Pipeline not initialized"}

        try:
            logger.info(f"Ingesting data from {csv_file}")

            # Check if file exists
            if not Path(csv_file).exists():
                return {"success": False, "error": f"CSV file not found: {csv_file}"}

            # Perform ingestion
            result = ingest_puzzle_pieces_csv(csv_file, self.db_config)

            logger.info(f"Ingestion completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return {"success": False, "error": str(e)}

    def perform_query(self, query_type: str, **kwargs) -> Any:
        """
        Perform a geospatial query.

        Args:
            query_type: Type of query
            **kwargs: Query parameters

        Returns:
            Query results
        """
        if not self.db:
            return {"success": False, "error": "Pipeline not initialized"}

        try:
            logger.info(f"Performing query: {query_type}")
            result = perform_geospatial_query(query_type, self.db, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {"success": False, "error": str(e)}

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with statistics
        """
        return self.perform_query("statistics")

    def cleanup(self):
        """
        Clean up resources.
        """
        if self.db:
            self.db.disconnect()
            logger.info("Pipeline cleaned up")


def main():
    """
    Main entry point for the pipeline.
    """
    parser = argparse.ArgumentParser(description="AI Puzzle Pieces Data Pipeline")
    parser.add_argument(
        "action", choices=["init", "ingest", "query", "stats"], help="Action to perform"
    )
    parser.add_argument("--csv", help="Path to CSV file for ingestion")
    parser.add_argument("--query-type", help="Type of query to perform")
    parser.add_argument("--host", default="localhost", help="Database host")
    parser.add_argument("--port", type=int, default=5432, help="Database port")
    parser.add_argument("--database", default="terra_constellata", help="Database name")
    parser.add_argument("--user", default="postgres", help="Database user")
    parser.add_argument("--password", default="", help="Database password")
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Batch size for ingestion"
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Database configuration
    db_config = {
        "host": args.host,
        "port": args.port,
        "database": args.database,
        "user": args.user,
        "password": args.password,
    }

    # Create pipeline
    pipeline = PuzzlePiecesPipeline(db_config)

    try:
        if args.action == "init":
            # Initialize database
            if pipeline.initialize():
                print("Database initialized successfully")
                return 0
            else:
                print("Database initialization failed")
                return 1

        elif args.action == "ingest":
            # Initialize if needed
            if not pipeline.initialize():
                print("Failed to initialize pipeline")
                return 1

            # Ingest CSV
            if not args.csv:
                print("CSV file path required for ingestion")
                return 1

            result = pipeline.ingest_csv(args.csv, args.batch_size)

            if result["success"]:
                print("Ingestion completed successfully:")
                print(f"  Records processed: {result.get('records_processed', 0)}")
                print(f"  Records inserted: {result.get('records_inserted', 0)}")
                return 0
            else:
                print(f"Ingestion failed: {result.get('error', 'Unknown error')}")
                return 1

        elif args.action == "query":
            # Initialize if needed
            if not pipeline.initialize():
                print("Failed to initialize pipeline")
                return 1

            # Perform query
            if not args.query_type:
                print("Query type required")
                return 1

            result = pipeline.perform_query(args.query_type)
            print(f"Query result: {result}")
            return 0

        elif args.action == "stats":
            # Initialize if needed
            if not pipeline.initialize():
                print("Failed to initialize pipeline")
                return 1

            # Get statistics
            stats = pipeline.get_statistics()
            print("Database Statistics:")
            print(f"  Total records: {stats.get('total_count', 'N/A')}")
            if "bounding_box" in stats and stats["bounding_box"]:
                bbox = stats["bounding_box"]
                print(
                    f"  Bounding box: {bbox['min_lon']:.4f}, {bbox['min_lat']:.4f} to {bbox['max_lon']:.4f}, {bbox['max_lat']:.4f}"
                )
            if "entity_distribution" in stats:
                print("  Entity distribution:")
                for entity in stats["entity_distribution"][:5]:  # Top 5
                    print(f"    {entity['entity']}: {entity['count']}")
            return 0

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return 1

    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    sys.exit(main())
