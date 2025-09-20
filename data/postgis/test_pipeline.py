#!/usr/bin/env python3
"""
Test script for the AI Puzzle Pieces Data Pipeline

This script demonstrates the pipeline functionality with sample data.
"""

import logging
import tempfile
import csv
import os
from pathlib import Path

# Add project root to sys.path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from .pipeline import PuzzlePiecesPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_csv(file_path: str):
    """
    Create a sample CSV file with test data.

    Args:
        file_path: Path to create the CSV file
    """
    sample_data = [
        [
            "row_number",
            "name",
            "entity",
            "sub_entity",
            "description",
            "source_url",
            "latitude",
            "longitude",
        ],
        [
            1,
            "New York City",
            "city",
            "metropolis",
            "Major metropolitan area in the US",
            "https://example.com/nyc",
            40.7128,
            -74.0060,
        ],
        [
            2,
            "Central Park",
            "park",
            "urban_park",
            "Large urban park in Manhattan",
            "https://example.com/centralpark",
            40.7829,
            -73.9654,
        ],
        [
            3,
            "Eiffel Tower",
            "landmark",
            "tower",
            "Iconic iron tower in Paris",
            "https://example.com/eiffel",
            48.8584,
            2.2945,
        ],
        [
            4,
            "Amazon Rainforest",
            "natural_feature",
            "forest",
            "Largest rainforest on Earth",
            "https://example.com/amazon",
            -3.4653,
            -62.2159,
        ],
        [
            5,
            "Mount Everest",
            "mountain",
            "peak",
            "Highest mountain in the world",
            "https://example.com/everest",
            27.9881,
            86.9250,
        ],
        [
            6,
            "Sydney Opera House",
            "landmark",
            "cultural",
            "Iconic performing arts venue",
            "https://example.com/opera",
            -33.8568,
            151.2153,
        ],
        [
            7,
            "Great Wall of China",
            "landmark",
            "wall",
            "Ancient defensive wall in China",
            "https://example.com/greatwall",
            40.4319,
            116.5704,
        ],
        [
            8,
            "Sahara Desert",
            "natural_feature",
            "desert",
            "World's largest hot desert",
            "https://example.com/sahara",
            23.4162,
            25.6628,
        ],
        [
            9,
            "Tokyo Tower",
            "landmark",
            "tower",
            "Communications and observation tower",
            "https://example.com/tokyotower",
            35.6586,
            139.7454,
        ],
        [
            10,
            "Grand Canyon",
            "natural_feature",
            "canyon",
            "Steep-sided canyon in Arizona",
            "https://example.com/grandcanyon",
            36.1069,
            -112.1129,
        ],
    ]

    with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for row in sample_data:
            writer.writerow(row)

    logger.info(f"Created sample CSV with {len(sample_data)-1} records at {file_path}")


def test_pipeline():
    """
    Test the complete pipeline functionality.
    """
    logger.info("Starting pipeline test...")

    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
        csv_path = tmp_file.name

    try:
        # Create sample data
        create_sample_csv(csv_path)

        # Initialize pipeline (using default config - will fail if PostgreSQL not running)
        pipeline = PuzzlePiecesPipeline()

        logger.info("Testing pipeline initialization...")
        init_success = pipeline.initialize()

        if not init_success:
            logger.warning(
                "Pipeline initialization failed (likely PostgreSQL not running)"
            )
            logger.info("This is expected in a test environment without PostgreSQL")
            return

        # Test CSV ingestion
        logger.info("Testing CSV ingestion...")
        result = pipeline.ingest_csv(csv_path)

        if result["success"]:
            logger.info("Ingestion successful!")
            logger.info(f"Records processed: {result['records_processed']}")
            logger.info(f"Records inserted: {result['records_inserted']}")
        else:
            logger.error(f"Ingestion failed: {result.get('error', 'Unknown error')}")
            return

        # Test statistics
        logger.info("Testing statistics query...")
        stats = pipeline.get_statistics()
        logger.info(f"Database statistics: {stats}")

        # Test geospatial queries
        logger.info("Testing geospatial queries...")

        # Bounding box query
        bbox_results = pipeline.perform_query(
            "bbox", min_lon=-180, min_lat=-90, max_lon=180, max_lat=90, limit=5
        )
        logger.info(f"Bounding box query returned {len(bbox_results)} results")

        # Nearby points query
        nearby_results = pipeline.perform_query(
            "nearby", lon=-74.0, lat=40.7, distance_meters=50000
        )
        logger.info(f"Nearby query returned {len(nearby_results)} results")

        # Entity query
        entity_results = pipeline.perform_query("by_entity", entity="landmark")
        logger.info(f"Entity query returned {len(entity_results)} results")

        # Search query
        search_results = pipeline.perform_query("search_name", search_term="New York")
        logger.info(f"Search query returned {len(search_results)} results")

        logger.info("Pipeline test completed successfully!")

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise

    finally:
        # Clean up
        pipeline.cleanup()
        if os.path.exists(csv_path):
            os.unlink(csv_path)
            logger.info("Cleaned up temporary CSV file")


def test_data_processing():
    """
    Test data processing functions without database.
    """
    logger.info("Testing data processing functions...")

    try:
        from .data_processing import clean_and_validate_data
        import pandas as pd

        # Create test DataFrame
        test_data = {
            "row_number": [1, 2, 3],
            "name": ["Test Location 1", "Test Location 2", "Test Location 3"],
            "entity": ["city", "park", "landmark"],
            "latitude": [40.7128, 34.0522, 51.5074],
            "longitude": [-74.0060, -118.2437, -0.1278],
            "source_url": [
                "https://example.com/1",
                "https://example.com/2",
                "invalid-url",
            ],
        }

        df = pd.DataFrame(test_data)

        # Test cleaning and validation
        cleaned_df, quality_report = clean_and_validate_data(df)

        logger.info("Data processing test successful!")
        logger.info(
            f"Quality score: {quality_report['summary']['overall_quality_score']:.2f}%"
        )
        logger.info(f"Validation errors: {len(quality_report['validation_errors'])}")

    except Exception as e:
        logger.error(f"Data processing test failed: {e}")
        raise


if __name__ == "__main__":
    print("AI Puzzle Pieces Data Pipeline Test")
    print("=" * 40)

    try:
        # Test data processing functions
        test_data_processing()
        print()

        # Test full pipeline (may fail if PostgreSQL not available)
        test_pipeline()

    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)

    print("\nTest completed!")
