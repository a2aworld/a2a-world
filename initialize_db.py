#!/usr/bin/env python3
"""
Initializes the PostGIS database by creating the required tables and extensions.
This is a standalone script to be run once during setup.
"""
import logging
import sys
from pathlib import Path

# Add the project root to the Python path to allow for module imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.postgis.connection import PostGISConnection
from data.postgis.schema import initialize_database

# Configure basic logging to see output in the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main function to connect and initialize the database."""
    logging.info("Starting database initialization process...")
    
    db_connection = PostGISConnection()
    
    if db_connection.connect():
        try:
            if initialize_database(db_connection):
                logging.info("--- DATABASE INITIALIZATION COMPLETE ---")
            else:
                logging.error("--- DATABASE INITIALIZATION FAILED ---")
        finally:
            db_connection.disconnect()
    else:
        logging.error("Could not establish database connection. Please check your.env file and ensure Docker containers are running.")

if __name__ == "__main__":
    main()