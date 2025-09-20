#!/usr/bin/env python3
"""
PostgreSQL/PostGIS Connection Module for Terra Constellata

This module provides database connection functionality for the AI Puzzle Pieces Data Pipeline.
"""

import psycopg2
import psycopg2.extras
import logging
from psycopg2 import sql
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class PostGISConnection:
    """
    Manages PostgreSQL/PostGIS database connections and operations.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "terra_constellata",
        user: str = "postgres",
        password: str = "",
        sslmode: str = "prefer",
    ):
        """
        Initialize database connection parameters.

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            sslmode: SSL connection mode
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.sslmode = sslmode
        self.connection = None
        self.cursor = None

    def connect(self) -> bool:
        """
        Establish database connection.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                sslmode=self.sslmode,
            )
            self.connection.autocommit = False
            self.cursor = self.connection.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            )
            logger.info(f"Connected to PostgreSQL database: {self.database}")
            return True
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    def disconnect(self):
        """
        Close database connection.
        """
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

    def execute_query(
        self, query: str, params: Optional[tuple] = None
    ) -> Optional[list]:
        """
        Execute a SELECT query and return results.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of result rows as dictionaries
        """
        try:
            self.cursor.execute(query, params)
            if self.cursor.description:
                return self.cursor.fetchall()
            return []
        except psycopg2.Error as e:
            logger.error(f"Query execution failed: {e}")
            self.connection.rollback()
            raise

    def execute_command(self, command: str, params: Optional[tuple] = None) -> bool:
        """
        Execute an INSERT, UPDATE, or DELETE command.

        Args:
            command: SQL command string
            params: Command parameters

        Returns:
            bool: True if successful
        """
        try:
            self.cursor.execute(command, params)
            self.connection.commit()
            return True
        except psycopg2.Error as e:
            logger.error(f"Command execution failed: {e}")
            self.connection.rollback()
            raise

    def create_database_if_not_exists(self) -> bool:
        """
        Create the database if it doesn't exist.

        Returns:
            bool: True if database exists or was created successfully
        """
        try:
            # Connect to default postgres database to create our database
            temp_conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database="postgres",
                user=self.user,
                password=self.password,
                sslmode=self.sslmode,
            )
            temp_conn.autocommit = True
            temp_cursor = temp_conn.cursor()

            # Check if database exists
            temp_cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (self.database,)
            )

            if not temp_cursor.fetchone():
                temp_cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.database))
                )
                logger.info(f"Created database: {self.database}")

            temp_cursor.close()
            temp_conn.close()
            return True

        except psycopg2.Error as e:
            logger.error(f"Failed to create database: {e}")
            return False

    def enable_postgis(self) -> bool:
        """
        Enable PostGIS extension in the database.

        Returns:
            bool: True if PostGIS is enabled
        """
        try:
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
            self.connection.commit()
            logger.info("PostGIS extension enabled")
            return True
        except psycopg2.Error as e:
            logger.error(f"Failed to enable PostGIS: {e}")
            return False

    def check_postgis_version(self) -> Optional[str]:
        """
        Check PostGIS version.

        Returns:
            PostGIS version string or None if not available
        """
        try:
            self.cursor.execute("SELECT PostGIS_Version();")
            result = self.cursor.fetchone()
            if result:
                return result["postgis_version"]
            return None
        except psycopg2.Error as e:
            logger.error(f"Failed to check PostGIS version: {e}")
            return None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


def get_db_connection(
    host: str = "localhost",
    port: int = 5432,
    database: str = "terra_constellata",
    user: str = "postgres",
    password: str = "",
) -> PostGISConnection:
    """
    Get a database connection instance.

    Args:
        host: Database host
        port: Database port
        database: Database name
        user: Database user
        password: Database password

    Returns:
        PostGISConnection instance
    """
    return PostGISConnection(host, port, database, user, password)


if __name__ == "__main__":
    # Test connection
    logging.basicConfig(level=logging.INFO)

    db = get_db_connection()
    if db.connect():
        db.create_database_if_not_exists()
        db.enable_postgis()
        version = db.check_postgis_version()
        if version:
            print(f"PostGIS version: {version}")
        db.disconnect()
    else:
        print("Failed to connect to database")
