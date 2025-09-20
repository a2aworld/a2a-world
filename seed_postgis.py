import os
import time
import pandas as pd
import psycopg2
from psycopg2 import sql
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Database Connection Details from Environment Variables ---
DB_NAME = os.getenv("POSTGRES_DB", "terra_constellata")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "1234")
DB_HOST = "localhost" # Inside the container, it connects to itself
DB_PORT = "5432"

# --- Data Details ---
CSV_FILE_PATH = '/data/puzzle_pieces.csv' # Path inside the Docker container
TABLE_NAME = 'puzzle_pieces'
# IMPORTANT: Update these with the actual column names from your CSV
LAT_COLUMN = 'latitude'
LON_COLUMN = 'longitude'

def wait_for_db():
    """Waits for the database to be ready to accept connections."""
    logging.info("Waiting for database to become available...")
    retries = 10
    while retries > 0:
        try:
            conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            )
            conn.close()
            logging.info("Database is available!")
            return True
        except psycopg2.OperationalError:
            retries -= 1
            logging.info(f"Database not available yet. Retrying in 5 seconds... ({retries} retries left)")
            time.sleep(5)
    logging.error("Could not connect to the database after several retries.")
    return False

def seed_data():
    """Reads data from CSV and inserts it into the PostGIS table."""
    if not wait_for_db():
        return

    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cur = conn.cursor()
        logging.info(f"Successfully connected to database '{DB_NAME}'")

        # --- Create Table with a GEOMETRY column ---
        # This assumes a simple table structure. Adjust columns and types as needed.
        create_table_query = sql.SQL("""
        CREATE TABLE IF NOT EXISTS {table} (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255),
            description TEXT,
            {lat} DOUBLE PRECISION,
            {lon} DOUBLE PRECISION,
            geom GEOMETRY(Point, 4326) -- SRID 4326 is standard for GPS data
        );
        """).format(
            table=sql.Identifier(TABLE_NAME),
            lat=sql.Identifier(LAT_COLUMN),
            lon=sql.Identifier(LON_COLUMN)
        )
        cur.execute(create_table_query)
        conn.commit()
        logging.info(f"Table '{TABLE_NAME}' created or already exists.")

        # --- Read CSV and Insert Data ---
        df = pd.read_csv(CSV_FILE_PATH)
        logging.info(f"Read {len(df)} rows from {CSV_FILE_PATH}")

        for _, row in df.iterrows():
            # Create a PostGIS-compatible point from latitude and longitude
            # ST_GeomFromText creates a geometry object from Well-Known Text (WKT) format
            # The '4326' is the Spatial Reference System Identifier for WGS 84 (standard GPS coordinates)
            geom_wkt = f"POINT({row[LON_COLUMN]} {row[LAT_COLUMN]})"
            
            # This assumes your CSV has 'name' and 'description' columns.
            # Adjust the INSERT statement to match your CSV columns.
            insert_query = sql.SQL("""
            INSERT INTO {table} (name, description, {lat}, {lon}, geom)
            VALUES (%s, %s, %s, %s, ST_GeomFromText(%s, 4326));
            """).format(
                table=sql.Identifier(TABLE_NAME),
                lat=sql.Identifier(LAT_COLUMN),
                lon=sql.Identifier(LON_COLUMN)
            )
            
            cur.execute(insert_query, (
                row['name'], 
                row['description'], 
                row[LAT_COLUMN], 
                row[LON_COLUMN], 
                geom_wkt
            ))

        conn.commit()
        logging.info("Data seeding complete.")

    except Exception as e:
        logging.error(f"An error occurred during seeding: {e}")
    finally:
        if 'conn' in locals() and conn:
            cur.close()
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    seed_data()