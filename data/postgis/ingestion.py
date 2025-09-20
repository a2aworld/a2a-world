import os
import logging
import pandas as pd
from sqlalchemy.orm import Session
from .connection import get_db, Base, engine
from . import models, schema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This path points to where the CSV will be inside the Docker container
CSV_FILE_PATH = os.getenv("CSV_FILE_PATH", "/data/Puzzle_Pieces.csv")

def parse_polygon_string(poly_string: str) -> str:
    """
    Parses the unique coordinate string format from the CSV into a standard
    Well-Known Text (WKT) POLYGON string that PostGIS can understand.
    """
    try:
        # Clean up the string by removing newlines and extra whitespace
        coords_text = ' '.join(poly_string.split())
        
        # Split into coordinate groups. The data uses ',0 ' as a separator.
        coord_pairs = coords_text.split(',0 ')
        
        points = []
        for pair in coord_pairs:
            # Clean up each pair and ensure it contains a comma
            clean_pair = pair.strip()
            if ',' in clean_pair:
                try:
                    # The format is lon,lat so we split by the comma
                    lon, lat = map(float, clean_pair.split(','))
                    points.append(f"{lon} {lat}")
                except ValueError:
                    logger.warning(f"Could not parse coordinate pair: {pair}")
                    continue # Skip malformed pairs

        # A valid polygon needs at least 4 points to define an area and close
        if len(points) < 4:
            return None

        # A WKT polygon must be "closed" (the first and last points are identical)
        if points[0] != points[-1]:
            points.append(points[0])

        # Format the final WKT string
        return f"POLYGON(({', '.join(points)}))"
    except Exception as e:
        logger.error(f"Could not parse polygon string: {poly_string[:70]}... | Error: {e}")
        return None

def seed_puzzle_pieces():
    """
    Reads the puzzle pieces CSV, processes the data, and ingests it into the
    PostGIS database. This function is designed to be idempotent (it won't
    create duplicates if run multiple times).
    """
    logger.info(f"Starting database seeding process from {CSV_FILE_PATH}...")
    
    if not os.path.exists(CSV_FILE_PATH):
        message = "Critical Error: Puzzle_Pieces.csv not found inside the container."
        logger.error(f"{message} Expected at: {CSV_FILE_PATH}")
        return {"status": "error", "message": message}

    try:
        df = pd.read_csv(CSV_FILE_PATH, header=None)
        # Assign names to the columns since the CSV has no header row
        df.columns = ['name_desc', 'image_html', 'polygon_str']
        logger.info(f"Successfully loaded {len(df)} rows from CSV.")
    except Exception as e:
        message = f"Failed to read or process CSV file: {e}"
        logger.error(message)
        return {"status": "error", "message": message}

    db: Session = next(get_db())
    
    # IMPORTANT: Check if the table has already been seeded to prevent duplicates.
    if db.query(models.PuzzlePiece).count() > 0:
        message = "Database has already been seeded with puzzle pieces. Skipping."
        logger.info(message)
        db.close()
        return {"status": "skipped", "message": message}

    count = 0
    errors = 0
    
    for _, row in df.iterrows():
        try:
            name_desc = row['name_desc']
            polygon_str = row['polygon_str']

            # Basic data validation for the row
            if not isinstance(name_desc, str) or not isinstance(polygon_str, str):
                logger.warning(f"Skipping row with invalid data types: {row}")
                errors += 1
                continue

            # Parse "Subject - Description" format from the first column
            parts = name_desc.split(' - ')
            name = parts[0].strip()
            description = parts[1].strip() if len(parts) > 1 else None

            # Parse coordinates and convert to the standard WKT format
            wkt_polygon = parse_polygon_string(polygon_str)
            if not wkt_polygon:
                logger.warning(f"Skipping row due to invalid polygon data for '{name}'")
                errors += 1
                continue

            # Create a data object using our Pydantic schema
            puzzle_piece_data = schema.PuzzlePieceCreate(
                name=name,
                description=description,
                geom=wkt_polygon
            )
            
            # Convert the schema object into a database model and add to session
            db_item = models.PuzzlePiece(**puzzle_piece_data.dict())
            db.add(db_item)
            count += 1

        except Exception as e:
            logger.error(f"Failed to process row: {row['name_desc']}. Error: {e}")
            errors += 1
            db.rollback() # Important: undo the failed addition
            continue

    try:
        # Commit all the new pieces to the database in a single transaction
        db.commit()
        message = f"Successfully committed {count} new puzzle pieces to the database."
        logger.info(message)
    except Exception as e:
        db.rollback()
        message = f"Database commit failed: {e}"
        logger.error(message)
        return {"status": "error", "message": message}
    finally:
        db.close()

    return {
        "status": "success",
        "message": message,
        "pieces_added": count,
        "rows_with_errors": errors,
    }

def create_database_tables():
    """
    Creates all tables in the database that are defined in our models.
    This is a crucial step for the first time the application runs.
    """
    try:
        logger.info("Creating database tables if they don't exist...")
        Base.metadata.create_all(bind=engine)
        logger.info("Tables created successfully.")
    except Exception as e:
        logger.error(f"An error occurred while creating tables: {e}")
        raise