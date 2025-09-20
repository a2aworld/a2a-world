from fastapi import APIRouter, HTTPException, status
from data.postgis.ingestion import seed_puzzle_pieces, create_database_tables

router = APIRouter()

@router.post("/initialize-database-tables", status_code=status.HTTP_201_CREATED)
def initialize_tables_endpoint():
    """
    This endpoint creates all the necessary tables in the database
    based on our SQLAlchemy models. It's a safe one-time setup.
    """
    try:
        create_database_tables()
        return {"status": "success", "message": "Database tables initialized successfully."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize database tables: {e}"
        )

@router.post("/seed-database", status_code=status.HTTP_201_CREATED)
def seed_database_endpoint():
    """
    Triggers the data seeding process to load the foundational
    'Puzzle_Pieces.csv' dataset into the PostGIS database.
    This operation is idempotent; it will not create duplicates if run again.
    """
    try:
        result = seed_puzzle_pieces()
        if result.get("status") in ["error", "skipped"]:
            # Return a different status code if the action was skipped or failed
            return result
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during seeding: {e}"
        )