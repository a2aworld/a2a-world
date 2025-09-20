import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

logger = logging.getLogger(__name__)

# --- Database Connection Details ---
# These are read from the environment variables set in your docker-compose.yml
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "1234")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres") # 'postgres' is the service name in Docker
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "terra_constellata")

# Create the full database URL that SQLAlchemy will use
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

try:
    # The engine is the core entry point to the database
    engine = create_engine(DATABASE_URL)

    # The SessionLocal class is a factory that creates new database session objects
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # This is a base class that our database table models will inherit from
    Base = declarative_base()

    logger.info("Database connection pool established successfully.")

except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    # If we can't connect to the database, the application can't run.
    exit(1)

def get_db():
    """
    This is a helper function that our API endpoints will use.
    It creates a new database session for a single request and ensures
    it's always closed afterward, which is very important for stability.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()