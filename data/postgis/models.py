from sqlalchemy import Column, Integer, String, Text
from geoalchemy2 import Geometry
from .connection import Base

class PuzzlePiece(Base):
    """
    SQLAlchemy model for the 'puzzle_pieces' table.
    
    This class defines the structure of the table in the PostGIS database,
    including the special 'geom' column for storing geospatial data.
    """
    __tablename__ = "puzzle_pieces"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text, nullable=True)
    
    # This special column type from GeoAlchemy2 tells the database
    # to store geospatial POLYGON data using the SRID 4326 standard (WGS 84).
    geom = Column(Geometry(geometry_type='POLYGON', srid=4326), nullable=False)