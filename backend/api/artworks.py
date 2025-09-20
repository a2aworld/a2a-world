from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from .. import crud, schemas

router = APIRouter()


@router.post("/", response_model=schemas.Artwork)
def create_artwork(artwork: schemas.ArtworkCreate, db: Session = Depends(get_db)):
    return crud.create_artwork(db=db, artwork=artwork)


@router.get("/{multimedia_id}")
def get_artworks_for_multimedia(multimedia_id: int, db: Session = Depends(get_db)):
    # This would need a query to get artworks by multimedia_id
    # For now, return placeholder
    return {"message": f"Artworks for multimedia {multimedia_id}"}
