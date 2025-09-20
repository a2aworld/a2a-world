from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from .. import crud, schemas

router = APIRouter()


@router.post("/", response_model=schemas.Map)
def create_map(map_data: schemas.MapCreate, db: Session = Depends(get_db)):
    return crud.create_map(db=db, map_data=map_data)


@router.get("/{content_id}")
def get_maps_for_content(content_id: int, db: Session = Depends(get_db)):
    # This would need a query to get maps by content_id
    # For now, return placeholder
    return {"message": f"Maps for content {content_id}"}
