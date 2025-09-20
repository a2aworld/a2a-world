from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from .. import crud, models, schemas
from typing import List

router = APIRouter()


@router.post("/", response_model=schemas.Content)
def create_content(content: schemas.ContentCreate, db: Session = Depends(get_db)):
    return crud.create_content(db=db, content=content)


@router.get("/", response_model=List[schemas.Content])
def read_contents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    contents = crud.get_contents(db, skip=skip, limit=limit)
    return contents


@router.get("/{content_id}", response_model=schemas.Content)
def read_content(content_id: int, db: Session = Depends(get_db)):
    db_content = crud.get_content(db, content_id=content_id)
    if db_content is None:
        raise HTTPException(status_code=404, detail="Content not found")
    return db_content


@router.put("/{content_id}", response_model=schemas.Content)
def update_content(
    content_id: int, content: schemas.ContentUpdate, db: Session = Depends(get_db)
):
    db_content = crud.update_content(db, content_id=content_id, content_update=content)
    if db_content is None:
        raise HTTPException(status_code=404, detail="Content not found")
    return db_content


@router.delete("/{content_id}")
def delete_content(content_id: int, db: Session = Depends(get_db)):
    db_content = crud.delete_content(db, content_id=content_id)
    if db_content is None:
        raise HTTPException(status_code=404, detail="Content not found")
    return {"message": "Content deleted successfully"}
