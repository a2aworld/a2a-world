from sqlalchemy.orm import Session
from . import models, schemas
from typing import List


# Content CRUD
def get_content(db: Session, content_id: int):
    return db.query(models.Content).filter(models.Content.id == content_id).first()


def get_contents(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Content).offset(skip).limit(limit).all()


def create_content(db: Session, content: schemas.ContentCreate):
    db_content = models.Content(**content.dict(exclude={"tag_ids"}))
    if content.tag_ids:
        tags = db.query(models.Tag).filter(models.Tag.id.in_(content.tag_ids)).all()
        db_content.tags = tags
    db.add(db_content)
    db.commit()
    db.refresh(db_content)
    return db_content


def update_content(db: Session, content_id: int, content_update: schemas.ContentUpdate):
    db_content = (
        db.query(models.Content).filter(models.Content.id == content_id).first()
    )
    if db_content:
        update_data = content_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_content, field, value)
        db.commit()
        db.refresh(db_content)
    return db_content


def delete_content(db: Session, content_id: int):
    db_content = (
        db.query(models.Content).filter(models.Content.id == content_id).first()
    )
    if db_content:
        db.delete(db_content)
        db.commit()
    return db_content


# Multimedia CRUD
def create_multimedia(
    db: Session, multimedia: schemas.MultimediaCreate, file_path: str
):
    db_multimedia = models.Multimedia(**multimedia.dict(), file_path=file_path)
    db.add(db_multimedia)
    db.commit()
    db.refresh(db_multimedia)
    return db_multimedia


def get_multimedia(db: Session, multimedia_id: int):
    return (
        db.query(models.Multimedia)
        .filter(models.Multimedia.id == multimedia_id)
        .first()
    )


# Tag CRUD
def get_or_create_tag(db: Session, tag_name: str):
    tag = db.query(models.Tag).filter(models.Tag.name == tag_name).first()
    if not tag:
        tag = models.Tag(name=tag_name)
        db.add(tag)
        db.commit()
        db.refresh(tag)
    return tag


# Map CRUD
def create_map(db: Session, map_data: schemas.MapCreate):
    db_map = models.Map(**map_data.dict())
    db.add(db_map)
    db.commit()
    db.refresh(db_map)
    return db_map


# Artwork CRUD
def create_artwork(db: Session, artwork: schemas.ArtworkCreate):
    db_artwork = models.Artwork(**artwork.dict())
    db.add(db_artwork)
    db.commit()
    db.refresh(db_artwork)
    return db_artwork


# CAT Score based publishing
def get_high_cat_score_contents(db: Session, threshold: float = 0.8):
    return (
        db.query(models.Content)
        .filter(
            models.Content.cat_score >= threshold, models.Content.is_published == False
        )
        .all()
    )


def publish_content(db: Session, content_id: int):
    content = db.query(models.Content).filter(models.Content.id == content_id).first()
    if content:
        content.is_published = True
        db.commit()
        db.refresh(content)
    return content
