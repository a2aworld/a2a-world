from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from ..database import get_db
from .. import crud
from typing import List

router = APIRouter()


def auto_publish_high_cat_score(db: Session, threshold: float = 0.8):
    """Background task to publish high CAT-score content"""
    contents = crud.get_high_cat_score_contents(db, threshold)
    for content in contents:
        crud.publish_content(db, content.id)
        print(
            f"Auto-published content: {content.title} (CAT-score: {content.cat_score})"
        )


@router.post("/auto-publish")
async def trigger_auto_publish(
    background_tasks: BackgroundTasks,
    threshold: float = 0.8,
    db: Session = Depends(get_db),
):
    background_tasks.add_task(auto_publish_high_cat_score, db, threshold)
    return {"message": "Auto-publishing task started"}


@router.get("/high-cat-score", response_model=List[dict])
def get_high_cat_score_contents(threshold: float = 0.8, db: Session = Depends(get_db)):
    contents = crud.get_high_cat_score_contents(db, threshold)
    return [{"id": c.id, "title": c.title, "cat_score": c.cat_score} for c in contents]


@router.post("/publish/{content_id}")
def publish_content(content_id: int, db: Session = Depends(get_db)):
    content = crud.publish_content(db, content_id)
    if not content:
        return {"error": "Content not found"}
    return {"message": f"Content '{content.title}' published successfully"}
