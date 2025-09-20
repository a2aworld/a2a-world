"""
Feedback collection API endpoints for Terra Constellata.
Handles user feedback, ratings, and continuous improvement data.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
import uuid
from ...logging_config import app_logger, log_business_event
from ...metrics import record_feedback
from ...learning.feedback_collector import FeedbackCollector, UserFeedback

router = APIRouter()


class FeedbackSubmission(BaseModel):
    """Model for user feedback submission."""

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    feedback_type: str = Field(
        ...,
        description="Type of feedback (rating, comment, bug_report, feature_request)",
    )
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating from 1-5")
    comment: Optional[str] = Field(None, max_length=1000)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class WorkflowFeedback(BaseModel):
    """Model for workflow-specific feedback."""

    workflow_id: str
    user_id: Optional[str] = None
    stage: str
    rating: int = Field(..., ge=1, le=5)
    feedback: Optional[str] = Field(None, max_length=500)
    improvement_suggestions: Optional[str] = Field(None, max_length=500)


class ArtworkFeedback(BaseModel):
    """Model for artwork-specific feedback."""

    artwork_id: str
    user_id: Optional[str] = None
    rating: int = Field(..., ge=1, le=5)
    style_feedback: Optional[str] = Field(None, max_length=500)
    content_feedback: Optional[str] = Field(None, max_length=500)
    tags: Optional[list[str]] = Field(default_factory=list)


# Initialize feedback collector
feedback_collector = FeedbackCollector()


@router.post("/submit", response_model=Dict[str, Any])
async def submit_feedback(
    feedback: FeedbackSubmission, background_tasks: BackgroundTasks
):
    """Submit general user feedback."""
    try:
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        feedback_data = {
            "id": feedback_id,
            "timestamp": timestamp.isoformat(),
            **feedback.dict(),
        }

        # Record metrics
        if feedback.rating:
            record_feedback(feedback.feedback_type, feedback.rating)

        # Log business event
        log_business_event(
            "feedback_submitted",
            {
                "feedback_id": feedback_id,
                "type": feedback.feedback_type,
                "rating": feedback.rating,
                "has_comment": bool(feedback.comment),
            },
            feedback.user_id,
        )

        # Store feedback asynchronously
        background_tasks.add_task(store_general_feedback, feedback_data)

        app_logger.info(
            f"Feedback submitted: {feedback_id}",
            extra={
                "feedback_id": feedback_id,
                "type": feedback.feedback_type,
                "rating": feedback.rating,
            },
        )

        return {
            "status": "success",
            "feedback_id": feedback_id,
            "message": "Feedback submitted successfully",
        }

    except Exception as e:
        app_logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")


@router.post("/workflow/{workflow_id}", response_model=Dict[str, Any])
async def submit_workflow_feedback(
    workflow_id: str, feedback: WorkflowFeedback, background_tasks: BackgroundTasks
):
    """Submit feedback for a specific workflow."""
    try:
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        feedback_data = {
            "id": feedback_id,
            "timestamp": timestamp.isoformat(),
            "workflow_id": workflow_id,
            **feedback.dict(),
        }

        # Record metrics
        record_feedback("workflow", feedback.rating)

        # Log business event
        log_business_event(
            "workflow_feedback_submitted",
            {
                "feedback_id": feedback_id,
                "workflow_id": workflow_id,
                "stage": feedback.stage,
                "rating": feedback.rating,
            },
            feedback.user_id,
        )

        # Store feedback asynchronously
        background_tasks.add_task(store_workflow_feedback, workflow_id, feedback_data)

        app_logger.info(
            f"Workflow feedback submitted: {feedback_id}",
            extra={
                "feedback_id": feedback_id,
                "workflow_id": workflow_id,
                "stage": feedback.stage,
                "rating": feedback.rating,
            },
        )

        return {
            "status": "success",
            "feedback_id": feedback_id,
            "message": "Workflow feedback submitted successfully",
        }

    except Exception as e:
        app_logger.error(f"Error submitting workflow feedback: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to submit workflow feedback"
        )


@router.post("/artwork/{artwork_id}", response_model=Dict[str, Any])
async def submit_artwork_feedback(
    artwork_id: str, feedback: ArtworkFeedback, background_tasks: BackgroundTasks
):
    """Submit feedback for a specific artwork."""
    try:
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        feedback_data = {
            "id": feedback_id,
            "timestamp": timestamp.isoformat(),
            "artwork_id": artwork_id,
            **feedback.dict(),
        }

        # Record metrics
        record_feedback("artwork", feedback.rating)

        # Log business event
        log_business_event(
            "artwork_feedback_submitted",
            {
                "feedback_id": feedback_id,
                "artwork_id": artwork_id,
                "rating": feedback.rating,
                "tags": feedback.tags,
            },
            feedback.user_id,
        )

        # Store feedback asynchronously
        background_tasks.add_task(store_artwork_feedback, artwork_id, feedback_data)

        app_logger.info(
            f"Artwork feedback submitted: {feedback_id}",
            extra={
                "feedback_id": feedback_id,
                "artwork_id": artwork_id,
                "rating": feedback.rating,
            },
        )

        return {
            "status": "success",
            "feedback_id": feedback_id,
            "message": "Artwork feedback submitted successfully",
        }

    except Exception as e:
        app_logger.error(f"Error submitting artwork feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit artwork feedback")


@router.get("/stats")
async def get_feedback_stats():
    """Get aggregated feedback statistics."""
    try:
        stats = await feedback_collector.get_feedback_stats()
        return {"status": "success", "data": stats}
    except Exception as e:
        app_logger.error(f"Error getting feedback stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get feedback statistics")


@router.get("/recent")
async def get_recent_feedback(limit: int = 10):
    """Get recent feedback entries."""
    try:
        # Get recent feedback from collector
        all_feedback = list(feedback_collector.feedback_data.values())
        recent_feedback = sorted(all_feedback, key=lambda x: x.timestamp, reverse=True)[
            :limit
        ]

        return {"status": "success", "data": [f.to_dict() for f in recent_feedback]}
    except Exception as e:
        app_logger.error(f"Error getting recent feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recent feedback")


# Helper functions for async feedback storage
def store_general_feedback(feedback_data: dict):
    """Store general feedback asynchronously."""
    try:
        # Convert to UserFeedback format
        user_feedback = UserFeedback(
            workflow_id=feedback_data.get(
                "workflow_id", f"general_{feedback_data['id']}"
            ),
            user_id=feedback_data.get("user_id", "anonymous"),
        )

        # Add ratings if available
        if feedback_data.get("rating"):
            user_feedback.set_ratings(satisfaction=feedback_data["rating"] / 5.0)

        if feedback_data.get("comment"):
            user_feedback.add_feedback_text(feedback_data["comment"])

        # Add metadata
        user_feedback.metadata = feedback_data.get("metadata", {})

        feedback_collector.submit_feedback(user_feedback)
        app_logger.info(f"General feedback stored: {feedback_data['id']}")

    except Exception as e:
        app_logger.error(f"Error storing general feedback: {e}")


def store_workflow_feedback(workflow_id: str, feedback_data: dict):
    """Store workflow feedback asynchronously."""
    try:
        user_feedback = UserFeedback(
            workflow_id, feedback_data.get("user_id", "anonymous")
        )

        # Add ratings
        user_feedback.set_ratings(satisfaction=feedback_data["rating"] / 5.0)

        if feedback_data.get("feedback"):
            user_feedback.add_feedback_text(feedback_data["feedback"])

        if feedback_data.get("improvement_suggestions"):
            for suggestion in feedback_data["improvement_suggestions"]:
                user_feedback.add_suggestion(suggestion)

        feedback_collector.submit_feedback(user_feedback)
        app_logger.info(f"Workflow feedback stored: {workflow_id}")

    except Exception as e:
        app_logger.error(f"Error storing workflow feedback: {e}")


def store_artwork_feedback(artwork_id: str, feedback_data: dict):
    """Store artwork feedback asynchronously."""
    try:
        workflow_id = f"artwork_{artwork_id}"
        user_feedback = UserFeedback(
            workflow_id, feedback_data.get("user_id", "anonymous")
        )

        # Add ratings
        user_feedback.set_ratings(quality=feedback_data["rating"] / 5.0)

        if feedback_data.get("style_feedback"):
            user_feedback.add_feedback_text(f"Style: {feedback_data['style_feedback']}")

        if feedback_data.get("content_feedback"):
            user_feedback.add_feedback_text(
                f"Content: {feedback_data['content_feedback']}"
            )

        if feedback_data.get("tags"):
            for tag in feedback_data["tags"]:
                user_feedback.add_tag(tag)

        feedback_collector.submit_feedback(user_feedback)
        app_logger.info(f"Artwork feedback stored: {artwork_id}")

    except Exception as e:
        app_logger.error(f"Error storing artwork feedback: {e}")
