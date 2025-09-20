"""
Feedback Collection System for User CAT Scores

This module implements a comprehensive feedback collection system that captures
user evaluations (CAT scores) and other feedback metrics to improve the
Collective Learning Loop and agent performance.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict
import asyncio

logger = logging.getLogger(__name__)


class CATScore:
    """Represents a Creative Assessment Tool (CAT) score."""

    def __init__(
        self,
        workflow_id: str,
        user_id: str,
        score: int,
        dimensions: Optional[Dict[str, int]] = None,
        comments: Optional[str] = None,
    ):
        """
        Initialize a CAT score.

        Args:
            workflow_id: ID of the workflow being evaluated
            user_id: ID of the user providing feedback
            score: Overall CAT score (1-10)
            dimensions: Scores for specific creative dimensions
            comments: Optional user comments
        """
        self.workflow_id = workflow_id
        self.user_id = user_id
        self.score = max(1, min(10, score))  # Clamp to valid range
        self.dimensions = dimensions or {}
        self.comments = comments
        self.timestamp = datetime.utcnow()
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert CAT score to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "user_id": self.user_id,
            "score": self.score,
            "dimensions": self.dimensions,
            "comments": self.comments,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CATScore":
        """Create CAT score from dictionary."""
        score = cls(
            workflow_id=data["workflow_id"],
            user_id=data["user_id"],
            score=data["score"],
            dimensions=data.get("dimensions", {}),
            comments=data.get("comments"),
        )
        score.timestamp = datetime.fromisoformat(data["timestamp"])
        score.metadata = data.get("metadata", {})
        return score

    def get_normalized_score(self) -> float:
        """Get normalized score (0-1)."""
        return (self.score - 1) / 9.0

    def get_dimension_score(self, dimension: str) -> Optional[float]:
        """Get normalized score for a specific dimension."""
        if dimension in self.dimensions:
            return (self.dimensions[dimension] - 1) / 9.0
        return None


class UserFeedback:
    """Represents comprehensive user feedback for a workflow."""

    def __init__(self, workflow_id: str, user_id: str):
        self.workflow_id = workflow_id
        self.user_id = user_id
        self.cat_score: Optional[CATScore] = None
        self.satisfaction_rating: Optional[float] = None  # 0-1
        self.usefulness_rating: Optional[float] = None  # 0-1
        self.novelty_rating: Optional[float] = None  # 0-1
        self.quality_rating: Optional[float] = None  # 0-1
        self.feedback_text: Optional[str] = None
        self.suggested_improvements: List[str] = []
        self.tags: List[str] = []
        self.timestamp = datetime.utcnow()
        self.metadata: Dict[str, Any] = {}

    def add_cat_score(
        self,
        score: int,
        dimensions: Optional[Dict[str, int]] = None,
        comments: Optional[str] = None,
    ):
        """Add a CAT score to this feedback."""
        self.cat_score = CATScore(
            workflow_id=self.workflow_id,
            user_id=self.user_id,
            score=score,
            dimensions=dimensions,
            comments=comments,
        )

    def set_ratings(
        self,
        satisfaction: Optional[float] = None,
        usefulness: Optional[float] = None,
        novelty: Optional[float] = None,
        quality: Optional[float] = None,
    ):
        """Set various rating scores."""
        if satisfaction is not None:
            self.satisfaction_rating = max(0.0, min(1.0, satisfaction))
        if usefulness is not None:
            self.usefulness_rating = max(0.0, min(1.0, usefulness))
        if novelty is not None:
            self.novelty_rating = max(0.0, min(1.0, novelty))
        if quality is not None:
            self.quality_rating = max(0.0, min(1.0, quality))

    def add_feedback_text(self, text: str):
        """Add textual feedback."""
        self.feedback_text = text

    def add_suggestion(self, suggestion: str):
        """Add a suggested improvement."""
        self.suggested_improvements.append(suggestion)

    def add_tag(self, tag: str):
        """Add a feedback tag."""
        if tag not in self.tags:
            self.tags.append(tag)

    def get_composite_score(self) -> float:
        """Calculate a composite feedback score."""
        scores = []
        weights = []

        if self.cat_score:
            scores.append(self.cat_score.get_normalized_score())
            weights.append(0.4)  # CAT score has highest weight

        if self.satisfaction_rating is not None:
            scores.append(self.satisfaction_rating)
            weights.append(0.2)

        if self.usefulness_rating is not None:
            scores.append(self.usefulness_rating)
            weights.append(0.15)

        if self.quality_rating is not None:
            scores.append(self.quality_rating)
            weights.append(0.15)

        if self.novelty_rating is not None:
            scores.append(self.novelty_rating)
            weights.append(0.1)

        if not scores:
            return 0.5  # Default neutral score

        # Weighted average
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    def to_dict(self) -> Dict[str, Any]:
        """Convert feedback to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "user_id": self.user_id,
            "cat_score": self.cat_score.to_dict() if self.cat_score else None,
            "satisfaction_rating": self.satisfaction_rating,
            "usefulness_rating": self.usefulness_rating,
            "novelty_rating": self.novelty_rating,
            "quality_rating": self.quality_rating,
            "feedback_text": self.feedback_text,
            "suggested_improvements": self.suggested_improvements,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat(),
            "composite_score": self.get_composite_score(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserFeedback":
        """Create feedback from dictionary."""
        feedback = cls(workflow_id=data["workflow_id"], user_id=data["user_id"])

        if data.get("cat_score"):
            feedback.cat_score = CATScore.from_dict(data["cat_score"])

        feedback.satisfaction_rating = data.get("satisfaction_rating")
        feedback.usefulness_rating = data.get("usefulness_rating")
        feedback.novelty_rating = data.get("novelty_rating")
        feedback.quality_rating = data.get("quality_rating")
        feedback.feedback_text = data.get("feedback_text")
        feedback.suggested_improvements = data.get("suggested_improvements", [])
        feedback.tags = data.get("tags", [])
        feedback.timestamp = datetime.fromisoformat(data["timestamp"])
        feedback.metadata = data.get("metadata", {})

        return feedback


class FeedbackCollector:
    """Main system for collecting and managing user feedback."""

    def __init__(self, storage_path: str = "./feedback"):
        """
        Initialize the feedback collector.

        Args:
            storage_path: Path to store feedback data
        """
        self.storage_path = storage_path
        self.feedback_data: Dict[str, UserFeedback] = {}  # workflow_id -> feedback
        self.user_feedback_history: Dict[
            str, List[UserFeedback]
        ] = {}  # user_id -> feedback list
        self.pending_feedback_requests: Dict[
            str, Dict[str, Any]
        ] = {}  # workflow_id -> request info

        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)

        # Load existing feedback
        self._load_feedback_data()

        logger.info("Feedback collector initialized")

    def submit_feedback(self, feedback: UserFeedback) -> bool:
        """
        Submit user feedback for a workflow.

        Args:
            feedback: UserFeedback object

        Returns:
            True if submission successful
        """
        try:
            self.feedback_data[feedback.workflow_id] = feedback

            # Update user history
            if feedback.user_id not in self.user_feedback_history:
                self.user_feedback_history[feedback.user_id] = []
            self.user_feedback_history[feedback.user_id].append(feedback)

            # Remove from pending requests
            if feedback.workflow_id in self.pending_feedback_requests:
                del self.pending_feedback_requests[feedback.workflow_id]

            # Save to disk
            self._save_feedback(feedback)

            logger.info(
                f"Feedback submitted for workflow {feedback.workflow_id} by user {feedback.user_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return False

    def request_feedback(
        self, workflow_id: str, user_id: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Request feedback from a user for a specific workflow.

        Args:
            workflow_id: ID of the workflow
            user_id: ID of the user to request feedback from
            context: Additional context for the feedback request

        Returns:
            Request ID for tracking
        """
        request_id = (
            f"req_{workflow_id}_{user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

        self.pending_feedback_requests[workflow_id] = {
            "request_id": request_id,
            "user_id": user_id,
            "workflow_id": workflow_id,
            "context": context or {},
            "timestamp": datetime.utcnow(),
            "status": "pending",
        }

        logger.info(
            f"Feedback requested for workflow {workflow_id} from user {user_id}"
        )
        return request_id

    def get_feedback(self, workflow_id: str) -> Optional[UserFeedback]:
        """Get feedback for a specific workflow."""
        return self.feedback_data.get(workflow_id)

    def get_user_feedback_history(self, user_id: str) -> List[UserFeedback]:
        """Get all feedback from a specific user."""
        return self.user_feedback_history.get(user_id, [])

    def get_pending_requests(
        self, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get pending feedback requests."""
        requests = list(self.pending_feedback_requests.values())

        if user_id:
            requests = [r for r in requests if r["user_id"] == user_id]

        return requests

    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get overall feedback statistics."""
        total_feedback = len(self.feedback_data)
        total_users = len(self.user_feedback_history)
        pending_requests = len(self.pending_feedback_requests)

        if total_feedback == 0:
            return {
                "total_feedback": 0,
                "total_users": 0,
                "pending_requests": pending_requests,
                "avg_cat_score": None,
                "avg_composite_score": None,
            }

        # Calculate averages
        cat_scores = []
        composite_scores = []

        for feedback in self.feedback_data.values():
            if feedback.cat_score:
                cat_scores.append(feedback.cat_score.score)
            composite_scores.append(feedback.get_composite_score())

        avg_cat_score = sum(cat_scores) / len(cat_scores) if cat_scores else None
        avg_composite_score = sum(composite_scores) / len(composite_scores)

        return {
            "total_feedback": total_feedback,
            "total_users": total_users,
            "pending_requests": pending_requests,
            "avg_cat_score": avg_cat_score,
            "avg_composite_score": avg_composite_score,
            "feedback_with_cat_scores": len(cat_scores),
        }

    def get_high_scoring_workflows(self, threshold: float = 0.8) -> List[str]:
        """Get workflows with high composite scores."""
        high_scoring = []
        for workflow_id, feedback in self.feedback_data.items():
            if feedback.get_composite_score() >= threshold:
                high_scoring.append(workflow_id)
        return high_scoring

    def get_feedback_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get feedback trends over the specified number of days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        recent_feedback = [
            f for f in self.feedback_data.values() if f.timestamp >= cutoff_date
        ]

        if not recent_feedback:
            return {"error": "No recent feedback data"}

        # Calculate trends
        cat_scores = [f.cat_score.score for f in recent_feedback if f.cat_score]
        composite_scores = [f.get_composite_score() for f in recent_feedback]

        return {
            "period_days": days,
            "total_feedback": len(recent_feedback),
            "avg_cat_score": sum(cat_scores) / len(cat_scores) if cat_scores else None,
            "avg_composite_score": sum(composite_scores) / len(composite_scores),
            "feedback_with_cat_scores": len(cat_scores),
            "score_distribution": self._calculate_score_distribution(composite_scores),
        }

    def _calculate_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Calculate score distribution."""
        distribution = {
            "0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0,
        }

        for score in scores:
            if score < 0.2:
                distribution["0-0.2"] += 1
            elif score < 0.4:
                distribution["0.2-0.4"] += 1
            elif score < 0.6:
                distribution["0.4-0.6"] += 1
            elif score < 0.8:
                distribution["0.6-0.8"] += 1
            else:
                distribution["0.8-1.0"] += 1

        return distribution

    def export_feedback_data(self, filename: str, format: str = "json"):
        """Export feedback data for analysis."""
        if format == "json":
            data = {
                "feedback": [f.to_dict() for f in self.feedback_data.values()],
                "statistics": self.get_feedback_statistics(),
                "export_timestamp": datetime.utcnow().isoformat(),
            }

            filepath = os.path.join(self.storage_path, f"{filename}.json")
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

        elif format == "csv":
            import csv

            filepath = os.path.join(self.storage_path, f"{filename}.csv")

            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "workflow_id",
                        "user_id",
                        "cat_score",
                        "composite_score",
                        "satisfaction",
                        "usefulness",
                        "novelty",
                        "quality",
                        "timestamp",
                    ]
                )

                for feedback in self.feedback_data.values():
                    writer.writerow(
                        [
                            feedback.workflow_id,
                            feedback.user_id,
                            feedback.cat_score.score if feedback.cat_score else None,
                            feedback.get_composite_score(),
                            feedback.satisfaction_rating,
                            feedback.usefulness_rating,
                            feedback.novelty_rating,
                            feedback.quality_rating,
                            feedback.timestamp.isoformat(),
                        ]
                    )

        logger.info(f"Feedback data exported to {filepath}")

    def _save_feedback(self, feedback: UserFeedback):
        """Save feedback to disk."""
        filename = f"feedback_{feedback.workflow_id}.json"
        filepath = os.path.join(self.storage_path, filename)

        with open(filepath, "w") as f:
            json.dump(feedback.to_dict(), f, indent=2, default=str)

    def _load_feedback_data(self):
        """Load existing feedback data from disk."""
        if not os.path.exists(self.storage_path):
            return

        import glob

        pattern = os.path.join(self.storage_path, "feedback_*.json")
        for filepath in glob.glob(pattern):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    feedback = UserFeedback.from_dict(data)
                    self.feedback_data[feedback.workflow_id] = feedback

                    # Update user history
                    if feedback.user_id not in self.user_feedback_history:
                        self.user_feedback_history[feedback.user_id] = []
                    self.user_feedback_history[feedback.user_id].append(feedback)

            except Exception as e:
                logger.error(f"Error loading feedback from {filepath}: {e}")

        logger.info(f"Loaded {len(self.feedback_data)} feedback records")


class FeedbackAPI:
    """API interface for feedback collection from external systems."""

    def __init__(self, feedback_collector: FeedbackCollector):
        self.feedback_collector = feedback_collector

    async def submit_cat_score(
        self,
        workflow_id: str,
        user_id: str,
        score: int,
        dimensions: Optional[Dict[str, int]] = None,
        comments: Optional[str] = None,
    ) -> Dict[str, Any]:
        """API endpoint for submitting CAT scores."""
        try:
            feedback = UserFeedback(workflow_id, user_id)
            feedback.add_cat_score(score, dimensions, comments)

            success = self.feedback_collector.submit_feedback(feedback)

            return {
                "success": success,
                "message": "CAT score submitted successfully"
                if success
                else "Submission failed",
                "normalized_score": feedback.cat_score.get_normalized_score()
                if success
                else None,
            }
        except Exception as e:
            logger.error(f"Error submitting CAT score: {e}")
            return {"success": False, "error": str(e)}

    async def submit_detailed_feedback(
        self,
        workflow_id: str,
        user_id: str,
        ratings: Dict[str, float],
        feedback_text: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """API endpoint for submitting detailed feedback."""
        try:
            feedback = UserFeedback(workflow_id, user_id)
            feedback.set_ratings(**ratings)

            if feedback_text:
                feedback.add_feedback_text(feedback_text)

            if suggestions:
                for suggestion in suggestions:
                    feedback.add_suggestion(suggestion)

            if tags:
                for tag in tags:
                    feedback.add_tag(tag)

            success = self.feedback_collector.submit_feedback(feedback)

            return {
                "success": success,
                "message": "Detailed feedback submitted successfully"
                if success
                else "Submission failed",
                "composite_score": feedback.get_composite_score() if success else None,
            }
        except Exception as e:
            logger.error(f"Error submitting detailed feedback: {e}")
            return {"success": False, "error": str(e)}

    async def get_feedback_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get feedback status for a workflow."""
        feedback = self.feedback_collector.get_feedback(workflow_id)

        if feedback:
            return {
                "has_feedback": True,
                "composite_score": feedback.get_composite_score(),
                "cat_score": feedback.cat_score.score if feedback.cat_score else None,
                "timestamp": feedback.timestamp.isoformat(),
            }
        else:
            return {"has_feedback": False}
