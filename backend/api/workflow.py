"""
Co-Creation Workflow API Router

This module provides REST API endpoints for the unified co-creation workflow,
enabling human-AI partnership through the complete doubt->discovery->art->wisdom->knowledge cycle.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

# Import will be done at runtime to avoid circular imports
cocreation_workflow = None


def set_cocreation_workflow(workflow_instance):
    """Set the co-creation workflow instance."""
    global cocreation_workflow
    cocreation_workflow = workflow_instance


router = APIRouter()


# Pydantic models
class WorkflowStartRequest(BaseModel):
    """Request model for starting a workflow."""

    trigger_source: str = Field(
        ..., description="Source of trigger ('human' or 'autonomous')"
    )
    human_input: Optional[str] = Field(None, description="Human input text")
    workflow_id: Optional[str] = Field(None, description="Optional custom workflow ID")


class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status."""

    workflow_id: str
    current_stage: str
    stages_completed: list
    success: bool
    completion_time: Optional[datetime]
    message: str


class FeedbackRequest(BaseModel):
    """Request model for human feedback."""

    workflow_id: str = Field(..., description="Workflow ID")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    feedback_text: Optional[str] = Field(None, description="Optional feedback text")
    curation_decisions: Optional[Dict[str, Any]] = Field(
        None, description="Curation decisions"
    )


class WorkflowHistoryResponse(BaseModel):
    """Response model for workflow history."""

    workflows: list
    total_count: int
    message: str


@router.post("/start", response_model=Dict[str, Any])
async def start_cocreation_workflow(
    request: WorkflowStartRequest, background_tasks: BackgroundTasks
):
    """
    Start a new co-creation workflow.

    This endpoint initiates the complete human-AI co-creation cycle:
    doubt -> discovery -> art -> wisdom -> knowledge
    """
    if not cocreation_workflow:
        raise HTTPException(
            status_code=503, detail="Co-creation workflow not available"
        )

    try:
        # Start workflow in background to avoid blocking
        background_tasks.add_task(
            cocreation_workflow.start_cocreation_workflow,
            request.trigger_source,
            request.human_input,
            request.workflow_id,
        )

        workflow_id = (
            request.workflow_id
            or f"cocreation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

        return {
            "message": "Co-creation workflow started successfully",
            "workflow_id": workflow_id,
            "trigger_source": request.trigger_source,
            "human_input": request.human_input,
            "status": "processing",
            "timestamp": datetime.utcnow(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start workflow: {str(e)}"
        )


@router.get("/status/{workflow_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str):
    """Get the current status of a workflow."""
    if not cocreation_workflow:
        raise HTTPException(
            status_code=503, detail="Co-creation workflow not available"
        )

    try:
        status = cocreation_workflow.get_workflow_status(workflow_id)

        if not status:
            raise HTTPException(
                status_code=404, detail=f"Workflow {workflow_id} not found"
            )

        return WorkflowStatusResponse(
            workflow_id=status["workflow_id"],
            current_stage=status["current_stage"],
            stages_completed=status["stages_completed"],
            success=status["success"],
            completion_time=status["completion_time"],
            message=f"Workflow is in {status['current_stage']} stage",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get workflow status: {str(e)}"
        )


@router.post("/feedback")
async def submit_human_feedback(request: FeedbackRequest):
    """
    Submit human feedback for a workflow.

    This enables the human-AI feedback loop, allowing humans to curate
    and influence the creative process at any stage.
    """
    if not cocreation_workflow:
        raise HTTPException(
            status_code=503, detail="Co-creation workflow not available"
        )

    try:
        feedback_data = {
            "rating": request.rating,
            "feedback_text": request.feedback_text,
            "curation_decisions": request.curation_decisions,
            "timestamp": datetime.utcnow(),
        }

        success = await cocreation_workflow.submit_human_feedback(
            request.workflow_id, feedback_data
        )

        if not success:
            raise HTTPException(
                status_code=404, detail=f"Workflow {request.workflow_id} not found"
            )

        return {
            "message": "Human feedback submitted successfully",
            "workflow_id": request.workflow_id,
            "feedback_processed": True,
            "timestamp": datetime.utcnow(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to submit feedback: {str(e)}"
        )


@router.get("/history", response_model=WorkflowHistoryResponse)
async def get_workflow_history():
    """Get the history of all co-creation workflows."""
    if not cocreation_workflow:
        raise HTTPException(
            status_code=503, detail="Co-creation workflow not available"
        )

    try:
        history = cocreation_workflow.get_workflow_history()

        return WorkflowHistoryResponse(
            workflows=history,
            total_count=len(history),
            message=f"Retrieved {len(history)} workflows from history",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get workflow history: {str(e)}"
        )


@router.post("/trigger-autonomous")
async def trigger_autonomous_workflow():
    """
    Trigger an autonomous workflow.

    This endpoint allows the system to initiate co-creation cycles
    without human input, based on system triggers or scheduled events.
    """
    if not cocreation_workflow:
        raise HTTPException(
            status_code=503, detail="Co-creation workflow not available"
        )

    try:
        # Generate autonomous trigger
        autonomous_input = (
            "Autonomous exploration of creative territories in geomythology"
        )

        # Start autonomous workflow
        workflow_result = await cocreation_workflow.start_cocreation_workflow(
            trigger_source="autonomous", human_input=autonomous_input
        )

        return {
            "message": "Autonomous co-creation workflow triggered",
            "workflow_id": workflow_result["workflow_id"],
            "trigger_type": "autonomous",
            "input": autonomous_input,
            "timestamp": datetime.utcnow(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to trigger autonomous workflow: {str(e)}"
        )


@router.get("/philosophy")
async def get_posthuman_creativity_philosophy():
    """
    Get information about the Posthuman Creativity philosophy.

    This endpoint provides context about the philosophical framework
    guiding the human-AI co-creation process.
    """
    philosophy_info = {
        "name": "Posthuman Creativity",
        "description": "A philosophical framework for human-AI creative partnership",
        "principles": [
            "Humans and AI as co-creators in a symbiotic relationship",
            "Technology as an extension of human creative potential",
            "Ethical integration of artificial and human intelligence",
            "Preservation of creative legacy across human-AI boundaries",
            "Continuous learning and adaptation in creative processes",
        ],
        "cycle": [
            "Doubt: Questioning and curiosity-driven exploration",
            "Discovery: Autonomous and collaborative knowledge generation",
            "Art: Creative expression through human-AI synthesis",
            "Wisdom: Synthesis of insights and pattern recognition",
            "Knowledge: Preservation and transmission of creative legacy",
        ],
        "goals": [
            "Enhance human creative potential through AI partnership",
            "Create novel forms of artistic expression",
            "Preserve cultural and creative knowledge",
            "Foster ethical human-AI collaboration",
            "Advance understanding of consciousness and creativity",
        ],
    }

    return philosophy_info


@router.get("/stats")
async def get_workflow_statistics():
    """Get statistics about co-creation workflows."""
    if not cocreation_workflow:
        raise HTTPException(
            status_code=503, detail="Co-creation workflow not available"
        )

    try:
        history = cocreation_workflow.get_workflow_history()

        # Calculate statistics
        total_workflows = len(history)
        completed_workflows = len([w for w in history if w["success"]])
        success_rate = (
            (completed_workflows / total_workflows * 100) if total_workflows > 0 else 0
        )

        # Stage completion stats
        stage_counts = {}
        for workflow in history:
            for stage in workflow["stages_completed"]:
                stage_counts[stage] = stage_counts.get(stage, 0) + 1

        return {
            "total_workflows": total_workflows,
            "completed_workflows": completed_workflows,
            "success_rate": round(success_rate, 2),
            "stage_completion_stats": stage_counts,
            "average_stages_completed": sum(len(w["stages_completed"]) for w in history)
            / total_workflows
            if total_workflows > 0
            else 0,
            "timestamp": datetime.utcnow(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get workflow statistics: {str(e)}"
        )
