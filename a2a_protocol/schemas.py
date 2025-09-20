"""
A2A Protocol Message Schemas

This module defines the JSON-RPC 2.0 message schemas for the A2A Protocol v2.1,
including enhanced message types for geospatial anomalies, inspiration requests,
creation feedback, tool proposals, narrative prompts, and certification requests.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import uuid


class A2AMessage(BaseModel):
    """Base class for all A2A messages"""

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sender_agent: str
    target_agent: Optional[str] = None


class GeospatialAnomalyIdentified(A2AMessage):
    """Message for identifying geospatial anomalies"""

    anomaly_type: str
    location: Dict[str, float]  # {"lat": float, "lon": float}
    confidence: float = Field(ge=0.0, le=1.0)
    description: str
    data_source: str
    metadata: Optional[Dict[str, Any]] = None


class InspirationRequest(A2AMessage):
    """Request for creative inspiration"""

    context: str
    domain: str  # e.g., "mythology", "geography", "cultural"
    constraints: Optional[List[str]] = None
    inspiration_type: str  # e.g., "narrative", "visual", "conceptual"


class CreationFeedback(A2AMessage):
    """Feedback on creative outputs"""

    original_request_id: str
    feedback_type: str  # "positive", "negative", "suggestion"
    content: str
    rating: Optional[int] = Field(None, ge=1, le=5)
    suggestions: Optional[List[str]] = None


class ToolProposal(A2AMessage):
    """Proposal for new tools or capabilities"""

    tool_name: str
    description: str
    capabilities: List[str]
    requirements: Optional[List[str]] = None
    use_case: str
    priority: str = "medium"  # "low", "medium", "high"


class NarrativePrompt(A2AMessage):
    """Prompt for narrative generation"""

    theme: str
    elements: List[str]
    style: str
    length: str = "medium"  # "short", "medium", "long"
    constraints: Optional[List[str]] = None


class CertificationRequest(A2AMessage):
    """Request for certification/validation"""

    subject: str
    certification_type: str
    evidence: Dict[str, Any]
    criteria: List[str]
    validity_period: Optional[int] = None  # days


# JSON-RPC 2.0 structures
class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 Request"""

    jsonrpc: str = "2.0"
    method: str
    params: Union[A2AMessage, Dict[str, Any]]
    id: Union[str, int]


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 Response"""

    jsonrpc: str = "2.0"
    result: Any
    id: Union[str, int]


class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 Error"""

    code: int
    message: str
    data: Optional[Any] = None


class JSONRPCErrorResponse(BaseModel):
    """JSON-RPC 2.0 Error Response"""

    jsonrpc: str = "2.0"
    error: JSONRPCError
    id: Union[str, int, None]


class JSONRPCNotification(BaseModel):
    """JSON-RPC 2.0 Notification (no response expected)"""

    jsonrpc: str = "2.0"
    method: str
    params: Union[A2AMessage, Dict[str, Any]]


# Message type registry for extensibility
MESSAGE_TYPES = {
    "GEOSPATIAL_ANOMALY_IDENTIFIED": GeospatialAnomalyIdentified,
    "INSPIRATION_REQUEST": InspirationRequest,
    "CREATION_FEEDBACK": CreationFeedback,
    "TOOL_PROPOSAL": ToolProposal,
    "NARRATIVE_PROMPT": NarrativePrompt,
    "CERTIFICATION_REQUEST": CertificationRequest,
}


def get_message_class(message_type: str) -> type:
    """Get the message class for a given type"""
    return MESSAGE_TYPES.get(message_type)


def create_message(message_type: str, **kwargs) -> A2AMessage:
    """Create a message instance of the specified type"""
    cls = get_message_class(message_type)
    if not cls:
        raise ValueError(f"Unknown message type: {message_type}")
    return cls(**kwargs)
