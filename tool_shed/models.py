"""
Tool Shed Data Models

Pydantic models for tools, proposals, versions, and registry entries.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class ToolMetadata(BaseModel):
    """Metadata for a tool"""

    name: str
    description: str
    author: str
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    tags: List[str] = Field(default_factory=list)
    category: str
    license: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)


class ToolCapabilities(BaseModel):
    """Capabilities and features of a tool"""

    functions: List[str] = Field(default_factory=list)
    input_types: List[str] = Field(default_factory=list)
    output_types: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    security_level: str = "medium"  # "low", "medium", "high"
    performance_requirements: Dict[str, Any] = Field(default_factory=dict)


class ToolValidation(BaseModel):
    """Validation results for a tool"""

    security_scan_passed: bool = False
    linting_passed: bool = False
    unit_tests_passed: bool = False
    integration_tests_passed: bool = False
    security_issues: List[str] = Field(default_factory=list)
    linting_issues: List[str] = Field(default_factory=list)
    test_coverage: float = 0.0
    validated_at: Optional[datetime] = None
    validated_by: Optional[str] = None


class Tool(BaseModel):
    """Complete tool definition"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: ToolMetadata
    capabilities: ToolCapabilities
    code: str  # The actual tool implementation code
    documentation: str
    examples: List[str] = Field(default_factory=list)
    validation: ToolValidation = Field(default_factory=ToolValidation)
    embedding: Optional[List[float]] = None  # For semantic search
    is_active: bool = True
    usage_count: int = 0
    rating: float = 0.0
    reviews: List[Dict[str, Any]] = Field(default_factory=list)


class ToolProposal(BaseModel):
    """Proposal for a new tool"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    proposer_agent: str
    tool_name: str
    description: str
    capabilities: List[str]
    use_case: str
    priority: str = "medium"  # "low", "medium", "high"
    proposed_code: Optional[str] = None
    requirements: List[str] = Field(default_factory=list)
    status: str = "pending"  # "pending", "approved", "rejected", "in_review"
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = None
    reviewer_agent: Optional[str] = None
    review_comments: Optional[str] = None


class ToolVersion(BaseModel):
    """Version information for tool evolution"""

    tool_id: str
    version: str
    changes: str
    author: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_breaking_change: bool = False
    backward_compatible: bool = True
    deprecation_warnings: List[str] = Field(default_factory=list)


class ToolRegistryEntry(BaseModel):
    """Entry in the tool registry"""

    tool: Tool
    versions: List[ToolVersion] = Field(default_factory=list)
    proposals: List[ToolProposal] = Field(default_factory=list)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 0


class SearchQuery(BaseModel):
    """Query for tool search"""

    query: str
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    author: Optional[str] = None
    min_rating: Optional[float] = None
    security_level: Optional[str] = None
    limit: int = 10
    offset: int = 0


class SearchResult(BaseModel):
    """Result from tool search"""

    tools: List[Tool]
    total_count: int
    query_time: float
    semantic_matches: List[Dict[str, Any]] = Field(default_factory=list)


class ToolEvolutionRequest(BaseModel):
    """Request for tool evolution"""

    tool_id: str
    evolution_type: str  # "enhancement", "bug_fix", "optimization", "deprecation"
    description: str
    proposed_changes: Dict[str, Any]
    requester_agent: str
    priority: str = "medium"
    status: str = "pending"
