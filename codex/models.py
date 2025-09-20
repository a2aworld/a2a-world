"""
Data models for the Agent's Codex system.

This module defines the core data structures for archiving agent contributions,
workflow histories, strategies, and legacy chapters.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum


class ContributionType(Enum):
    """Types of agent contributions."""

    TASK_EXECUTION = "task_execution"
    CREATIVE_OUTPUT = "creative_output"
    PROBLEM_SOLVING = "problem_solving"
    COLLABORATION = "collaboration"
    LEARNING_INSIGHT = "learning_insight"
    STRATEGY_DEVELOPMENT = "strategy_development"


class StrategyType(Enum):
    """Types of documented strategies."""

    WORKFLOW_PATTERN = "workflow_pattern"
    DECISION_MAKING = "decision_making"
    PROBLEM_SOLVING = "problem_solving"
    OPTIMIZATION = "optimization"
    COLLABORATION = "collaboration"


@dataclass
class AttributionRecord:
    """Record of attribution for AI partners and contributors."""

    agent_name: str
    agent_type: str
    contribution_type: ContributionType
    timestamp: datetime
    ai_model: Optional[str] = None
    ai_provider: Optional[str] = None
    human_contributor: Optional[str] = None
    contribution_weight: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data["contribution_type"] = self.contribution_type.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttributionRecord":
        """Create from dictionary."""
        data["contribution_type"] = ContributionType(data["contribution_type"])
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class AgentContribution:
    """Record of an agent's contribution."""

    contribution_id: str
    agent_name: str
    agent_type: str
    task_description: str
    contribution_type: ContributionType
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    success_metrics: Dict[str, Any]
    timestamp: datetime
    duration: Optional[float] = None
    workflow_context: Optional[str] = None
    collaboration_partners: List[str] = None
    attribution_records: List[AttributionRecord] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.collaboration_partners is None:
            self.collaboration_partners = []
        if self.attribution_records is None:
            self.attribution_records = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data["contribution_type"] = self.contribution_type.value
        data["timestamp"] = self.timestamp.isoformat()
        data["attribution_records"] = [
            record.to_dict() for record in self.attribution_records
        ]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentContribution":
        """Create from dictionary."""
        data["contribution_type"] = ContributionType(data["contribution_type"])
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["attribution_records"] = [
            AttributionRecord.from_dict(r) for r in data["attribution_records"]
        ]
        return cls(**data)


@dataclass
class StrategyDocument:
    """Documented strategy or pattern."""

    strategy_id: str
    strategy_type: StrategyType
    title: str
    description: str
    context: str
    steps: List[Dict[str, Any]]
    success_criteria: List[str]
    lessons_learned: List[str]
    created_by: str
    created_at: datetime
    validated_count: int = 0
    success_rate: float = 0.0
    related_contributions: List[str] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.related_contributions is None:
            self.related_contributions = []
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data["strategy_type"] = self.strategy_type.value
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyDocument":
        """Create from dictionary."""
        data["strategy_type"] = StrategyType(data["strategy_type"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class LegacyChapter:
    """Generated legacy chapter for the Galactic Storybook."""

    chapter_id: str
    title: str
    narrative: str
    theme: str
    key_events: List[Dict[str, Any]]
    agent_heroes: List[str]
    lessons_embodied: List[str]
    generated_at: datetime
    source_contributions: List[str]
    source_strategies: List[str]
    attribution_summary: Dict[str, Any]
    cat_score: Optional[float] = None
    published: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.source_contributions is None:
            self.source_contributions = []
        if self.source_strategies is None:
            self.source_strategies = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data["generated_at"] = self.generated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LegacyChapter":
        """Create from dictionary."""
        data["generated_at"] = datetime.fromisoformat(data["generated_at"])
        return cls(**data)


@dataclass
class KnowledgeEntry:
    """Entry in the knowledge base."""

    entry_id: str
    category: str
    title: str
    content: str
    source_type: str
    source_id: str
    confidence_score: float
    created_at: datetime
    last_updated: datetime
    access_count: int = 0
    usefulness_score: float = 0.0
    tags: List[str] = None
    related_entries: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.related_entries is None:
            self.related_entries = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["last_updated"] = self.last_updated.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEntry":
        """Create from dictionary."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["last_updated"] = datetime.fromisoformat(data["last_updated"])
        return cls(**data)


@dataclass
class CodexStatistics:
    """Statistics for the Codex system."""

    total_contributions: int
    total_strategies: int
    total_chapters: int
    total_knowledge_entries: int
    active_agents: int
    avg_contribution_quality: float
    top_contributing_agents: List[Dict[str, Any]]
    most_successful_strategies: List[Dict[str, Any]]
    knowledge_coverage: Dict[str, int]
    generated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["generated_at"] = self.generated_at.isoformat()
        return data
