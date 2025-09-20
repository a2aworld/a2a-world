"""
Tool Shed - Dynamic Tool Registry System

A comprehensive system for agent self-improvement through tool discovery,
validation, evolution, and autonomous management.

Components:
- ToolRegistry: Dynamic registry for tool management
- ToolSmithAgent: Gatekeeper for tool validation and approval
- ToolVectorStore: Vector database for semantic search
- SemanticSearchEngine: Advanced search capabilities
- ToolEvolutionManager: Versioning and evolution support
"""

from .models import (
    Tool,
    ToolProposal,
    ToolVersion,
    ToolEvolutionRequest,
    ToolMetadata,
    ToolCapabilities,
    ToolValidation,
    ToolRegistryEntry,
    SearchQuery,
    SearchResult,
)
from .registry import ToolRegistry
from .vector_store import ToolVectorStore
from .search import SemanticSearchEngine
from .evolution import ToolEvolutionManager
from .tool_smith_agent import ToolSmithAgent

__version__ = "1.0.0"
__author__ = "Terra Constellata"

__all__ = [
    # Models
    "Tool",
    "ToolProposal",
    "ToolVersion",
    "ToolEvolutionRequest",
    "ToolMetadata",
    "ToolCapabilities",
    "ToolValidation",
    "ToolRegistryEntry",
    "SearchQuery",
    "SearchResult",
    # Core Components
    "ToolRegistry",
    "ToolVectorStore",
    "SemanticSearchEngine",
    "ToolEvolutionManager",
    "ToolSmithAgent",
]
