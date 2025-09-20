"""
Tool Registry System

Dynamic registry for managing tool registration, validation, and discovery.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from .models import (
    Tool,
    ToolProposal,
    ToolRegistryEntry,
    ToolVersion,
    ToolEvolutionRequest,
    SearchQuery,
    SearchResult,
)
from .vector_store import ToolVectorStore

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Dynamic tool registry system.

    Manages tool registration, versioning, proposals, and provides
    integration with vector store for discovery.
    """

    def __init__(self, vector_store: Optional[ToolVectorStore] = None):
        """
        Initialize the tool registry.

        Args:
            vector_store: Optional vector store for semantic search
        """
        self.registry: Dict[str, ToolRegistryEntry] = {}
        self.vector_store = vector_store or ToolVectorStore()
        self.proposals: Dict[str, ToolProposal] = {}
        self.evolution_requests: Dict[str, ToolEvolutionRequest] = {}

        logger.info("Initialized ToolRegistry")

    async def register_tool(self, tool: Tool, proposer_agent: str = "system") -> bool:
        """
        Register a new tool in the registry.

        Args:
            tool: Tool to register
            proposer_agent: Agent proposing the tool

        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Check if tool already exists
            if tool.id in self.registry:
                logger.warning(f"Tool {tool.metadata.name} already exists")
                return False

            # Create registry entry
            entry = ToolRegistryEntry(
                tool=tool,
                versions=[
                    ToolVersion(
                        tool_id=tool.id,
                        version=tool.metadata.version,
                        changes="Initial registration",
                        author=proposer_agent,
                        created_at=datetime.utcnow(),
                    )
                ],
            )

            # Add to registry
            self.registry[tool.id] = entry

            # Add to vector store
            success = self.vector_store.add_tool(tool)
            if not success:
                logger.error(f"Failed to add tool {tool.metadata.name} to vector store")
                # Still keep in registry but mark as inactive
                tool.is_active = False

            logger.info(f"Registered tool: {tool.metadata.name} by {proposer_agent}")
            return True

        except Exception as e:
            logger.error(f"Failed to register tool {tool.metadata.name}: {e}")
            return False

    async def update_tool(self, tool_id: str, updated_tool: Tool, author: str) -> bool:
        """
        Update an existing tool.

        Args:
            tool_id: ID of tool to update
            updated_tool: Updated tool definition
            author: Agent making the update

        Returns:
            True if update successful, False otherwise
        """
        try:
            if tool_id not in self.registry:
                logger.error(f"Tool {tool_id} not found in registry")
                return False

            entry = self.registry[tool_id]
            current_version = entry.tool.metadata.version

            # Create new version entry
            version = ToolVersion(
                tool_id=tool_id,
                version=updated_tool.metadata.version,
                changes=f"Updated from {current_version}",
                author=author,
                created_at=datetime.utcnow(),
            )

            # Update tool
            entry.tool = updated_tool
            entry.versions.append(version)
            entry.last_accessed = datetime.utcnow()

            # Update vector store
            success = self.vector_store.update_tool(updated_tool)
            if not success:
                logger.error(f"Failed to update tool {tool_id} in vector store")

            logger.info(
                f"Updated tool {tool_id} to version {updated_tool.metadata.version}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to update tool {tool_id}: {e}")
            return False

    async def remove_tool(self, tool_id: str) -> bool:
        """
        Remove a tool from the registry.

        Args:
            tool_id: ID of tool to remove

        Returns:
            True if removal successful, False otherwise
        """
        try:
            if tool_id not in self.registry:
                logger.error(f"Tool {tool_id} not found in registry")
                return False

            # Remove from registry
            del self.registry[tool_id]

            # Remove from vector store
            success = self.vector_store.remove_tool(tool_id)
            if not success:
                logger.warning(f"Failed to remove tool {tool_id} from vector store")

            logger.info(f"Removed tool {tool_id} from registry")
            return True

        except Exception as e:
            logger.error(f"Failed to remove tool {tool_id}: {e}")
            return False

    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """
        Get a tool by ID.

        Args:
            tool_id: ID of tool to retrieve

        Returns:
            Tool if found, None otherwise
        """
        entry = self.registry.get(tool_id)
        if entry:
            entry.last_accessed = datetime.utcnow()
            entry.access_count += 1
            return entry.tool
        return None

    def list_tools(
        self,
        category: Optional[str] = None,
        author: Optional[str] = None,
        active_only: bool = True,
    ) -> List[Tool]:
        """
        List tools with optional filtering.

        Args:
            category: Filter by category
            author: Filter by author
            active_only: Only return active tools

        Returns:
            List of matching tools
        """
        tools = []

        for entry in self.registry.values():
            tool = entry.tool

            if active_only and not tool.is_active:
                continue

            if category and tool.metadata.category != category:
                continue

            if author and tool.metadata.author != author:
                continue

            tools.append(tool)

        return tools

    async def submit_proposal(self, proposal: ToolProposal) -> str:
        """
        Submit a tool proposal.

        Args:
            proposal: Tool proposal to submit

        Returns:
            Proposal ID
        """
        self.proposals[proposal.id] = proposal
        logger.info(f"Submitted proposal for tool: {proposal.tool_name}")
        return proposal.id

    async def approve_proposal(self, proposal_id: str, reviewer_agent: str) -> bool:
        """
        Approve a tool proposal.

        Args:
            proposal_id: ID of proposal to approve
            reviewer_agent: Agent approving the proposal

        Returns:
            True if approval successful, False otherwise
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            logger.error(f"Proposal {proposal_id} not found")
            return False

        proposal.status = "approved"
        proposal.reviewed_at = datetime.utcnow()
        proposal.reviewer_agent = reviewer_agent

        logger.info(f"Approved proposal {proposal_id} for {proposal.tool_name}")
        return True

    async def reject_proposal(
        self, proposal_id: str, reviewer_agent: str, comments: str
    ) -> bool:
        """
        Reject a tool proposal.

        Args:
            proposal_id: ID of proposal to reject
            reviewer_agent: Agent rejecting the proposal
            comments: Rejection comments

        Returns:
            True if rejection successful, False otherwise
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            logger.error(f"Proposal {proposal_id} not found")
            return False

        proposal.status = "rejected"
        proposal.reviewed_at = datetime.utcnow()
        proposal.reviewer_agent = reviewer_agent
        proposal.review_comments = comments

        logger.info(f"Rejected proposal {proposal_id} for {proposal.tool_name}")
        return True

    def get_pending_proposals(self) -> List[ToolProposal]:
        """Get all pending tool proposals."""
        return [p for p in self.proposals.values() if p.status == "pending"]

    async def submit_evolution_request(self, request: ToolEvolutionRequest) -> str:
        """
        Submit a tool evolution request.

        Args:
            request: Evolution request to submit

        Returns:
            Request ID
        """
        self.evolution_requests[request.id] = request
        logger.info(f"Submitted evolution request for tool {request.tool_id}")
        return request.id

    def get_evolution_requests(
        self, tool_id: Optional[str] = None
    ) -> List[ToolEvolutionRequest]:
        """
        Get evolution requests, optionally filtered by tool ID.

        Args:
            tool_id: Optional tool ID to filter by

        Returns:
            List of matching evolution requests
        """
        requests = list(self.evolution_requests.values())

        if tool_id:
            requests = [r for r in requests if r.tool_id == tool_id]

        return requests

    async def search_tools(self, query: SearchQuery) -> SearchResult:
        """
        Search for tools using the vector store.

        Args:
            query: Search query parameters

        Returns:
            Search results
        """
        return self.vector_store.search_tools(query)

    def get_tool_versions(self, tool_id: str) -> List[ToolVersion]:
        """
        Get version history for a tool.

        Args:
            tool_id: ID of the tool

        Returns:
            List of tool versions
        """
        entry = self.registry.get(tool_id)
        if entry:
            return entry.versions
        return []

    def get_similar_tools(self, tool_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find tools similar to the given tool.

        Args:
            tool_id: ID of the reference tool
            limit: Maximum number of similar tools

        Returns:
            List of similar tools with metadata
        """
        return self.vector_store.get_similar_tools(tool_id, limit)

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the registry."""
        total_tools = len(self.registry)
        active_tools = sum(
            1 for entry in self.registry.values() if entry.tool.is_active
        )
        total_proposals = len(self.proposals)
        pending_proposals = sum(
            1 for p in self.proposals.values() if p.status == "pending"
        )
        total_evolution_requests = len(self.evolution_requests)

        categories = {}
        authors = {}

        for entry in self.registry.values():
            tool = entry.tool
            categories[tool.metadata.category] = (
                categories.get(tool.metadata.category, 0) + 1
            )
            authors[tool.metadata.author] = authors.get(tool.metadata.author, 0) + 1

        return {
            "total_tools": total_tools,
            "active_tools": active_tools,
            "inactive_tools": total_tools - active_tools,
            "total_proposals": total_proposals,
            "pending_proposals": pending_proposals,
            "total_evolution_requests": total_evolution_requests,
            "categories": categories,
            "authors": authors,
            "vector_store_stats": self.vector_store.get_collection_stats(),
        }

    async def export_registry(self, filepath: str) -> bool:
        """
        Export the registry to a JSON file.

        Args:
            filepath: Path to export file

        Returns:
            True if export successful, False otherwise
        """
        try:
            data = {
                "registry": {
                    tool_id: entry.dict() for tool_id, entry in self.registry.items()
                },
                "proposals": {
                    prop_id: prop.dict() for prop_id, prop in self.proposals.items()
                },
                "evolution_requests": {
                    req_id: req.dict()
                    for req_id, req in self.evolution_requests.items()
                },
                "exported_at": datetime.utcnow().isoformat(),
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Exported registry to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export registry: {e}")
            return False

    async def import_registry(self, filepath: str) -> bool:
        """
        Import registry from a JSON file.

        Args:
            filepath: Path to import file

        Returns:
            True if import successful, False otherwise
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Import registry entries
            for tool_id, entry_data in data.get("registry", {}).items():
                entry = ToolRegistryEntry(**entry_data)
                self.registry[tool_id] = entry

                # Add to vector store
                self.vector_store.add_tool(entry.tool)

            # Import proposals
            for prop_id, prop_data in data.get("proposals", {}).items():
                self.proposals[prop_id] = ToolProposal(**prop_data)

            # Import evolution requests
            for req_id, req_data in data.get("evolution_requests", {}).items():
                self.evolution_requests[req_id] = ToolEvolutionRequest(**req_data)

            logger.info(f"Imported registry from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to import registry: {e}")
            return False
