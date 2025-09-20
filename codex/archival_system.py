"""
Archival System for the Agent's Codex.

This module provides automated archival of agent contributions, workflow histories,
and successful strategies for preservation and future learning.
"""

import json
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .models import (
    AgentContribution,
    ContributionType,
    AttributionRecord,
    StrategyDocument,
    StrategyType,
)

logger = logging.getLogger(__name__)


class ArchivalSystem:
    """
    Automated archival system for capturing and preserving agent activities.

    This system integrates with the base agent framework to automatically
    capture contributions, workflows, and strategies for the Codex.
    """

    def __init__(self, storage_path: str = "./codex_archive"):
        """
        Initialize the archival system.

        Args:
            storage_path: Path to store archived data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.contributions_path = self.storage_path / "contributions"
        self.strategies_path = self.storage_path / "strategies"
        self.workflows_path = self.storage_path / "workflows"

        for path in [
            self.contributions_path,
            self.strategies_path,
            self.workflows_path,
        ]:
            path.mkdir(exist_ok=True)

        # In-memory caches
        self.contributions: Dict[str, AgentContribution] = {}
        self.strategies: Dict[str, StrategyDocument] = {}

        # Load existing data
        self._load_existing_data()

        logger.info(f"Initialized ArchivalSystem with storage at {storage_path}")

    def _load_existing_data(self):
        """Load existing archived data from storage."""
        # Load contributions
        for file_path in self.contributions_path.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    contribution = AgentContribution.from_dict(data)
                    self.contributions[contribution.contribution_id] = contribution
            except Exception as e:
                logger.error(f"Error loading contribution from {file_path}: {e}")

        # Load strategies
        for file_path in self.strategies_path.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    strategy = StrategyDocument.from_dict(data)
                    self.strategies[strategy.strategy_id] = strategy
            except Exception as e:
                logger.error(f"Error loading strategy from {file_path}: {e}")

        logger.info(
            f"Loaded {len(self.contributions)} contributions and {len(self.strategies)} strategies"
        )

    def archive_contribution(
        self,
        agent_name: str,
        agent_type: str,
        task_description: str,
        contribution_type: ContributionType,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        success_metrics: Dict[str, Any],
        duration: Optional[float] = None,
        workflow_context: Optional[str] = None,
        collaboration_partners: Optional[List[str]] = None,
        ai_model: Optional[str] = None,
        ai_provider: Optional[str] = None,
        human_contributor: Optional[str] = None,
    ) -> str:
        """
        Archive an agent contribution.

        Args:
            agent_name: Name of the contributing agent
            agent_type: Type of the agent
            task_description: Description of the task
            contribution_type: Type of contribution
            input_data: Input data for the task
            output_data: Output/result data
            success_metrics: Metrics measuring success
            duration: Time taken for the task
            workflow_context: Context of the workflow
            collaboration_partners: Other agents involved
            ai_model: AI model used (if applicable)
            ai_provider: AI provider (if applicable)
            human_contributor: Human contributor (if applicable)

        Returns:
            Contribution ID
        """
        # Generate unique ID
        content_hash = hashlib.md5(
            f"{agent_name}{task_description}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()
        contribution_id = f"contrib_{content_hash[:16]}"

        # Handle contribution_type conversion
        if isinstance(contribution_type, str):
            contribution_type = ContributionType(contribution_type)

        # Create attribution record
        attribution = AttributionRecord(
            agent_name=agent_name,
            agent_type=agent_type,
            contribution_type=contribution_type,
            timestamp=datetime.utcnow(),
            ai_model=ai_model,
            ai_provider=ai_provider,
            human_contributor=human_contributor,
        )

        # Create contribution record
        contribution = AgentContribution(
            contribution_id=contribution_id,
            agent_name=agent_name,
            agent_type=agent_type,
            task_description=task_description,
            contribution_type=contribution_type,
            input_data=input_data,
            output_data=output_data,
            success_metrics=success_metrics,
            timestamp=datetime.utcnow(),
            duration=duration,
            workflow_context=workflow_context,
            collaboration_partners=collaboration_partners or [],
            attribution_records=[attribution],
        )

        # Store in memory
        self.contributions[contribution_id] = contribution

        # Save to file
        self._save_contribution(contribution)

        logger.info(f"Archived contribution: {contribution_id} from {agent_name}")
        return contribution_id

    def archive_strategy(
        self,
        title: str,
        strategy_type: StrategyType,
        description: str,
        context: str,
        steps: List[Dict[str, Any]],
        success_criteria: List[str],
        lessons_learned: List[str],
        created_by: str,
        related_contributions: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Archive a documented strategy or pattern.

        Args:
            title: Strategy title
            strategy_type: Type of strategy
            description: Strategy description
            context: Context in which strategy applies
            steps: Steps of the strategy
            success_criteria: Criteria for success
            lessons_learned: Lessons from application
            created_by: Agent that created the strategy
            related_contributions: Related contribution IDs
            tags: Strategy tags

        Returns:
            Strategy ID
        """
        # Generate unique ID
        content_hash = hashlib.md5(
            f"{title}{created_by}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()
        strategy_id = f"strategy_{content_hash[:16]}"

        # Create strategy document
        strategy = StrategyDocument(
            strategy_id=strategy_id,
            strategy_type=strategy_type,
            title=title,
            description=description,
            context=context,
            steps=steps,
            success_criteria=success_criteria,
            lessons_learned=lessons_learned,
            created_by=created_by,
            created_at=datetime.utcnow(),
            related_contributions=related_contributions or [],
            tags=tags or [],
        )

        # Store in memory
        self.strategies[strategy_id] = strategy

        # Save to file
        self._save_strategy(strategy)

        logger.info(f"Archived strategy: {strategy_id} - {title}")
        return strategy_id

    def archive_workflow_trace(self, workflow_trace: Dict[str, Any]) -> str:
        """
        Archive a workflow trace from the WorkflowTracer.

        Args:
            workflow_trace: Workflow trace data

        Returns:
            Archive ID
        """
        # Generate unique ID
        workflow_id = workflow_trace.get("workflow_id", "unknown")
        archive_id = (
            f"workflow_{workflow_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

        # Save workflow trace
        filename = f"{archive_id}.json"
        filepath = self.workflows_path / filename

        with open(filepath, "w") as f:
            json.dump(workflow_trace, f, indent=2, default=str)

        logger.info(f"Archived workflow trace: {archive_id}")
        return archive_id

    def get_contribution(self, contribution_id: str) -> Optional[AgentContribution]:
        """Get a contribution by ID."""
        return self.contributions.get(contribution_id)

    def get_strategy(self, strategy_id: str) -> Optional[StrategyDocument]:
        """Get a strategy by ID."""
        return self.strategies.get(strategy_id)

    def get_contributions_by_agent(self, agent_name: str) -> List[AgentContribution]:
        """Get all contributions by a specific agent."""
        return [c for c in self.contributions.values() if c.agent_name == agent_name]

    def get_strategies_by_type(
        self, strategy_type: StrategyType
    ) -> List[StrategyDocument]:
        """Get strategies by type."""
        return [s for s in self.strategies.values() if s.strategy_type == strategy_type]

    def get_contributions_by_type(
        self, contribution_type: ContributionType
    ) -> List[AgentContribution]:
        """Get contributions by type."""
        return [
            c
            for c in self.contributions.values()
            if c.contribution_type == contribution_type
        ]

    def search_contributions(self, query: str) -> List[AgentContribution]:
        """Search contributions by text query."""
        query_lower = query.lower()
        results = []

        for contribution in self.contributions.values():
            if (
                query_lower in contribution.task_description.lower()
                or query_lower in contribution.agent_name.lower()
                or any(
                    query_lower in partner.lower()
                    for partner in contribution.collaboration_partners
                )
            ):
                results.append(contribution)

        return results

    def search_strategies(self, query: str) -> List[StrategyDocument]:
        """Search strategies by text query."""
        query_lower = query.lower()
        results = []

        for strategy in self.strategies.values():
            if (
                query_lower in strategy.title.lower()
                or query_lower in strategy.description.lower()
                or any(query_lower in tag.lower() for tag in strategy.tags)
            ):
                results.append(strategy)

        return results

    def get_archival_statistics(self) -> Dict[str, Any]:
        """Get statistics about archived data."""
        total_contributions = len(self.contributions)
        total_strategies = len(self.strategies)

        agent_contribution_counts = {}
        for contribution in self.contributions.values():
            agent = contribution.agent_name
            agent_contribution_counts[agent] = (
                agent_contribution_counts.get(agent, 0) + 1
            )

        strategy_type_counts = {}
        for strategy in self.strategies.values():
            stype = strategy.strategy_type.value
            strategy_type_counts[stype] = strategy_type_counts.get(stype, 0) + 1

        contribution_type_counts = {}
        for contribution in self.contributions.values():
            ctype = contribution.contribution_type.value
            contribution_type_counts[ctype] = contribution_type_counts.get(ctype, 0) + 1

        return {
            "total_contributions": total_contributions,
            "total_strategies": total_strategies,
            "agent_contribution_counts": agent_contribution_counts,
            "strategy_type_counts": strategy_type_counts,
            "contribution_type_counts": contribution_type_counts,
            "most_active_agent": max(
                agent_contribution_counts.keys(),
                key=lambda k: agent_contribution_counts[k],
            )
            if agent_contribution_counts
            else None,
        }

    def _save_contribution(self, contribution: AgentContribution):
        """Save contribution to file."""
        filename = f"{contribution.contribution_id}.json"
        filepath = self.contributions_path / filename

        with open(filepath, "w") as f:
            json.dump(contribution.to_dict(), f, indent=2, default=str)

    def _save_strategy(self, strategy: StrategyDocument):
        """Save strategy to file."""
        filename = f"{strategy.strategy_id}.json"
        filepath = self.strategies_path / filename

        with open(filepath, "w") as f:
            json.dump(strategy.to_dict(), f, indent=2, default=str)
