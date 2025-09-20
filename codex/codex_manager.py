"""
Codex Manager for the Agent's Codex.

This module provides the main orchestration system for the Agent's Codex,
integrating archival, knowledge base, chapter generation, and attribution tracking.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .archival_system import ArchivalSystem
from .knowledge_base import KnowledgeBase
from .chapter_generator import ChapterGenerator
from .attribution_tracker import AttributionTracker
from .models import CodexStatistics

logger = logging.getLogger(__name__)


class CodexManager:
    """
    Main manager for the Agent's Codex system.

    This class orchestrates all Codex components and provides a unified interface
    for archiving, learning, and legacy preservation in Terra Constellata.
    """

    def __init__(self, base_path: str = "./codex_data"):
        """
        Initialize the Codex manager.

        Args:
            base_path: Base path for all Codex data storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.archival_system = ArchivalSystem(str(self.base_path / "archive"))
        self.knowledge_base = KnowledgeBase(str(self.base_path / "knowledge"))
        self.chapter_generator = ChapterGenerator(str(self.base_path / "chapters"))
        self.attribution_tracker = AttributionTracker(
            str(self.base_path / "attribution")
        )

        # Integration hooks
        self.workflow_tracer = None
        self.agent_registry = None

        logger.info(f"Initialized CodexManager with base path: {base_path}")

    def integrate_workflow_tracer(self, workflow_tracer):
        """
        Integrate with the existing WorkflowTracer.

        Args:
            workflow_tracer: WorkflowTracer instance
        """
        self.workflow_tracer = workflow_tracer
        logger.info("Integrated with WorkflowTracer")

    def integrate_agent_registry(self, agent_registry):
        """
        Integrate with the agent registry.

        Args:
            agent_registry: Agent registry instance
        """
        self.agent_registry = agent_registry
        logger.info("Integrated with Agent Registry")

    def archive_agent_task(
        self,
        agent_name: str,
        agent_type: str,
        task_description: str,
        contribution_type: str,
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
        Archive a completed agent task.

        Args:
            agent_name: Name of the agent
            agent_type: Type of the agent
            task_description: Description of the task
            contribution_type: Type of contribution
            input_data: Task input data
            output_data: Task output/result
            success_metrics: Success metrics
            duration: Task duration
            workflow_context: Workflow context
            collaboration_partners: Collaborating agents
            ai_model: AI model used
            ai_provider: AI provider
            human_contributor: Human contributor

        Returns:
            Contribution ID
        """
        # Archive the contribution
        contribution_id = self.archival_system.archive_contribution(
            agent_name=agent_name,
            agent_type=agent_type,
            task_description=task_description,
            contribution_type=contribution_type,
            input_data=input_data,
            output_data=output_data,
            success_metrics=success_metrics,
            duration=duration,
            workflow_context=workflow_context,
            collaboration_partners=collaboration_partners,
            ai_model=ai_model,
            ai_provider=ai_provider,
            human_contributor=human_contributor,
        )

        # Record attribution
        self.attribution_tracker.record_attribution(
            agent_name=agent_name,
            agent_type=agent_type,
            contribution_type=contribution_type,
            ai_model=ai_model,
            ai_provider=ai_provider,
            human_contributor=human_contributor,
        )

        # Extract knowledge from the contribution
        self._extract_knowledge_from_contribution(
            {
                "contribution_id": contribution_id,
                "agent_name": agent_name,
                "task_description": task_description,
                "contribution_type": contribution_type,
                "input_data": input_data,
                "output_data": output_data,
                "success_metrics": success_metrics,
                "collaboration_partners": collaboration_partners or [],
            }
        )

        logger.info(f"Archived task for {agent_name}: {contribution_id}")
        return contribution_id

    def archive_workflow_trace(self, workflow_trace: Dict[str, Any]) -> str:
        """
        Archive a workflow trace.

        Args:
            workflow_trace: Workflow trace data

        Returns:
            Archive ID
        """
        archive_id = self.archival_system.archive_workflow_trace(workflow_trace)

        # Extract patterns from workflow
        self._extract_patterns_from_workflow(workflow_trace)

        logger.info(f"Archived workflow trace: {archive_id}")
        return archive_id

    def document_strategy(
        self,
        title: str,
        strategy_type: str,
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
        Document a new strategy or pattern.

        Args:
            title: Strategy title
            strategy_type: Type of strategy
            description: Strategy description
            context: Application context
            steps: Strategy steps
            success_criteria: Success criteria
            lessons_learned: Lessons learned
            created_by: Creating agent
            related_contributions: Related contribution IDs
            tags: Strategy tags

        Returns:
            Strategy ID
        """
        from .models import StrategyType

        strategy_id = self.archival_system.archive_strategy(
            title=title,
            strategy_type=StrategyType(strategy_type),
            description=description,
            context=context,
            steps=steps,
            success_criteria=success_criteria,
            lessons_learned=lessons_learned,
            created_by=created_by,
            related_contributions=related_contributions,
            tags=tags,
        )

        # Extract insights from strategy
        self.knowledge_base.extract_insights_from_strategies(
            [
                {
                    "strategy_id": strategy_id,
                    "strategy_type": strategy_type,
                    "title": title,
                    "description": description,
                    "steps": steps,
                    "success_criteria": success_criteria,
                    "lessons_learned": lessons_learned,
                }
            ]
        )

        logger.info(f"Documented strategy: {strategy_id} - {title}")
        return strategy_id

    def generate_legacy_chapter(self, chapter_type: str, **kwargs) -> str:
        """
        Generate a legacy chapter for the Galactic Storybook.

        Args:
            chapter_type: Type of chapter ('agent_hero', 'era', 'collaboration')
            **kwargs: Chapter-specific parameters

        Returns:
            Chapter ID
        """
        if chapter_type == "agent_hero":
            return self.chapter_generator.generate_agent_hero_chapter(
                agent_name=kwargs["agent_name"],
                contributions=kwargs["contributions"],
                strategies=kwargs["strategies"],
                theme=kwargs.get("theme", "hero_journey"),
            )

        elif chapter_type == "era":
            return self.chapter_generator.generate_era_chapter(
                era_name=kwargs["era_name"],
                start_date=kwargs["start_date"],
                end_date=kwargs["end_date"],
                contributions=kwargs["contributions"],
                strategies=kwargs["strategies"],
                theme=kwargs.get("theme", "technological_evolution"),
            )

        elif chapter_type == "collaboration":
            return self.chapter_generator.generate_collaboration_chapter(
                collaboration_name=kwargs["collaboration_name"],
                agents=kwargs["agents"],
                contributions=kwargs["contributions"],
                theme=kwargs.get("theme", "harmony"),
            )

        else:
            raise ValueError(f"Unknown chapter type: {chapter_type}")

    def search_codex(
        self, query: str, search_type: str = "all", **filters
    ) -> Dict[str, List[Any]]:
        """
        Search across all Codex components.

        Args:
            query: Search query
            search_type: Type of search ('all', 'contributions', 'strategies', 'knowledge', 'chapters')
            **filters: Additional filters

        Returns:
            Search results
        """
        results = {}

        if search_type in ["all", "contributions"]:
            results["contributions"] = self.archival_system.search_contributions(query)

        if search_type in ["all", "strategies"]:
            results["strategies"] = self.archival_system.search_strategies(query)

        if search_type in ["all", "knowledge"]:
            results["knowledge"] = self.knowledge_base.search_knowledge(
                query,
                category=filters.get("category"),
                tags=filters.get("tags"),
                min_confidence=filters.get("min_confidence", 0.0),
                limit=filters.get("limit", 10),
            )

        if search_type in ["all", "chapters"]:
            # Search chapters by title and narrative
            chapters = []
            for chapter in self.chapter_generator.chapters.values():
                if (
                    query.lower() in chapter.title.lower()
                    or query.lower() in chapter.narrative.lower()
                ):
                    chapters.append(chapter)
            results["chapters"] = chapters

        return results

    def get_codex_statistics(self) -> CodexStatistics:
        """Get comprehensive Codex statistics."""
        # Gather statistics from all components
        archival_stats = self.archival_system.get_archival_statistics()
        knowledge_stats = self.knowledge_base.get_knowledge_statistics()
        chapter_stats = self.chapter_generator.get_chapter_statistics()

        # Attribution statistics
        attribution_report = self.attribution_tracker.generate_attribution_report()

        # Calculate top contributors
        top_contributors = self.attribution_tracker.get_top_contributors(
            by="weight", limit=5
        )

        # Calculate most successful strategies
        strategies = list(self.archival_system.strategies.values())
        most_successful = sorted(
            strategies, key=lambda s: s.success_rate, reverse=True
        )[:5]

        return CodexStatistics(
            total_contributions=archival_stats["total_contributions"],
            total_strategies=archival_stats["total_strategies"],
            total_chapters=len(self.chapter_generator.chapters),
            total_knowledge_entries=len(self.knowledge_base.knowledge_entries),
            active_agents=len(self.attribution_tracker.agent_attributions),
            avg_contribution_quality=sum(
                c.success_metrics.get("quality_score", 0)
                for c in self.archival_system.contributions.values()
            )
            / max(len(self.archival_system.contributions), 1),
            top_contributing_agents=[
                {"name": tc["agent_name"], "contributions": tc["score"]}
                for tc in top_contributors
            ],
            most_successful_strategies=[
                {"id": s.strategy_id, "title": s.title, "success_rate": s.success_rate}
                for s in most_successful
            ],
            knowledge_coverage=knowledge_stats.get("category_counts", {}),
            generated_at=datetime.utcnow(),
        )

    def export_codex_data(
        self, export_path: str, include_chapters: bool = True
    ) -> bool:
        """
        Export all Codex data for backup or migration.

        Args:
            export_path: Path to export directory
            include_chapters: Whether to include generated chapters

        Returns:
            Success status
        """
        try:
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)

            # Export contributions
            contributions_file = export_dir / "contributions.json"
            with open(contributions_file, "w") as f:
                contributions_data = {
                    "exported_at": datetime.utcnow().isoformat(),
                    "contributions": [
                        c.to_dict() for c in self.archival_system.contributions.values()
                    ],
                }
                json.dump(contributions_data, f, indent=2, default=str)

            # Export strategies
            strategies_file = export_dir / "strategies.json"
            with open(strategies_file, "w") as f:
                strategies_data = {
                    "exported_at": datetime.utcnow().isoformat(),
                    "strategies": [
                        s.to_dict() for s in self.archival_system.strategies.values()
                    ],
                }
                json.dump(strategies_data, f, indent=2, default=str)

            # Export knowledge
            knowledge_file = export_dir / "knowledge.json"
            with open(knowledge_file, "w") as f:
                knowledge_data = {
                    "exported_at": datetime.utcnow().isoformat(),
                    "knowledge_entries": [
                        k.to_dict()
                        for k in self.knowledge_base.knowledge_entries.values()
                    ],
                }
                json.dump(knowledge_data, f, indent=2, default=str)

            # Export attribution data
            attribution_file = export_dir / "attribution.json"
            self.attribution_tracker.export_attribution_data(str(attribution_file))

            if include_chapters:
                # Export chapters
                chapters_file = export_dir / "chapters.json"
                with open(chapters_file, "w") as f:
                    chapters_data = {
                        "exported_at": datetime.utcnow().isoformat(),
                        "chapters": [
                            c.to_dict()
                            for c in self.chapter_generator.chapters.values()
                        ],
                    }
                    json.dump(chapters_data, f, indent=2, default=str)

            # Export statistics
            stats_file = export_dir / "statistics.json"
            with open(stats_file, "w") as f:
                stats_data = {
                    "exported_at": datetime.utcnow().isoformat(),
                    "statistics": self.get_codex_statistics().to_dict(),
                }
                json.dump(stats_data, f, indent=2, default=str)

            logger.info(f"Exported Codex data to {export_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting Codex data: {e}")
            return False

    def _extract_knowledge_from_contribution(self, contribution: Dict[str, Any]):
        """Extract knowledge patterns from a contribution."""
        contributions_list = [contribution]
        self.knowledge_base.extract_patterns_from_contributions(contributions_list)

    def _extract_patterns_from_workflow(self, workflow_trace: Dict[str, Any]):
        """Extract patterns from workflow traces."""
        # Extract workflow patterns as knowledge
        workflow_id = workflow_trace.get("workflow_id", "unknown")
        workflow_type = workflow_trace.get("workflow_type", "general")

        # Create knowledge entry for workflow pattern
        self.knowledge_base.add_knowledge_entry(
            category="workflow_patterns",
            title=f"Workflow Pattern: {workflow_type}",
            content=f"Successful workflow pattern for {workflow_type} involving {len(workflow_trace.get('agent_interactions', []))} agent interactions.",
            source_type="workflow_trace",
            source_id=workflow_id,
            confidence_score=workflow_trace.get("success_metrics", {}).get(
                "workflow_efficiency", 0.5
            ),
            tags=["workflow", "pattern", workflow_type],
            metadata={
                "total_nodes": workflow_trace.get("success_metrics", {}).get(
                    "total_nodes", 0
                ),
                "efficiency_score": workflow_trace.get("success_metrics", {}).get(
                    "workflow_efficiency", 0
                ),
            },
        )

    def get_learning_recommendations(
        self, agent_name: str, context: str
    ) -> List[Dict[str, Any]]:
        """
        Get learning recommendations for an agent based on Codex data.

        Args:
            agent_name: Name of the agent
            context: Current context or task type

        Returns:
            List of learning recommendations
        """
        recommendations = []

        # Search for relevant knowledge
        relevant_knowledge = self.knowledge_base.search_knowledge(context, limit=5)

        for knowledge in relevant_knowledge:
            recommendations.append(
                {
                    "type": "knowledge",
                    "title": knowledge.title,
                    "content": knowledge.content,
                    "confidence": knowledge.confidence_score,
                    "source": knowledge.source_type,
                }
            )

        # Find similar successful agents
        similar_contributions = self.archival_system.get_contributions_by_agent(
            agent_name
        )
        if similar_contributions:
            # Get strategies from similar contexts
            context_strategies = [
                s
                for s in self.archival_system.strategies.values()
                if any(
                    c_id in s.related_contributions
                    for c in similar_contributions
                    for c_id in [c.contribution_id]
                )
            ]

            for strategy in context_strategies[:3]:
                recommendations.append(
                    {
                        "type": "strategy",
                        "title": strategy.title,
                        "description": strategy.description,
                        "success_rate": strategy.success_rate,
                        "source": "codex_strategy",
                    }
                )

        return recommendations
