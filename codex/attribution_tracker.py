"""
Attribution Tracker for the Agent's Codex.

This module provides comprehensive tracking and attribution of contributions
from AI partners, ensuring proper credit and recognition for all collaborators.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from collections import defaultdict

from .models import AttributionRecord

logger = logging.getLogger(__name__)


class AttributionTracker:
    """
    Tracks and manages attribution for AI partners and contributors.

    This system ensures that all contributions to Terra Constellata are properly
    attributed, maintaining transparency and recognition for collaborative work.
    """

    def __init__(self, storage_path: str = "./codex_attribution"):
        """
        Initialize the attribution tracker.

        Args:
            storage_path: Path to store attribution data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory attribution store
        self.attribution_records: Dict[str, AttributionRecord] = {}
        self.agent_attributions: Dict[str, List[str]] = defaultdict(list)
        self.model_attributions: Dict[str, List[str]] = defaultdict(list)
        self.provider_attributions: Dict[str, List[str]] = defaultdict(list)

        # Load existing attribution data
        self._load_existing_attributions()

        logger.info(f"Initialized AttributionTracker with storage at {storage_path}")

    def _load_existing_attributions(self):
        """Load existing attribution records from storage."""
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    record = AttributionRecord.from_dict(data)
                    self.attribution_records[
                        record.agent_name + "_" + record.timestamp.isoformat()
                    ] = record

                    # Update indexes
                    self.agent_attributions[record.agent_name].append(
                        record.agent_name + "_" + record.timestamp.isoformat()
                    )
                    if record.ai_model:
                        self.model_attributions[record.ai_model].append(
                            record.agent_name + "_" + record.timestamp.isoformat()
                        )
                    if record.ai_provider:
                        self.provider_attributions[record.ai_provider].append(
                            record.agent_name + "_" + record.timestamp.isoformat()
                        )

            except Exception as e:
                logger.error(f"Error loading attribution record from {file_path}: {e}")

        logger.info(f"Loaded {len(self.attribution_records)} attribution records")

    def record_attribution(
        self,
        agent_name: str,
        agent_type: str,
        contribution_type: str,
        ai_model: Optional[str] = None,
        ai_provider: Optional[str] = None,
        human_contributor: Optional[str] = None,
        contribution_weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a new attribution.

        Args:
            agent_name: Name of the contributing agent
            agent_type: Type of the agent
            contribution_type: Type of contribution
            ai_model: AI model used
            ai_provider: AI provider
            human_contributor: Human contributor
            contribution_weight: Weight of the contribution
            metadata: Additional metadata

        Returns:
            Attribution record ID
        """
        from .models import ContributionType

        # Create attribution record
        record = AttributionRecord(
            agent_name=agent_name,
            agent_type=agent_type,
            contribution_type=ContributionType(contribution_type),
            timestamp=datetime.utcnow(),
            ai_model=ai_model,
            ai_provider=ai_provider,
            human_contributor=human_contributor,
            contribution_weight=contribution_weight,
            metadata=metadata or {},
        )

        # Generate unique ID
        record_id = f"{agent_name}_{record.timestamp.strftime('%Y%m%d_%H%M%S')}"

        # Store in memory
        self.attribution_records[record_id] = record

        # Update indexes
        self.agent_attributions[agent_name].append(record_id)
        if ai_model:
            self.model_attributions[ai_model].append(record_id)
        if ai_provider:
            self.provider_attributions[ai_provider].append(record_id)

        # Save to file
        self._save_attribution_record(record, record_id)

        logger.info(f"Recorded attribution for {agent_name}: {record_id}")
        return record_id

    def get_agent_attributions(
        self, agent_name: str, days: Optional[int] = None
    ) -> List[AttributionRecord]:
        """
        Get attribution records for a specific agent.

        Args:
            agent_name: Name of the agent
            days: Number of days to look back (optional)

        Returns:
            List of attribution records
        """
        record_ids = self.agent_attributions.get(agent_name, [])
        records = [
            self.attribution_records[rid]
            for rid in record_ids
            if rid in self.attribution_records
        ]

        if days:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            records = [r for r in records if r.timestamp >= cutoff_date]

        return sorted(records, key=lambda r: r.timestamp, reverse=True)

    def get_model_attributions(
        self, ai_model: str, days: Optional[int] = None
    ) -> List[AttributionRecord]:
        """
        Get attribution records for a specific AI model.

        Args:
            ai_model: Name of the AI model
            days: Number of days to look back (optional)

        Returns:
            List of attribution records
        """
        record_ids = self.model_attributions.get(ai_model, [])
        records = [
            self.attribution_records[rid]
            for rid in record_ids
            if rid in self.attribution_records
        ]

        if days:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            records = [r for r in records if r.timestamp >= cutoff_date]

        return sorted(records, key=lambda r: r.timestamp, reverse=True)

    def get_provider_attributions(
        self, ai_provider: str, days: Optional[int] = None
    ) -> List[AttributionRecord]:
        """
        Get attribution records for a specific AI provider.

        Args:
            ai_provider: Name of the AI provider
            days: Number of days to look back (optional)

        Returns:
            List of attribution records
        """
        record_ids = self.provider_attributions.get(ai_provider, [])
        records = [
            self.attribution_records[rid]
            for rid in record_ids
            if rid in self.attribution_records
        ]

        if days:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            records = [r for r in records if r.timestamp >= cutoff_date]

        return sorted(records, key=lambda r: r.timestamp, reverse=True)

    def get_contribution_summary(
        self,
        agent_name: Optional[str] = None,
        ai_model: Optional[str] = None,
        ai_provider: Optional[str] = None,
        days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get a summary of contributions.

        Args:
            agent_name: Filter by agent name
            ai_model: Filter by AI model
            ai_provider: Filter by AI provider
            days: Number of days to look back

        Returns:
            Contribution summary
        """
        # Get relevant records
        records = list(self.attribution_records.values())

        if agent_name:
            records = [r for r in records if r.agent_name == agent_name]
        if ai_model:
            records = [r for r in records if r.ai_model == ai_model]
        if ai_provider:
            records = [r for r in records if r.ai_provider == ai_provider]
        if days:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            records = [r for r in records if r.timestamp >= cutoff_date]

        # Calculate summary
        total_contributions = len(records)
        total_weight = sum(r.contribution_weight for r in records)

        # Contribution types
        type_counts = defaultdict(int)
        for record in records:
            type_counts[record.contribution_type.value] += 1

        # Agent types
        agent_type_counts = defaultdict(int)
        for record in records:
            agent_type_counts[record.agent_type] += 1

        # AI models and providers
        models_used = set(r.ai_model for r in records if r.ai_model)
        providers_used = set(r.ai_provider for r in records if r.ai_provider)
        human_contributors = set(
            r.human_contributor for r in records if r.human_contributor
        )

        return {
            "total_contributions": total_contributions,
            "total_weight": total_weight,
            "contribution_types": dict(type_counts),
            "agent_types": dict(agent_type_counts),
            "ai_models_used": list(models_used),
            "ai_providers_used": list(providers_used),
            "human_contributors": list(human_contributors),
            "avg_contribution_weight": total_weight / max(total_contributions, 1),
        }

    def get_top_contributors(
        self, by: str = "weight", limit: int = 10, days: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get top contributors by various metrics.

        Args:
            by: Metric to rank by ('weight', 'count', 'recent')
            limit: Number of top contributors to return
            days: Number of days to look back

        Returns:
            List of top contributors
        """
        records = list(self.attribution_records.values())

        if days:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            records = [r for r in records if r.timestamp >= cutoff_date]

        if by == "weight":
            # Group by agent and sum weights
            agent_weights = defaultdict(float)
            for record in records:
                agent_weights[record.agent_name] += record.contribution_weight

            sorted_agents = sorted(
                agent_weights.items(), key=lambda x: x[1], reverse=True
            )

        elif by == "count":
            # Group by agent and count contributions
            agent_counts = defaultdict(int)
            for record in records:
                agent_counts[record.agent_name] += 1

            sorted_agents = sorted(
                agent_counts.items(), key=lambda x: x[1], reverse=True
            )

        elif by == "recent":
            # Sort by timestamp and get most recent contributors
            sorted_records = sorted(records, key=lambda r: r.timestamp, reverse=True)
            recent_agents = []
            seen = set()

            for record in sorted_records:
                if record.agent_name not in seen:
                    recent_agents.append((record.agent_name, record.timestamp))
                    seen.add(record.agent_name)
                    if len(recent_agents) >= limit:
                        break

            sorted_agents = recent_agents

        else:
            return []

        return [
            {"agent_name": agent, "score": score, "metric": by}
            for agent, score in sorted_agents[:limit]
        ]

    def generate_attribution_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_individual_records: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive attribution report.

        Args:
            start_date: Start date for the report
            end_date: End date for the report
            include_individual_records: Whether to include individual records

        Returns:
            Attribution report
        """
        # Filter records by date range
        records = list(self.attribution_records.values())

        if start_date:
            records = [r for r in records if r.timestamp >= start_date]
        if end_date:
            records = [r for r in records if r.timestamp <= end_date]

        # Overall statistics
        total_records = len(records)
        date_range = {
            "start": start_date.isoformat() if start_date else None,
            "end": end_date.isoformat() if end_date else None,
        }

        # Agent statistics
        agent_stats = self.get_contribution_summary(
            days=None
            if not start_date
            else (end_date - start_date).days
            if end_date
            else None
        )

        # Top contributors
        top_by_weight = self.get_top_contributors(
            by="weight",
            limit=5,
            days=None
            if not start_date
            else (end_date - start_date).days
            if end_date
            else None,
        )
        top_by_count = self.get_top_contributors(
            by="count",
            limit=5,
            days=None
            if not start_date
            else (end_date - start_date).days
            if end_date
            else None,
        )

        # Contribution type distribution
        type_distribution = defaultdict(int)
        for record in records:
            type_distribution[record.contribution_type.value] += 1

        # AI model usage
        model_usage = defaultdict(int)
        for record in records:
            if record.ai_model:
                model_usage[record.ai_model] += 1

        # Provider distribution
        provider_distribution = defaultdict(int)
        for record in records:
            if record.ai_provider:
                provider_distribution[record.ai_provider] += 1

        report = {
            "report_generated_at": datetime.utcnow().isoformat(),
            "date_range": date_range,
            "total_records": total_records,
            "agent_statistics": agent_stats,
            "top_contributors_by_weight": top_by_weight,
            "top_contributors_by_count": top_by_count,
            "contribution_type_distribution": dict(type_distribution),
            "ai_model_usage": dict(model_usage),
            "ai_provider_distribution": dict(provider_distribution),
            "most_used_ai_model": max(model_usage.keys(), key=lambda k: model_usage[k])
            if model_usage
            else None,
            "most_used_ai_provider": max(
                provider_distribution.keys(), key=lambda k: provider_distribution[k]
            )
            if provider_distribution
            else None,
        }

        if include_individual_records:
            report["individual_records"] = [record.to_dict() for record in records]

        return report

    def export_attribution_data(self, filepath: str, format: str = "json") -> bool:
        """
        Export attribution data to a file.

        Args:
            filepath: Path to export file
            format: Export format ('json', 'csv')

        Returns:
            Success status
        """
        try:
            if format == "json":
                data = {
                    "exported_at": datetime.utcnow().isoformat(),
                    "records": [
                        record.to_dict() for record in self.attribution_records.values()
                    ],
                }
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2, default=str)

            elif format == "csv":
                import csv

                with open(filepath, "w", newline="") as f:
                    writer = csv.writer(f)
                    # Write header
                    writer.writerow(
                        [
                            "agent_name",
                            "agent_type",
                            "contribution_type",
                            "timestamp",
                            "ai_model",
                            "ai_provider",
                            "human_contributor",
                            "contribution_weight",
                        ]
                    )
                    # Write records
                    for record in self.attribution_records.values():
                        writer.writerow(
                            [
                                record.agent_name,
                                record.agent_type,
                                record.contribution_type.value,
                                record.timestamp.isoformat(),
                                record.ai_model or "",
                                record.ai_provider or "",
                                record.human_contributor or "",
                                record.contribution_weight,
                            ]
                        )

            logger.info(f"Exported attribution data to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error exporting attribution data: {e}")
            return False

    def _save_attribution_record(self, record: AttributionRecord, record_id: str):
        """Save attribution record to file."""
        filename = f"{record_id}.json"
        filepath = self.storage_path / filename

        with open(filepath, "w") as f:
            json.dump(record.to_dict(), f, indent=2, default=str)
