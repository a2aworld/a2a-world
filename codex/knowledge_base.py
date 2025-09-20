"""
Knowledge Base for the Agent's Codex.

This module provides a comprehensive knowledge base that stores lessons learned,
patterns, strategies, and insights for future agents to learn from predecessors.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from collections import defaultdict

from .models import KnowledgeEntry

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    Knowledge base system for storing and retrieving agent learning insights.

    This system extracts patterns, lessons, and strategies from archived data
    and makes them available for future agent learning and decision making.
    """

    def __init__(self, storage_path: str = "./codex_knowledge"):
        """
        Initialize the knowledge base.

        Args:
            storage_path: Path to store knowledge data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory knowledge store
        self.knowledge_entries: Dict[str, KnowledgeEntry] = {}
        self.category_index: Dict[str, List[str]] = defaultdict(list)
        self.tag_index: Dict[str, List[str]] = defaultdict(list)

        # Load existing knowledge
        self._load_existing_knowledge()

        logger.info(f"Initialized KnowledgeBase with storage at {storage_path}")

    def _load_existing_knowledge(self):
        """Load existing knowledge entries from storage."""
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    entry = KnowledgeEntry.from_dict(data)
                    self.knowledge_entries[entry.entry_id] = entry

                    # Update indexes
                    self.category_index[entry.category].append(entry.entry_id)
                    for tag in entry.tags:
                        self.tag_index[tag].append(entry.entry_id)

            except Exception as e:
                logger.error(f"Error loading knowledge entry from {file_path}: {e}")

        logger.info(f"Loaded {len(self.knowledge_entries)} knowledge entries")

    def add_knowledge_entry(
        self,
        category: str,
        title: str,
        content: str,
        source_type: str,
        source_id: str,
        confidence_score: float = 1.0,
        tags: Optional[List[str]] = None,
        related_entries: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a new knowledge entry.

        Args:
            category: Knowledge category
            title: Entry title
            content: Knowledge content
            source_type: Type of source (contribution, strategy, etc.)
            source_id: ID of the source
            confidence_score: Confidence in the knowledge
            tags: Knowledge tags
            related_entries: Related knowledge entry IDs
            metadata: Additional metadata

        Returns:
            Knowledge entry ID
        """
        # Generate unique ID
        entry_id = f"kb_{category}_{hash(title + content) % 1000000:06d}"

        # Create knowledge entry
        entry = KnowledgeEntry(
            entry_id=entry_id,
            category=category,
            title=title,
            content=content,
            source_type=source_type,
            source_id=source_id,
            confidence_score=confidence_score,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            tags=tags or [],
            related_entries=related_entries or [],
            metadata=metadata or {},
        )

        # Store in memory
        self.knowledge_entries[entry_id] = entry

        # Update indexes
        self.category_index[category].append(entry_id)
        for tag in entry.tags:
            self.tag_index[tag].append(entry_id)

        # Save to file
        self._save_entry(entry)

        logger.info(f"Added knowledge entry: {entry_id} - {title}")
        return entry_id

    def extract_patterns_from_contributions(
        self, contributions: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract patterns from agent contributions.

        Args:
            contributions: List of contribution data

        Returns:
            List of knowledge entry IDs created
        """
        entry_ids = []

        # Analyze success patterns
        success_patterns = self._analyze_success_patterns(contributions)
        for pattern in success_patterns:
            entry_id = self.add_knowledge_entry(
                category="success_patterns",
                title=f"Success Pattern: {pattern['title']}",
                content=pattern["description"],
                source_type="contribution_analysis",
                source_id="pattern_extraction",
                confidence_score=pattern["confidence"],
                tags=["pattern", "success", "learning"],
                metadata={
                    "pattern_type": pattern["type"],
                    "frequency": pattern["frequency"],
                },
            )
            entry_ids.append(entry_id)

        # Analyze collaboration patterns
        collab_patterns = self._analyze_collaboration_patterns(contributions)
        for pattern in collab_patterns:
            entry_id = self.add_knowledge_entry(
                category="collaboration_patterns",
                title=f"Collaboration Pattern: {pattern['title']}",
                content=pattern["description"],
                source_type="contribution_analysis",
                source_id="collaboration_analysis",
                confidence_score=pattern["confidence"],
                tags=["collaboration", "teamwork", "pattern"],
                metadata={
                    "agents_involved": pattern["agents"],
                    "success_rate": pattern["success_rate"],
                },
            )
            entry_ids.append(entry_id)

        # Analyze failure lessons
        failure_lessons = self._analyze_failure_lessons(contributions)
        for lesson in failure_lessons:
            entry_id = self.add_knowledge_entry(
                category="failure_lessons",
                title=f"Failure Lesson: {lesson['title']}",
                content=lesson["description"],
                source_type="contribution_analysis",
                source_id="failure_analysis",
                confidence_score=lesson["confidence"],
                tags=["failure", "lesson", "improvement"],
                metadata={
                    "failure_type": lesson["type"],
                    "recovery_strategy": lesson["recovery"],
                },
            )
            entry_ids.append(entry_id)

        return entry_ids

    def extract_insights_from_strategies(
        self, strategies: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract insights from documented strategies.

        Args:
            strategies: List of strategy data

        Returns:
            List of knowledge entry IDs created
        """
        entry_ids = []

        for strategy in strategies:
            # Extract general insights
            insights = self._extract_strategy_insights(strategy)
            for insight in insights:
                entry_id = self.add_knowledge_entry(
                    category="strategy_insights",
                    title=f"Strategy Insight: {insight['title']}",
                    content=insight["content"],
                    source_type="strategy",
                    source_id=strategy["strategy_id"],
                    confidence_score=insight["confidence"],
                    tags=["strategy", "insight", "optimization"],
                    metadata={
                        "strategy_type": strategy["strategy_type"],
                        "applicability": insight["applicability"],
                    },
                )
                entry_ids.append(entry_id)

        return entry_ids

    def search_knowledge(
        self,
        query: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> List[KnowledgeEntry]:
        """
        Search knowledge base for relevant entries.

        Args:
            query: Search query
            category: Filter by category
            tags: Filter by tags
            min_confidence: Minimum confidence score
            limit: Maximum results

        Returns:
            List of matching knowledge entries
        """
        candidates = list(self.knowledge_entries.values())

        # Filter by category
        if category:
            candidates = [e for e in candidates if e.category == category]

        # Filter by tags
        if tags:
            candidates = [e for e in candidates if any(tag in e.tags for tag in tags)]

        # Filter by confidence
        candidates = [e for e in candidates if e.confidence_score >= min_confidence]

        # Search by content
        query_lower = query.lower()
        matches = []
        for entry in candidates:
            score = 0
            if query_lower in entry.title.lower():
                score += 3
            if query_lower in entry.content.lower():
                score += 2
            if any(query_lower in tag.lower() for tag in entry.tags):
                score += 1

            if score > 0:
                matches.append((entry, score))

        # Sort by score and confidence
        matches.sort(key=lambda x: (x[1], x[0].confidence_score), reverse=True)

        # Update access counts
        results = []
        for entry, _ in matches[:limit]:
            entry.access_count += 1
            results.append(entry)

        return results

    def get_knowledge_by_category(self, category: str) -> List[KnowledgeEntry]:
        """Get all knowledge entries in a category."""
        entry_ids = self.category_index.get(category, [])
        return [
            self.knowledge_entries[eid]
            for eid in entry_ids
            if eid in self.knowledge_entries
        ]

    def get_knowledge_by_tag(self, tag: str) -> List[KnowledgeEntry]:
        """Get all knowledge entries with a specific tag."""
        entry_ids = self.tag_index.get(tag, [])
        return [
            self.knowledge_entries[eid]
            for eid in entry_ids
            if eid in self.knowledge_entries
        ]

    def get_related_knowledge(self, entry_id: str) -> List[KnowledgeEntry]:
        """Get knowledge entries related to a specific entry."""
        if entry_id not in self.knowledge_entries:
            return []

        entry = self.knowledge_entries[entry_id]
        related_ids = entry.related_entries
        return [
            self.knowledge_entries[eid]
            for eid in related_ids
            if eid in self.knowledge_entries
        ]

    def update_knowledge_usefulness(self, entry_id: str, usefulness_score: float):
        """
        Update the usefulness score of a knowledge entry.

        Args:
            entry_id: Knowledge entry ID
            usefulness_score: New usefulness score (0-1)
        """
        if entry_id in self.knowledge_entries:
            entry = self.knowledge_entries[entry_id]
            entry.usefulness_score = usefulness_score
            entry.last_updated = datetime.utcnow()
            self._save_entry(entry)

    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        total_entries = len(self.knowledge_entries)
        categories = set(entry.category for entry in self.knowledge_entries.values())
        avg_confidence = sum(
            e.confidence_score for e in self.knowledge_entries.values()
        ) / max(total_entries, 1)
        avg_usefulness = sum(
            e.usefulness_score for e in self.knowledge_entries.values()
        ) / max(total_entries, 1)

        category_counts = {}
        for entry in self.knowledge_entries.values():
            category_counts[entry.category] = category_counts.get(entry.category, 0) + 1

        tag_counts = {}
        for entry in self.knowledge_entries.values():
            for tag in entry.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "total_entries": total_entries,
            "categories": list(categories),
            "category_counts": category_counts,
            "tag_counts": tag_counts,
            "avg_confidence": avg_confidence,
            "avg_usefulness": avg_usefulness,
            "most_accessed": max(
                self.knowledge_entries.values(),
                key=lambda e: e.access_count,
                default=None,
            ),
        }

    def _analyze_success_patterns(
        self, contributions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze successful patterns from contributions."""
        # Simplified pattern analysis
        patterns = []

        # Pattern: High success with collaboration
        collab_contributions = [
            c for c in contributions if c.get("collaboration_partners")
        ]
        if collab_contributions:
            success_rate = sum(
                1
                for c in collab_contributions
                if c.get("success_metrics", {}).get("success", False)
            ) / len(collab_contributions)
            if success_rate > 0.7:
                patterns.append(
                    {
                        "title": "Collaboration Improves Success",
                        "description": f"Contributions involving collaboration have a {success_rate:.1%} success rate, suggesting collaborative approaches are more effective.",
                        "type": "collaboration",
                        "confidence": min(success_rate, 0.9),
                        "frequency": len(collab_contributions),
                    }
                )

        # Pattern: Fast execution correlates with success
        fast_contributions = [
            c for c in contributions if c.get("duration", float("inf")) < 60
        ]  # Under 1 minute
        if fast_contributions:
            success_rate = sum(
                1
                for c in fast_contributions
                if c.get("success_metrics", {}).get("success", False)
            ) / len(fast_contributions)
            if success_rate > 0.8:
                patterns.append(
                    {
                        "title": "Fast Execution Indicates Efficiency",
                        "description": f"Quick task completion (under 1 minute) correlates with {success_rate:.1%} success rate, indicating efficient processes.",
                        "type": "efficiency",
                        "confidence": min(success_rate * 0.8, 0.85),
                        "frequency": len(fast_contributions),
                    }
                )

        return patterns

    def _analyze_collaboration_patterns(
        self, contributions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze collaboration patterns."""
        patterns = []

        # Find most common collaboration pairs
        collab_pairs = defaultdict(int)
        for contribution in contributions:
            partners = contribution.get("collaboration_partners", [])
            if len(partners) >= 2:
                for i in range(len(partners)):
                    for j in range(i + 1, len(partners)):
                        pair = tuple(sorted([partners[i], partners[j]]))
                        collab_pairs[pair] += 1

        if collab_pairs:
            top_pair = max(collab_pairs.items(), key=lambda x: x[1])
            patterns.append(
                {
                    "title": f"Frequent Collaboration: {top_pair[0][0]} + {top_pair[0][1]}",
                    "description": f"These agents collaborate frequently ({top_pair[1]} times), suggesting effective partnership patterns.",
                    "agents": list(top_pair[0]),
                    "confidence": min(
                        top_pair[1] / 10, 0.9
                    ),  # Scale confidence by frequency
                    "success_rate": 0.85,  # Placeholder
                }
            )

        return patterns

    def _analyze_failure_lessons(
        self, contributions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze lessons from failures."""
        lessons = []

        failed_contributions = [
            c
            for c in contributions
            if not c.get("success_metrics", {}).get("success", True)
        ]
        if failed_contributions:
            # Common failure types
            timeout_failures = [
                c for c in failed_contributions if c.get("duration", 0) > 300
            ]  # Over 5 minutes
            if timeout_failures:
                lessons.append(
                    {
                        "title": "Avoid Long-Running Tasks",
                        "description": f"{len(timeout_failures)} contributions failed due to timeouts. Consider breaking down complex tasks or implementing timeouts.",
                        "type": "timeout",
                        "confidence": 0.8,
                        "recovery": "Implement task decomposition and timeout handling",
                    }
                )

        return lessons

    def _extract_strategy_insights(
        self, strategy: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract insights from a strategy document."""
        insights = []

        # Extract step-based insights
        steps = strategy.get("steps", [])
        if len(steps) > 5:
            insights.append(
                {
                    "title": "Complex Strategy Requires Careful Planning",
                    "content": f"Strategies with {len(steps)} steps need thorough validation and monitoring.",
                    "confidence": 0.7,
                    "applicability": "complex_tasks",
                }
            )

        # Extract success criteria insights
        success_criteria = strategy.get("success_criteria", [])
        if success_criteria:
            insights.append(
                {
                    "title": "Clear Success Criteria Improve Outcomes",
                    "content": f"Defining {len(success_criteria)} success criteria helps measure and validate strategy effectiveness.",
                    "confidence": 0.8,
                    "applicability": "strategy_design",
                }
            )

        return insights

    def _save_entry(self, entry: KnowledgeEntry):
        """Save knowledge entry to file."""
        filename = f"{entry.entry_id}.json"
        filepath = self.storage_path / filename

        with open(filepath, "w") as f:
            json.dump(entry.to_dict(), f, indent=2, default=str)
