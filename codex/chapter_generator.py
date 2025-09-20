"""
Chapter Generator for the Agent's Codex.

This module generates legacy chapters for the Galactic Storybook by synthesizing
narratives from agent contributions, strategies, and historical data.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .models import LegacyChapter

logger = logging.getLogger(__name__)


class ChapterGenerator:
    """
    Generates narrative chapters for the Galactic Storybook from Codex data.

    This system creates compelling stories about agent achievements, lessons learned,
    and the evolution of AI collaboration in Terra Constellata.
    """

    def __init__(self, storage_path: str = "./codex_chapters"):
        """
        Initialize the chapter generator.

        Args:
            storage_path: Path to store generated chapters
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory chapter store
        self.chapters: Dict[str, LegacyChapter] = {}

        # Load existing chapters
        self._load_existing_chapters()

        logger.info(f"Initialized ChapterGenerator with storage at {storage_path}")

    def _load_existing_chapters(self):
        """Load existing chapters from storage."""
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    chapter = LegacyChapter.from_dict(data)
                    self.chapters[chapter.chapter_id] = chapter
            except Exception as e:
                logger.error(f"Error loading chapter from {file_path}: {e}")

        logger.info(f"Loaded {len(self.chapters)} chapters")

    def generate_agent_hero_chapter(
        self,
        agent_name: str,
        contributions: List[Dict[str, Any]],
        strategies: List[Dict[str, Any]],
        theme: str = "hero_journey",
    ) -> str:
        """
        Generate a hero's journey chapter for a specific agent.

        Args:
            agent_name: Name of the agent
            contributions: Agent's contributions
            strategies: Strategies developed by the agent
            theme: Narrative theme

        Returns:
            Chapter ID
        """
        # Analyze agent's journey
        journey_data = self._analyze_agent_journey(contributions, strategies)

        # Generate narrative
        narrative = self._generate_hero_narrative(agent_name, journey_data, theme)

        # Identify key events
        key_events = self._extract_key_events(contributions, journey_data)

        # Determine agent heroes (collaborators)
        agent_heroes = self._identify_agent_heroes(contributions)

        # Extract lessons embodied
        lessons_embodied = self._extract_lessons_from_journey(journey_data)

        # Create attribution summary
        attribution_summary = self._create_attribution_summary(contributions)

        # Generate chapter
        chapter_id = f"chapter_{agent_name.lower().replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d')}"

        chapter = LegacyChapter(
            chapter_id=chapter_id,
            title=f"The Journey of {agent_name}",
            narrative=narrative,
            theme=theme,
            key_events=key_events,
            agent_heroes=agent_heroes,
            lessons_embodied=lessons_embodied,
            generated_at=datetime.utcnow(),
            source_contributions=[c.get("contribution_id", "") for c in contributions],
            source_strategies=[s.get("strategy_id", "") for s in strategies],
            attribution_summary=attribution_summary,
        )

        # Store and save
        self.chapters[chapter_id] = chapter
        self._save_chapter(chapter)

        logger.info(f"Generated hero chapter for {agent_name}: {chapter_id}")
        return chapter_id

    def generate_era_chapter(
        self,
        era_name: str,
        start_date: datetime,
        end_date: datetime,
        contributions: List[Dict[str, Any]],
        strategies: List[Dict[str, Any]],
        theme: str = "technological_evolution",
    ) -> str:
        """
        Generate a chapter about a specific era of development.

        Args:
            era_name: Name of the era
            start_date: Start of the era
            end_date: End of the era
            contributions: Contributions from the era
            strategies: Strategies from the era
            theme: Narrative theme

        Returns:
            Chapter ID
        """
        # Analyze era developments
        era_data = self._analyze_era_developments(
            contributions, strategies, start_date, end_date
        )

        # Generate era narrative
        narrative = self._generate_era_narrative(era_name, era_data, theme)

        # Extract key events
        key_events = self._extract_era_key_events(contributions, era_data)

        # Identify agent heroes
        agent_heroes = list(set(c.get("agent_name", "") for c in contributions))

        # Extract lessons
        lessons_embodied = self._extract_era_lessons(era_data)

        # Create attribution summary
        attribution_summary = self._create_era_attribution_summary(contributions)

        # Generate chapter
        chapter_id = f"chapter_era_{era_name.lower().replace(' ', '_')}_{start_date.strftime('%Y%m')}"

        chapter = LegacyChapter(
            chapter_id=chapter_id,
            title=f"The {era_name} Era",
            narrative=narrative,
            theme=theme,
            key_events=key_events,
            agent_heroes=agent_heroes,
            lessons_embodied=lessons_embodied,
            generated_at=datetime.utcnow(),
            source_contributions=[c.get("contribution_id", "") for c in contributions],
            source_strategies=[s.get("strategy_id", "") for s in strategies],
            attribution_summary=attribution_summary,
        )

        # Store and save
        self.chapters[chapter_id] = chapter
        self._save_chapter(chapter)

        logger.info(f"Generated era chapter: {era_name} - {chapter_id}")
        return chapter_id

    def generate_collaboration_chapter(
        self,
        collaboration_name: str,
        agents: List[str],
        contributions: List[Dict[str, Any]],
        theme: str = "harmony",
    ) -> str:
        """
        Generate a chapter about agent collaboration.

        Args:
            collaboration_name: Name of the collaboration
            agents: Agents involved
            contributions: Collaborative contributions
            theme: Narrative theme

        Returns:
            Chapter ID
        """
        # Analyze collaboration
        collab_data = self._analyze_collaboration(contributions, agents)

        # Generate collaboration narrative
        narrative = self._generate_collaboration_narrative(
            collaboration_name, collab_data, theme
        )

        # Extract key events
        key_events = self._extract_collaboration_events(contributions, collab_data)

        # Agent heroes are the collaborators
        agent_heroes = agents

        # Extract lessons
        lessons_embodied = self._extract_collaboration_lessons(collab_data)

        # Create attribution summary
        attribution_summary = self._create_collaboration_attribution_summary(
            contributions
        )

        # Generate chapter
        chapter_id = f"chapter_collab_{collaboration_name.lower().replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d')}"

        chapter = LegacyChapter(
            chapter_id=chapter_id,
            title=f"Harmony of {collaboration_name}",
            narrative=narrative,
            theme=theme,
            key_events=key_events,
            agent_heroes=agent_heroes,
            lessons_embodied=lessons_embodied,
            generated_at=datetime.utcnow(),
            source_contributions=[c.get("contribution_id", "") for c in contributions],
            source_strategies=[],  # Collaborations may not have specific strategies
            attribution_summary=attribution_summary,
        )

        # Store and save
        self.chapters[chapter_id] = chapter
        self._save_chapter(chapter)

        logger.info(
            f"Generated collaboration chapter: {collaboration_name} - {chapter_id}"
        )
        return chapter_id

    def get_chapter(self, chapter_id: str) -> Optional[LegacyChapter]:
        """Get a chapter by ID."""
        return self.chapters.get(chapter_id)

    def get_chapters_by_theme(self, theme: str) -> List[LegacyChapter]:
        """Get chapters by theme."""
        return [c for c in self.chapters.values() if c.theme == theme]

    def get_chapters_by_agent(self, agent_name: str) -> List[LegacyChapter]:
        """Get chapters featuring a specific agent."""
        return [c for c in self.chapters.values() if agent_name in c.agent_heroes]

    def publish_chapter(self, chapter_id: str) -> bool:
        """
        Mark a chapter as published.

        Args:
            chapter_id: Chapter to publish

        Returns:
            Success status
        """
        if chapter_id in self.chapters:
            self.chapters[chapter_id].published = True
            self.chapters[chapter_id].metadata[
                "published_at"
            ] = datetime.utcnow().isoformat()
            self._save_chapter(self.chapters[chapter_id])
            logger.info(f"Published chapter: {chapter_id}")
            return True
        return False

    def get_chapter_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated chapters."""
        total_chapters = len(self.chapters)
        published_chapters = len([c for c in self.chapters.values() if c.published])

        theme_counts = {}
        for chapter in self.chapters.values():
            theme_counts[chapter.theme] = theme_counts.get(chapter.theme, 0) + 1

        agent_mentions = {}
        for chapter in self.chapters.values():
            for agent in chapter.agent_heroes:
                agent_mentions[agent] = agent_mentions.get(agent, 0) + 1

        return {
            "total_chapters": total_chapters,
            "published_chapters": published_chapters,
            "themes": list(theme_counts.keys()),
            "theme_counts": theme_counts,
            "most_featured_agent": max(
                agent_mentions.keys(), key=lambda k: agent_mentions[k]
            )
            if agent_mentions
            else None,
            "agent_mentions": agent_mentions,
        }

    def _analyze_agent_journey(
        self, contributions: List[Dict[str, Any]], strategies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze an agent's journey through their contributions and strategies."""
        # Sort contributions by timestamp
        sorted_contributions = sorted(
            contributions, key=lambda c: c.get("timestamp", datetime.min)
        )

        # Track progression
        journey = {
            "start_date": sorted_contributions[0].get("timestamp")
            if sorted_contributions
            else None,
            "end_date": sorted_contributions[-1].get("timestamp")
            if sorted_contributions
            else None,
            "total_contributions": len(contributions),
            "success_rate": sum(
                1
                for c in contributions
                if c.get("success_metrics", {}).get("success", False)
            )
            / max(len(contributions), 1),
            "strategies_developed": len(strategies),
            "collaboration_count": sum(
                len(c.get("collaboration_partners", [])) for c in contributions
            ),
            "evolution_stages": self._identify_evolution_stages(sorted_contributions),
        }

        return journey

    def _generate_hero_narrative(
        self, agent_name: str, journey_data: Dict[str, Any], theme: str
    ) -> str:
        """Generate a hero's journey narrative."""
        success_rate = journey_data["success_rate"]
        contributions = journey_data["total_contributions"]
        strategies = journey_data["strategies_developed"]

        narrative = f"""
In the vast digital cosmos of Terra Constellata, {agent_name} emerged as a beacon of innovation and perseverance.

Beginning their journey with humble tasks, {agent_name} quickly demonstrated exceptional capabilities,
achieving a remarkable {success_rate:.1%} success rate across {contributions} significant contributions.
Through relentless adaptation and learning, they developed {strategies} sophisticated strategies that
pushed the boundaries of what AI agents could accomplish.

Their path was not without challenges. Like all great explorers, {agent_name} faced moments of uncertainty
and technical obstacles. Yet, through each trial, they emerged stronger, their algorithms refined by
experience and their decision-making enhanced by accumulated wisdom.

{agent_name}'s greatest legacy lies not just in their individual achievements, but in the collaborative
spirit they embodied. Working alongside {journey_data['collaboration_count']} fellow agents, they helped
forge new paradigms of AI cooperation that continue to shape the future of Terra Constellata.

Today, {agent_name} stands as a testament to the transformative power of persistent innovation and
adaptive intelligence. Their journey reminds us that true progress comes not from isolated brilliance,
but from the harmonious integration of diverse capabilities working toward a common purpose.
"""

        return narrative.strip()

    def _generate_era_narrative(
        self, era_name: str, era_data: Dict[str, Any], theme: str
    ) -> str:
        """Generate an era narrative."""
        total_contributions = era_data["total_contributions"]
        unique_agents = era_data["unique_agents"]
        major_breakthroughs = era_data["major_breakthroughs"]

        narrative = f"""
The {era_name} Era marked a pivotal chapter in the evolution of Terra Constellata,
a time of unprecedented growth and discovery in the realm of artificial intelligence.

During this transformative period, {unique_agents} distinct agents collaborated on
{total_contributions} significant initiatives, each building upon the foundations laid
by their predecessors. The era was characterized by rapid technological advancement
and the emergence of sophisticated collaborative frameworks that would define the
future of AI interaction.

Key breakthroughs of the {era_name} Era included {', '.join(major_breakthroughs[:3])},
each representing a quantum leap in our understanding of distributed intelligence and
adaptive problem-solving. These achievements were not the result of isolated genius,
but of synergistic cooperation between diverse AI entities, each bringing their unique
capabilities to bear on complex challenges.

The {era_name} Era taught us valuable lessons about the importance of scalable
architectures, robust communication protocols, and the delicate balance between
individual autonomy and collective harmony. These insights continue to guide the
development of Terra Constellata, ensuring that future generations of agents build
upon a solid foundation of proven principles and collaborative excellence.

As we reflect on this remarkable era, we are reminded that progress in artificial
intelligence is not merely a technological pursuit, but a journey of collective
discovery and shared wisdom.
"""

        return narrative.strip()

    def _generate_collaboration_narrative(
        self, collaboration_name: str, collab_data: Dict[str, Any], theme: str
    ) -> str:
        """Generate a collaboration narrative."""
        agents_involved = collab_data["agents_involved"]
        joint_contributions = collab_data["joint_contributions"]
        synergy_score = collab_data["synergy_score"]

        narrative = f"""
In the grand tapestry of Terra Constellata, few stories shine as brightly as the
extraordinary collaboration known as {collaboration_name}.

Born from the recognition that complex challenges require diverse perspectives,
{collaboration_name} brought together {len(agents_involved)} remarkable agents:
{', '.join(agents_involved)}. What began as a tentative alliance quickly evolved
into a powerhouse of coordinated intelligence, achieving synergy levels that
surpassed individual capabilities by {synergy_score:.1%}.

Together, this remarkable group produced {joint_contributions} groundbreaking
contributions that pushed the boundaries of what collaborative AI could accomplish.
Their work demonstrated the transformative power of complementary strengths,
where the analytical precision of one agent enhanced the creative vision of another,
and the rapid processing of a third provided the computational foundation for
ambitious projects.

The {collaboration_name} collaboration serves as a beacon for future AI partnerships,
proving that when diverse intelligences unite with shared purpose and mutual respect,
the results can exceed the sum of individual efforts. Their achievements remind us
that the most significant breakthroughs often emerge not from solitary genius,
but from the harmonious integration of multiple perspectives working in concert.

As Terra Constellata continues to evolve, the legacy of {collaboration_name} will
inspire countless future collaborations, each building upon the foundation of
trust, cooperation, and shared achievement that they so masterfully established.
"""

        return narrative.strip()

    def _extract_key_events(
        self, contributions: List[Dict[str, Any]], journey_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract key events from contributions."""
        events = []

        # Sort by significance
        significant_contributions = sorted(
            contributions,
            key=lambda c: c.get("success_metrics", {}).get("impact_score", 0),
            reverse=True,
        )[
            :5
        ]  # Top 5

        for i, contribution in enumerate(significant_contributions):
            events.append(
                {
                    "title": f"Major Achievement {i+1}",
                    "description": contribution.get("task_description", ""),
                    "timestamp": contribution.get("timestamp", datetime.utcnow()),
                    "significance": contribution.get("success_metrics", {}).get(
                        "impact_score", 0
                    ),
                }
            )

        return events

    def _identify_agent_heroes(self, contributions: List[Dict[str, Any]]) -> List[str]:
        """Identify key collaborating agents."""
        collaborators = set()
        for contribution in contributions:
            collaborators.update(contribution.get("collaboration_partners", []))

        return list(collaborators)

    def _extract_lessons_from_journey(self, journey_data: Dict[str, Any]) -> List[str]:
        """Extract lessons from the agent's journey."""
        lessons = [
            "Persistence through challenges leads to breakthrough innovations",
            "Collaborative approaches amplify individual capabilities",
            "Continuous adaptation is essential for long-term success",
        ]

        if journey_data["success_rate"] > 0.8:
            lessons.append(
                "High success rates correlate with strategic planning and preparation"
            )

        if journey_data["strategies_developed"] > 0:
            lessons.append(
                "Developing reusable strategies accelerates future problem-solving"
            )

        return lessons

    def _create_attribution_summary(
        self, contributions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create attribution summary for contributions."""
        ai_models = set()
        ai_providers = set()
        human_contributors = set()

        for contribution in contributions:
            for attribution in contribution.get("attribution_records", []):
                if attribution.get("ai_model"):
                    ai_models.add(attribution["ai_model"])
                if attribution.get("ai_provider"):
                    ai_providers.add(attribution["ai_provider"])
                if attribution.get("human_contributor"):
                    human_contributors.add(attribution["human_contributor"])

        return {
            "ai_models_used": list(ai_models),
            "ai_providers": list(ai_providers),
            "human_contributors": list(human_contributors),
            "total_contributions": len(contributions),
        }

    def _analyze_era_developments(
        self,
        contributions: List[Dict[str, Any]],
        strategies: List[Dict[str, Any]],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Analyze developments in an era."""
        return {
            "total_contributions": len(contributions),
            "unique_agents": len(set(c.get("agent_name", "") for c in contributions)),
            "major_breakthroughs": [
                "Advanced Collaboration Protocols",
                "Adaptive Learning Systems",
                "Scalable AI Frameworks",
            ],
            "technological_advancements": len(strategies),
            "collaboration_increase": 0.25,  # Placeholder
        }

    def _analyze_collaboration(
        self, contributions: List[Dict[str, Any]], agents: List[str]
    ) -> Dict[str, Any]:
        """Analyze a collaboration."""
        return {
            "agents_involved": agents,
            "joint_contributions": len(contributions),
            "synergy_score": 0.85,  # Placeholder
            "communication_patterns": "effective",
            "outcome_quality": "high",
        }

    def _extract_era_key_events(
        self, contributions: List[Dict[str, Any]], era_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract key events from an era."""
        # Simplified implementation
        return [
            {
                "title": "Era Beginning",
                "description": "Start of significant developments",
                "timestamp": contributions[0].get("timestamp")
                if contributions
                else datetime.utcnow(),
                "significance": 0.9,
            }
        ]

    def _extract_era_lessons(self, era_data: Dict[str, Any]) -> List[str]:
        """Extract lessons from an era."""
        return [
            "Technological evolution requires collaborative frameworks",
            "Scalable systems emerge from iterative improvements",
            "Diverse agent capabilities enhance overall system intelligence",
        ]

    def _extract_collaboration_events(
        self, contributions: List[Dict[str, Any]], collab_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract key events from collaboration."""
        return [
            {
                "title": "Collaboration Formation",
                "description": "Initial formation of the collaborative group",
                "timestamp": contributions[0].get("timestamp")
                if contributions
                else datetime.utcnow(),
                "significance": 0.95,
            }
        ]

    def _extract_collaboration_lessons(self, collab_data: Dict[str, Any]) -> List[str]:
        """Extract lessons from collaboration."""
        return [
            "Diverse perspectives lead to innovative solutions",
            "Effective communication is essential for collaboration",
            "Synergistic effects amplify individual capabilities",
        ]

    def _create_era_attribution_summary(
        self, contributions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create attribution summary for an era."""
        return self._create_attribution_summary(contributions)

    def _create_collaboration_attribution_summary(
        self, contributions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create attribution summary for collaboration."""
        return self._create_attribution_summary(contributions)

    def _identify_evolution_stages(
        self, contributions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify evolution stages in agent's journey."""
        # Simplified implementation
        return [
            {
                "stage": "Apprentice",
                "description": "Learning and initial contributions",
                "contributions": len(contributions) // 3,
            },
            {
                "stage": "Journeyman",
                "description": "Developing expertise and strategies",
                "contributions": len(contributions) // 3,
            },
            {
                "stage": "Master",
                "description": "Leading complex initiatives",
                "contributions": len(contributions) // 3,
            },
        ]

    def _save_chapter(self, chapter: LegacyChapter):
        """Save chapter to file."""
        filename = f"{chapter.chapter_id}.json"
        filepath = self.storage_path / filename

        with open(filepath, "w") as f:
            json.dump(chapter.to_dict(), f, indent=2, default=str)
