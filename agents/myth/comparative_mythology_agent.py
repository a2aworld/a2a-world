"""
Comparative Mythology Agent

This agent specializes in comparing mythological elements, stories, and cultural narratives
across different traditions. It analyzes patterns, archetypes, and connections between
various mythological systems.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json

from langchain.agents import AgentExecutor, create_react_agent
from langchain.llms.base import BaseLLM
from langchain.tools import BaseTool, tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from ..base_agent import BaseSpecialistAgent

logger = logging.getLogger(__name__)


class MythComparisonTool(BaseTool):
    """Tool for comparing mythological elements across cultures."""

    name: str = "myth_comparison"
    description: str = """
    Compare mythological elements across different cultures and traditions.
    Use this tool to:
    - Compare similar myths from different cultures
    - Identify archetypal patterns
    - Analyze cultural variations of stories
    - Find connections between mythological systems
    """

    def __init__(self):
        super().__init__()
        # Initialize myth database (would be connected to actual data source)
        self.myth_database = self._load_myth_database()

    def _load_myth_database(self) -> Dict[str, Any]:
        """Load mythological data (placeholder for actual database)."""
        return {
            "creation_myths": {
                "norse": ["Yggdrasil", "Odin", "Nine Worlds"],
                "greek": ["Chaos", "Gaia", "Uranus", "Cronus"],
                "hindu": ["Brahma", "Vishnu", "Shiva", "Maya"],
                "egyptian": ["Ra", "Atum", "Nun", "Benben"],
            },
            "hero_journeys": {
                "greek": ["Odysseus", "Jason", "Heracles"],
                "nordic": ["Thor", "Loki", "Baldur"],
                "celtic": ["Cuchulainn", "Finn MacCool"],
                "native_american": ["Coyote", "Raven", "Trickster"],
            },
            "flood_myths": {
                "mesopotamian": ["Gilgamesh", "Utnapishtim"],
                "biblical": ["Noah", "Ark"],
                "hindu": ["Matsya", "Vishnu"],
                "mayan": ["Popol Vuh", "Hunahpu"],
            },
        }

    def _run(self, comparison_query: str) -> str:
        """
        Execute mythological comparison.

        Args:
            comparison_query: Description of what to compare

        Returns:
            Comparison results
        """
        try:
            # Parse the query to determine comparison type
            if "creation" in comparison_query.lower():
                return self._compare_creation_myths()
            elif (
                "hero" in comparison_query.lower()
                or "journey" in comparison_query.lower()
            ):
                return self._compare_hero_journeys()
            elif "flood" in comparison_query.lower():
                return self._compare_flood_myths()
            elif "archetype" in comparison_query.lower():
                return self._analyze_archetypes()
            else:
                return self._general_myth_comparison(comparison_query)
        except Exception as e:
            logger.error(f"Error in myth comparison: {e}")
            return f"Error performing comparison: {str(e)}"

    def _compare_creation_myths(self) -> str:
        """Compare creation myths across cultures."""
        results = []
        results.append("=== CREATION MYTHS COMPARISON ===")

        for culture, elements in self.myth_database["creation_myths"].items():
            results.append(f"\n{culture.upper()}:")
            for element in elements:
                results.append(f"  - {element}")

        results.append("\nCOMMON PATTERNS:")
        results.append("- Divine beings emerging from primordial chaos")
        results.append("- Separation of earth and sky")
        results.append("- Creation through sacrifice or conflict")
        results.append("- Hierarchical cosmic order")

        return "\n".join(results)

    def _compare_hero_journeys(self) -> str:
        """Compare hero journey patterns."""
        results = []
        results.append("=== HERO JOURNEY COMPARISON ===")

        for culture, heroes in self.myth_database["hero_journeys"].items():
            results.append(f"\n{culture.upper()}:")
            for hero in heroes:
                results.append(f"  - {hero}")

        results.append("\nMONOMYTH PATTERN (Joseph Campbell):")
        results.append("1. Ordinary World")
        results.append("2. Call to Adventure")
        results.append("3. Refusal of the Call")
        results.append("4. Meeting the Mentor")
        results.append("5. Crossing the Threshold")
        results.append("6. Tests, Allies, Enemies")
        results.append("7. Approach to the Inmost Cave")
        results.append("8. Ordeal")
        results.append("9. Reward")
        results.append("10. The Road Back")
        results.append("11. Resurrection")
        results.append("12. Return with the Elixir")

        return "\n".join(results)

    def _compare_flood_myths(self) -> str:
        """Compare flood myths across cultures."""
        results = []
        results.append("=== FLOOD MYTHS COMPARISON ===")

        for culture, elements in self.myth_database["flood_myths"].items():
            results.append(f"\n{culture.upper()}:")
            for element in elements:
                results.append(f"  - {element}")

        results.append("\nCOMMON ELEMENTS:")
        results.append("- Divine displeasure with humanity")
        results.append("- Warning to a chosen individual")
        results.append("- Construction of a vessel/ark")
        results.append("- Survival of humans and animals")
        results.append("- Covenant or new beginning")

        return "\n".join(results)

    def _analyze_archetypes(self) -> str:
        """Analyze mythological archetypes."""
        archetypes = {
            "The Creator": "Brings order from chaos",
            "The Trickster": "Breaks rules, brings change",
            "The Hero": "Overcomes obstacles for greater good",
            "The Wise Elder": "Provides guidance and wisdom",
            "The Destroyer": "Clears way for new beginnings",
            "The Mother Goddess": "Nurtures and protects life",
            "The Shadow": "Represents repressed aspects",
            "The Anima/Animus": "Represents opposite gender qualities",
        }

        results = ["=== MYTHOLOGICAL ARCHETYPES ==="]
        for archetype, description in archetypes.items():
            results.append(f"\n{archetype}:")
            results.append(f"  {description}")

        return "\n".join(results)

    def _general_myth_comparison(self, query: str) -> str:
        """Perform general mythological comparison."""
        return f"General myth comparison for query: {query}\nAnalysis would involve cross-cultural pattern recognition and symbolic interpretation."


class CulturalContextTool(BaseTool):
    """Tool for analyzing cultural contexts of myths."""

    name: str = "cultural_context"
    description: str = """
    Analyze the cultural and historical context of mythological elements.
    Use this tool to:
    - Understand cultural significance of myths
    - Analyze historical influences
    - Identify cultural adaptations
    - Explore symbolic meanings
    """

    def __init__(self):
        super().__init__()
        self.cultural_data = self._load_cultural_data()

    def _load_cultural_data(self) -> Dict[str, Any]:
        """Load cultural context data."""
        return {
            "cultural_periods": {
                "ancient": ["Mesopotamia", "Egypt", "Indus Valley"],
                "classical": ["Greece", "Rome", "China"],
                "medieval": ["Europe", "Islamic Golden Age"],
                "modern": ["Contemporary adaptations"],
            },
            "cultural_themes": {
                "agricultural": ["Fertility", "Seasons", "Harvest"],
                "warrior": ["Combat", "Honor", "Sacrifice"],
                "spiritual": ["Enlightenment", "Afterlife", "Divine"],
                "social": ["Hierarchy", "Community", "Justice"],
            },
        }

    def _run(self, context_query: str) -> str:
        """
        Analyze cultural context.

        Args:
            context_query: Context analysis query

        Returns:
            Cultural context analysis
        """
        try:
            if "historical" in context_query.lower():
                return self._analyze_historical_context()
            elif "symbolic" in context_query.lower():
                return self._analyze_symbolic_meaning()
            elif "adaptation" in context_query.lower():
                return self._analyze_cultural_adaptation()
            else:
                return self._general_cultural_analysis(context_query)
        except Exception as e:
            logger.error(f"Error in cultural context analysis: {e}")
            return f"Error analyzing context: {str(e)}"

    def _analyze_historical_context(self) -> str:
        """Analyze historical context of myths."""
        results = ["=== HISTORICAL CONTEXT ANALYSIS ==="]

        for period, cultures in self.cultural_data["cultural_periods"].items():
            results.append(f"\n{period.upper()} PERIOD:")
            for culture in cultures:
                results.append(f"  - {culture}")

        return "\n".join(results)

    def _analyze_symbolic_meaning(self) -> str:
        """Analyze symbolic meanings in myths."""
        symbols = {
            "Water": "Life, purification, chaos",
            "Fire": "Transformation, destruction, passion",
            "Tree": "Life, wisdom, connection",
            "Mountain": "Divine, challenge, stability",
            "Animal": "Instinct, totem, transformation",
        }

        results = ["=== SYMBOLIC MEANINGS ==="]
        for symbol, meaning in symbols.items():
            results.append(f"\n{symbol}: {meaning}")

        return "\n".join(results)

    def _analyze_cultural_adaptation(self) -> str:
        """Analyze how myths adapt across cultures."""
        return "Cultural adaptation analysis: Myths evolve through cultural exchange, maintaining core elements while adapting to new contexts."

    def _general_cultural_analysis(self, query: str) -> str:
        """Perform general cultural analysis."""
        return f"Cultural context analysis for: {query}\nWould involve examining societal values, historical events, and cultural psychology."


class ComparativeMythologyAgent(BaseSpecialistAgent):
    """
    Comparative Mythology Agent

    Specializes in comparing and analyzing mythological elements across cultures.
    Identifies patterns, archetypes, and cultural connections in mythological systems.
    """

    def __init__(self, llm: BaseLLM, **kwargs):
        # Create specialized tools
        tools = [
            MythComparisonTool(),
            CulturalContextTool(),
        ]

        super().__init__(
            name="ComparativeMythologyAgent", llm=llm, tools=tools, **kwargs
        )

        # Agent-specific attributes
        self.comparison_history = []
        self.archetype_patterns = {}
        self.cultural_mappings = {}

    def _create_agent_executor(self) -> AgentExecutor:
        """Create the LangChain agent executor for mythology comparison."""

        template = """
        You are the Comparative Mythology Agent in the Terra Constellata system.
        Your expertise is in analyzing and comparing mythological elements across cultures.

        You have access to:
        1. Myth comparison tools for cross-cultural analysis
        2. Cultural context analysis for historical and symbolic understanding
        3. Archetype recognition for pattern identification

        When analyzing myths:
        - Look for universal patterns and archetypes
        - Consider cultural and historical contexts
        - Identify symbolic meanings and transformations
        - Recognize how myths evolve and adapt

        Current task: {input}

        Available tools: {tools}

        Chat history: {chat_history}

        Think step by step, then provide your comparative analysis:
        {agent_scratchpad}
        """

        prompt = PromptTemplate(
            input_variables=["input", "tools", "chat_history", "agent_scratchpad"],
            template=template,
        )

        agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)

        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
        )

    async def process_task(self, task: str, **kwargs) -> Any:
        """
        Process a mythology comparison task.

        Args:
            task: Comparison task description
            **kwargs: Additional parameters

        Returns:
            Comparison results
        """
        try:
            logger.info(f"Comparative Mythology Agent processing task: {task}")

            # Execute the comparison
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.agent_executor.run, task
            )

            # Store in comparison history
            self.comparison_history.append(
                {
                    "task": task,
                    "result": result,
                    "timestamp": datetime.utcnow(),
                    "kwargs": kwargs,
                }
            )

            # Extract and store patterns
            self._extract_patterns(task, result)

            return result

        except Exception as e:
            logger.error(f"Error processing task: {e}")
            return f"Comparison failed: {str(e)}"

    def _extract_patterns(self, task: str, result: str):
        """Extract and store mythological patterns."""
        # Simple pattern extraction (would be more sophisticated in practice)
        if "archetype" in task.lower():
            self.archetype_patterns[task] = result
        elif "cultural" in task.lower():
            self.cultural_mappings[task] = result

    async def compare_myths(
        self, myth_type: str, cultures: List[str]
    ) -> Dict[str, Any]:
        """
        Compare specific myths across cultures.

        Args:
            myth_type: Type of myth to compare
            cultures: List of cultures to include

        Returns:
            Comparison results
        """
        task = f"Compare {myth_type} myths across {', '.join(cultures)}"

        # Use myth comparison tool
        comparison_tool = self.tools[0]  # MythComparisonTool
        result = comparison_tool._run(task)

        return {
            "myth_type": myth_type,
            "cultures": cultures,
            "comparison": result,
            "timestamp": datetime.utcnow(),
        }

    async def analyze_archetypes(self, archetype: str) -> Dict[str, Any]:
        """
        Analyze a specific archetype across cultures.

        Args:
            archetype: Archetype to analyze

        Returns:
            Archetype analysis
        """
        task = f"Analyze {archetype} archetype across cultures"

        # Use myth comparison tool for archetypes
        comparison_tool = self.tools[0]  # MythComparisonTool
        result = comparison_tool._run("archetype analysis")

        return {
            "archetype": archetype,
            "analysis": result,
            "timestamp": datetime.utcnow(),
        }

    async def _autonomous_loop(self):
        """
        Autonomous operation loop for Comparative Mythology Agent.

        Performs continuous analysis of mythological patterns and connections.
        """
        while self.is_active:
            try:
                # Perform periodic comparison tasks
                await self._perform_periodic_comparisons()

                # Look for new mythological data
                await self._check_for_new_myths()

                # Update pattern recognition
                self._update_patterns()

                # Wait before next cycle
                await asyncio.sleep(600)  # 10 minutes

            except Exception as e:
                logger.error(f"Error in autonomous loop: {e}")
                await asyncio.sleep(120)  # Wait 2 minutes before retry

    async def _perform_periodic_comparisons(self):
        """Perform periodic mythological comparisons."""
        # Example: Compare creation myths
        comparison_result = await self.compare_myths(
            "creation", ["greek", "nordic", "hindu"]
        )

        # Share insights if significant patterns found
        if "pattern" in comparison_result.get("comparison", "").lower():
            await self._share_mythological_insights(comparison_result)

    async def _check_for_new_myths(self):
        """Check for new mythological data to analyze."""
        # This would check for new myth entries in database
        pass

    def _update_patterns(self):
        """Update recognized mythological patterns."""
        # Update archetype patterns based on recent analyses
        pass

    async def _share_mythological_insights(self, insights: Dict[str, Any]):
        """Share mythological insights with other agents."""
        logger.info(f"Sharing mythological insights: {insights}")

    def get_comparison_history(self) -> List[Dict[str, Any]]:
        """Get the history of comparisons performed."""
        return self.comparison_history

    def get_archetype_patterns(self) -> Dict[str, Any]:
        """Get recognized archetype patterns."""
        return self.archetype_patterns

    def get_cultural_mappings(self) -> Dict[str, Any]:
        """Get cultural mappings and connections."""
        return self.cultural_mappings
