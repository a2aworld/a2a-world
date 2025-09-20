"""
Linguist Agent

This agent specializes in language processing, analysis, and linguistic pattern recognition.
It handles text analysis, language identification, translation, and linguistic insights.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import re

from langchain.agents import AgentExecutor, create_react_agent
from langchain.llms.base import BaseLLM
from langchain.tools import BaseTool, tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from ..base_agent import BaseSpecialistAgent

logger = logging.getLogger(__name__)


class TextAnalysisTool(BaseTool):
    """Tool for analyzing text and linguistic patterns."""

    name: str = "text_analysis"
    description: str = """
    Analyze text for linguistic patterns, structure, and meaning.
    Use this tool to:
    - Identify language and dialect
    - Analyze sentence structure
    - Extract key linguistic features
    - Detect stylistic patterns
    """

    def __init__(self):
        super().__init__()
        self.language_patterns = self._load_language_patterns()

    def _load_language_patterns(self) -> Dict[str, Any]:
        """Load language identification patterns."""
        return {
            "english": {
                "common_words": ["the", "and", "is", "in", "to", "of", "a", "that"],
                "sentence_enders": [".", "!", "?"],
                "vowels": "aeiouAEIOU",
            },
            "spanish": {
                "common_words": ["el", "la", "de", "que", "y", "en", "un", "es"],
                "sentence_enders": [".", "!", "¿", "?"],
                "vowels": "aeiouáéíóúAEIOUÁÉÍÓÚ",
            },
            "french": {
                "common_words": ["le", "la", "et", "à", "un", "il", "être", "et"],
                "sentence_enders": [".", "!", "?"],
                "vowels": "aeiouàâéèêëïîôùûüAEIOUÀÂÉÈÊËÏÎÔÙÛÜ",
            },
        }

    def _run(self, text: str) -> str:
        """
        Analyze text linguistically.

        Args:
            text: Text to analyze

        Returns:
            Linguistic analysis results
        """
        try:
            analysis = []

            # Basic text statistics
            analysis.append(self._basic_statistics(text))

            # Language identification
            analysis.append(self._identify_language(text))

            # Sentence structure analysis
            analysis.append(self._analyze_sentence_structure(text))

            # Stylistic features
            analysis.append(self._extract_stylistic_features(text))

            return "\n".join(analysis)

        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            return f"Error analyzing text: {str(e)}"

    def _basic_statistics(self, text: str) -> str:
        """Calculate basic text statistics."""
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        return f"""=== BASIC STATISTICS ===
Words: {len(words)}
Sentences: {len(sentences)}
Characters: {len(text)}
Average word length: {sum(len(word) for word in words) / len(words) if words else 0:.1f}
Average sentence length: {len(words) / len(sentences) if sentences else 0:.1f} words"""

    def _identify_language(self, text: str) -> str:
        """Identify the language of the text."""
        words = text.lower().split()
        scores = {}

        for lang, patterns in self.language_patterns.items():
            score = 0
            for word in words[:50]:  # Check first 50 words
                if word in patterns["common_words"]:
                    score += 1
            scores[lang] = score

        best_lang = max(scores, key=scores.get)
        confidence = scores[best_lang] / max(scores.values()) if scores else 0

        return f"=== LANGUAGE IDENTIFICATION ===\nDetected: {best_lang.upper()}\nConfidence: {confidence:.2f}"

    def _analyze_sentence_structure(self, text: str) -> str:
        """Analyze sentence structure."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return "=== SENTENCE STRUCTURE ===\nNo sentences detected"

        lengths = [len(s.split()) for s in sentences]

        return f"""=== SENTENCE STRUCTURE ===
Total sentences: {len(sentences)}
Short sentences (1-10 words): {sum(1 for l in lengths if l <= 10)}
Medium sentences (11-20 words): {sum(1 for l in lengths if 11 <= l <= 20)}
Long sentences (21+ words): {sum(1 for l in lengths if l > 20)}
Average sentence length: {sum(lengths) / len(lengths):.1f} words"""

    def _extract_stylistic_features(self, text: str) -> str:
        """Extract stylistic features."""
        words = text.split()
        if not words:
            return "=== STYLISTIC FEATURES ===\nNo words to analyze"

        # Calculate various metrics
        unique_words = len(set(w.lower() for w in words))
        lexical_diversity = unique_words / len(words)

        # Count punctuation
        punctuation_count = len(re.findall(r'[.!?;:,()"\']', text))

        # Count uppercase words (potential proper nouns or emphasis)
        uppercase_words = sum(1 for word in words if word.isupper() and len(word) > 1)

        return f"""=== STYLISTIC FEATURES ===
Lexical diversity: {lexical_diversity:.3f}
Unique words: {unique_words}
Punctuation marks: {punctuation_count}
Uppercase words: {uppercase_words}"""


class TranslationTool(BaseTool):
    """Tool for text translation and linguistic conversion."""

    name: str = "translation"
    description: str = """
    Translate text between languages and analyze translation challenges.
    Use this tool to:
    - Translate text between supported languages
    - Identify translation difficulties
    - Analyze cultural linguistic nuances
    - Suggest translation alternatives
    """

    def __init__(self):
        super().__init__()
        self.translation_dict = self._load_translation_dict()

    def _load_translation_dict(self) -> Dict[str, Dict[str, str]]:
        """Load basic translation dictionary."""
        return {
            "greetings": {
                "english": "hello",
                "spanish": "hola",
                "french": "bonjour",
                "german": "hallo",
            },
            "goodbye": {
                "english": "goodbye",
                "spanish": "adiós",
                "french": "au revoir",
                "german": "auf wiedersehen",
            },
        }

    def _run(self, query: str) -> str:
        """
        Perform translation or translation analysis.

        Args:
            query: Translation query

        Returns:
            Translation results
        """
        try:
            # Parse query to determine translation request
            if "translate" in query.lower():
                return self._perform_translation(query)
            elif "challenge" in query.lower() or "difficulty" in query.lower():
                return self._analyze_translation_challenges(query)
            else:
                return self._general_translation_analysis(query)
        except Exception as e:
            logger.error(f"Error in translation: {e}")
            return f"Error performing translation: {str(e)}"

    def _perform_translation(self, query: str) -> str:
        """Perform basic translation."""
        # Simple translation logic (would use actual translation service)
        return "=== TRANSLATION ===\nTranslation service would process the text here.\nNote: This is a placeholder for actual translation functionality."

    def _analyze_translation_challenges(self, query: str) -> str:
        """Analyze translation challenges."""
        challenges = [
            "Idiomatic expressions that don't translate literally",
            "Cultural references that may not exist in target language",
            "Word order differences between languages",
            "Grammatical structures unique to source or target language",
            "Register and formality levels",
            "Pronunciation and phonetic challenges",
        ]

        return "=== TRANSLATION CHALLENGES ===\n" + "\n".join(
            f"• {challenge}" for challenge in challenges
        )

    def _general_translation_analysis(self, query: str) -> str:
        """Perform general translation analysis."""
        return f"=== TRANSLATION ANALYSIS ===\nAnalyzing translation aspects of: {query}\nWould involve linguistic comparison and cultural context analysis."


class LinguisticPatternTool(BaseTool):
    """Tool for identifying linguistic patterns and structures."""

    name: str = "linguistic_patterns"
    description: str = """
    Identify and analyze linguistic patterns in text.
    Use this tool to:
    - Find recurring linguistic structures
    - Analyze phonological patterns
    - Identify morphological patterns
    - Detect syntactic regularities
    """

    def __init__(self):
        super().__init__()

    def _run(self, text: str) -> str:
        """
        Analyze linguistic patterns in text.

        Args:
            text: Text to analyze for patterns

        Returns:
            Pattern analysis results
        """
        try:
            patterns = []

            # Phonological patterns
            patterns.append(self._analyze_phonological_patterns(text))

            # Morphological patterns
            patterns.append(self._analyze_morphological_patterns(text))

            # Syntactic patterns
            patterns.append(self._analyze_syntactic_patterns(text))

            return "\n".join(patterns)

        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            return f"Error analyzing patterns: {str(e)}"

    def _analyze_phonological_patterns(self, text: str) -> str:
        """Analyze phonological patterns."""
        vowels = "aeiouAEIOU"
        consonants = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"

        vowel_count = sum(1 for char in text if char in vowels)
        consonant_count = sum(1 for char in text if char in consonants)

        return f"""=== PHONOLOGICAL PATTERNS ===
Vowels: {vowel_count}
Consonants: {consonant_count}
Vowel-Consonant ratio: {vowel_count / consonant_count if consonant_count > 0 else 0:.2f}"""

    def _analyze_morphological_patterns(self, text: str) -> str:
        """Analyze morphological patterns."""
        words = text.split()
        if not words:
            return "=== MORPHOLOGICAL PATTERNS ===\nNo words to analyze"

        # Simple morphological analysis
        prefixes = ["un", "in", "dis", "re", "pre", "post"]
        suffixes = ["ing", "ed", "er", "est", "ly", "ness", "ment"]

        prefix_count = sum(
            1 for word in words for prefix in prefixes if word.startswith(prefix)
        )
        suffix_count = sum(
            1 for word in words for suffix in suffixes if word.endswith(suffix)
        )

        return f"""=== MORPHOLOGICAL PATTERNS ===
Words with common prefixes: {prefix_count}
Words with common suffixes: {suffix_count}
Average word length: {sum(len(word) for word in words) / len(words):.1f}"""

    def _analyze_syntactic_patterns(self, text: str) -> str:
        """Analyze syntactic patterns."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return "=== SYNTACTIC PATTERNS ===\nNo sentences to analyze"

        # Simple syntactic analysis
        question_count = text.count("?")
        exclamation_count = text.count("!")

        return f"""=== SYNTACTIC PATTERNS ===
Total sentences: {len(sentences)}
Questions: {question_count}
Exclamations: {exclamation_count}
Statements: {len(sentences) - question_count - exclamation_count}"""


class LinguistAgent(BaseSpecialistAgent):
    """
    Linguist Agent

    Specializes in language processing, analysis, and linguistic insights.
    Handles text analysis, translation, and pattern recognition.
    """

    def __init__(self, llm: BaseLLM, **kwargs):
        # Create specialized tools
        tools = [
            TextAnalysisTool(),
            TranslationTool(),
            LinguisticPatternTool(),
        ]

        super().__init__(name="LinguistAgent", llm=llm, tools=tools, **kwargs)

        # Agent-specific attributes
        self.analysis_history = []
        self.language_profiles = {}
        self.translation_cache = {}

    def _create_agent_executor(self) -> AgentExecutor:
        """Create the LangChain agent executor for linguistic analysis."""

        template = """
        You are the Linguist Agent in the Terra Constellata system.
        Your expertise is in language processing, analysis, and linguistic insights.

        You have access to:
        1. Text analysis tools for linguistic structure and patterns
        2. Translation tools for cross-language communication
        3. Linguistic pattern recognition for deeper language understanding

        When analyzing language:
        - Consider multiple linguistic levels (phonological, morphological, syntactic, semantic)
        - Identify language-specific features and challenges
        - Recognize cultural and contextual influences on language
        - Provide insights that enhance communication and understanding

        Current task: {input}

        Available tools: {tools}

        Chat history: {chat_history}

        Think step by step, then provide your linguistic analysis:
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
        Process a linguistic analysis task.

        Args:
            task: Analysis task description
            **kwargs: Additional parameters

        Returns:
            Analysis results
        """
        try:
            logger.info(f"Linguist Agent processing task: {task}")

            # Execute the analysis
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.agent_executor.run, task
            )

            # Store in analysis history
            self.analysis_history.append(
                {
                    "task": task,
                    "result": result,
                    "timestamp": datetime.utcnow(),
                    "kwargs": kwargs,
                }
            )

            # Cache language profiles
            self._cache_language_profile(task, result)

            return result

        except Exception as e:
            logger.error(f"Error processing task: {e}")
            return f"Analysis failed: {str(e)}"

    def _cache_language_profile(self, task: str, result: str):
        """Cache language profiles from analysis."""
        # Extract language information from results
        if "LANGUAGE IDENTIFICATION" in result:
            # Simple caching of language detection results
            self.language_profiles[task[:50]] = result

    async def analyze_text(
        self, text: str, analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Analyze text with specified analysis type.

        Args:
            text: Text to analyze
            analysis_type: Type of analysis to perform

        Returns:
            Analysis results
        """
        task = f"Analyze text: {text[:100]}... (type: {analysis_type})"

        # Use text analysis tool
        analysis_tool = self.tools[0]  # TextAnalysisTool
        result = analysis_tool._run(text)

        return {
            "text_preview": text[:200],
            "analysis_type": analysis_type,
            "results": result,
            "timestamp": datetime.utcnow(),
        }

    async def translate_text(
        self, text: str, source_lang: str, target_lang: str
    ) -> Dict[str, Any]:
        """
        Translate text between languages.

        Args:
            text: Text to translate
            source_lang: Source language
            target_lang: Target language

        Returns:
            Translation results
        """
        task = f"Translate from {source_lang} to {target_lang}: {text[:100]}..."

        # Use translation tool
        translation_tool = self.tools[1]  # TranslationTool
        result = translation_tool._run(
            f"translate {text} from {source_lang} to {target_lang}"
        )

        return {
            "original_text": text,
            "source_language": source_lang,
            "target_language": target_lang,
            "translation": result,
            "timestamp": datetime.utcnow(),
        }

    async def _autonomous_loop(self):
        """
        Autonomous operation loop for Linguist Agent.

        Performs continuous linguistic monitoring and analysis.
        """
        while self.is_active:
            try:
                # Perform periodic linguistic analysis
                await self._perform_periodic_analysis()

                # Monitor for new text to analyze
                await self._check_for_new_text()

                # Update language profiles
                self._update_language_profiles()

                # Wait before next cycle
                await asyncio.sleep(300)  # 5 minutes

            except Exception as e:
                logger.error(f"Error in autonomous loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

    async def _perform_periodic_analysis(self):
        """Perform periodic linguistic analysis tasks."""
        # Example: Analyze recent communications
        analysis_result = await self.analyze_text(
            "Sample text for periodic analysis", "linguistic"
        )

        # Share insights if significant patterns found
        if "pattern" in analysis_result.get("results", "").lower():
            await self._share_linguistic_insights(analysis_result)

    async def _check_for_new_text(self):
        """Check for new text that needs linguistic analysis."""
        # This would check for new text entries in the system
        pass

    def _update_language_profiles(self):
        """Update language profiles based on recent analyses."""
        # Update language detection and analysis patterns
        pass

    async def _share_linguistic_insights(self, insights: Dict[str, Any]):
        """Share linguistic insights with other agents."""
        logger.info(f"Sharing linguistic insights: {insights}")

    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get the history of linguistic analyses performed."""
        return self.analysis_history

    def get_language_profiles(self) -> Dict[str, Any]:
        """Get cached language profiles."""
        return self.language_profiles
