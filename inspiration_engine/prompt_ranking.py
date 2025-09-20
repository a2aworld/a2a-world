"""
Prompt Ranking System for Creative Potential

This module ranks creative prompts and ideas based on their novelty scores
and creative potential using the Inspiration Engine's algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from .algorithms import NoveltyDetector, NoveltyScore

logger = logging.getLogger(__name__)


@dataclass
class CreativePrompt:
    """Represents a creative prompt with its metadata"""

    id: str
    content: str
    domain: str
    context: Dict[str, Any]
    novelty_scores: Dict[str, NoveltyScore]
    combined_score: float
    creative_potential: float
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class PromptRanking:
    """Container for prompt ranking results"""

    ranked_prompts: List[CreativePrompt]
    ranking_criteria: Dict[str, float]
    top_prompt: Optional[CreativePrompt]
    diversity_score: float
    timestamp: datetime


class PromptRanker:
    """
    Ranks creative prompts based on novelty and creative potential
    """

    def __init__(self, novelty_detector: Optional[NoveltyDetector] = None):
        """
        Initialize prompt ranker

        Args:
            novelty_detector: NoveltyDetector instance (creates new one if None)
        """
        self.novelty_detector = novelty_detector or NoveltyDetector()

        # Creative potential weights
        self.creative_weights = {
            "novelty": 0.4,
            "diversity": 0.3,
            "feasibility": 0.2,
            "impact": 0.1,
        }

        # Domain-specific creativity factors
        self.domain_factors = {
            "mythology": {"novelty_weight": 1.2, "creativity_multiplier": 1.1},
            "geography": {"novelty_weight": 1.0, "creativity_multiplier": 1.0},
            "cultural": {"novelty_weight": 1.1, "creativity_multiplier": 1.2},
            "narrative": {"novelty_weight": 0.9, "creativity_multiplier": 1.3},
            "visual": {"novelty_weight": 1.0, "creativity_multiplier": 1.1},
        }

    def rank_prompts(
        self,
        prompts: List[Dict[str, Any]],
        context_data: Dict[str, Any] = None,
        ranking_criteria: Dict[str, float] = None,
    ) -> PromptRanking:
        """
        Rank creative prompts based on novelty and creative potential

        Args:
            prompts: List of prompt dictionaries with 'content', 'domain', etc.
            context_data: Additional context data for novelty detection
            ranking_criteria: Custom ranking criteria weights

        Returns:
            PromptRanking with ranked prompts and metadata
        """
        if ranking_criteria:
            self.creative_weights.update(ranking_criteria)

        creative_prompts = []

        for prompt_data in prompts:
            try:
                prompt = self._create_creative_prompt(prompt_data, context_data)
                creative_prompts.append(prompt)
            except Exception as e:
                logger.error(
                    f"Failed to process prompt {prompt_data.get('id', 'unknown')}: {e}"
                )

        # Sort by creative potential
        creative_prompts.sort(key=lambda x: x.creative_potential, reverse=True)

        # Calculate diversity score
        diversity_score = self._calculate_diversity_score(creative_prompts)

        ranking = PromptRanking(
            ranked_prompts=creative_prompts,
            ranking_criteria=self.creative_weights.copy(),
            top_prompt=creative_prompts[0] if creative_prompts else None,
            diversity_score=diversity_score,
            timestamp=datetime.utcnow(),
        )

        logger.info(
            f"Ranked {len(creative_prompts)} prompts with diversity score {diversity_score:.3f}"
        )
        return ranking

    def _create_creative_prompt(
        self, prompt_data: Dict[str, Any], context_data: Dict[str, Any] = None
    ) -> CreativePrompt:
        """
        Create a CreativePrompt object with novelty analysis

        Args:
            prompt_data: Raw prompt data
            context_data: Context for novelty detection

        Returns:
            CreativePrompt with computed scores
        """
        prompt_id = prompt_data.get(
            "id", f"prompt_{hash(prompt_data.get('content', ''))}"
        )
        content = prompt_data.get("content", "")
        domain = prompt_data.get("domain", "general")

        # Prepare data for novelty detection
        novelty_input = self._prepare_novelty_input(content, domain, context_data)

        # Detect novelty using all algorithms
        novelty_scores = self.novelty_detector.detect_novelty(
            data=novelty_input,
            context={"domain": domain, "content_type": "creative_prompt"},
        )

        # Calculate combined novelty score
        combined_score = self.novelty_detector.calculate_combined_score(novelty_scores)

        # Calculate creative potential
        creative_potential = self._calculate_creative_potential(
            combined_score, novelty_scores, domain, prompt_data
        )

        return CreativePrompt(
            id=prompt_id,
            content=content,
            domain=domain,
            context=prompt_data.get("context", {}),
            novelty_scores=novelty_scores,
            combined_score=combined_score,
            creative_potential=creative_potential,
            timestamp=datetime.utcnow(),
            metadata=prompt_data.get("metadata", {}),
        )

    def _prepare_novelty_input(
        self, content: str, domain: str, context_data: Dict[str, Any] = None
    ) -> Any:
        """
        Prepare input data for novelty detection based on prompt content

        Args:
            content: Prompt content
            domain: Creative domain
            context_data: Additional context

        Returns:
            Prepared data for novelty detection
        """
        # Extract features from content
        features = {
            "word_count": len(content.split()),
            "sentence_count": len(content.split(".")),
            "unique_words": len(set(content.lower().split())),
            "domain": domain,
            "content_length": len(content),
        }

        # Add domain-specific features
        if domain == "mythology":
            features.update(self._extract_mythology_features(content))
        elif domain == "geography":
            features.update(self._extract_geography_features(content))
        elif domain == "narrative":
            features.update(self._extract_narrative_features(content))

        # Include context data if available
        if context_data:
            features["context_similarity"] = self._calculate_context_similarity(
                content, context_data
            )

        return features

    def _extract_mythology_features(self, content: str) -> Dict[str, Any]:
        """Extract mythology-specific features"""
        mythology_keywords = [
            "god",
            "goddess",
            "myth",
            "legend",
            "hero",
            "quest",
            "oracle",
            "temple",
            "ritual",
            "divine",
            "mortal",
            "immortal",
            "prophecy",
        ]

        content_lower = content.lower()
        keyword_count = sum(
            1 for keyword in mythology_keywords if keyword in content_lower
        )

        return {
            "mythology_keywords": keyword_count,
            "archetype_density": keyword_count / len(content.split())
            if content.split()
            else 0,
        }

    def _extract_geography_features(self, content: str) -> Dict[str, Any]:
        """Extract geography-specific features"""
        geography_keywords = [
            "mountain",
            "river",
            "sea",
            "ocean",
            "valley",
            "desert",
            "forest",
            "city",
            "village",
            "land",
            "territory",
            "boundary",
            "journey",
        ]

        content_lower = content.lower()
        keyword_count = sum(
            1 for keyword in geography_keywords if keyword in content_lower
        )

        return {
            "geography_keywords": keyword_count,
            "spatial_density": keyword_count / len(content.split())
            if content.split()
            else 0,
        }

    def _extract_narrative_features(self, content: str) -> Dict[str, Any]:
        """Extract narrative-specific features"""
        narrative_keywords = [
            "story",
            "tale",
            "narrative",
            "plot",
            "character",
            "conflict",
            "resolution",
            "beginning",
            "middle",
            "end",
            "climax",
            "twist",
        ]

        content_lower = content.lower()
        keyword_count = sum(
            1 for keyword in narrative_keywords if keyword in content_lower
        )

        return {
            "narrative_keywords": keyword_count,
            "structure_density": keyword_count / len(content.split())
            if content.split()
            else 0,
        }

    def _calculate_context_similarity(
        self, content: str, context_data: Dict[str, Any]
    ) -> float:
        """Calculate similarity between prompt content and context data"""
        if not context_data:
            return 0.0

        # Simple keyword overlap similarity
        content_words = set(content.lower().split())
        context_words = set()

        for key, value in context_data.items():
            if isinstance(value, str):
                context_words.update(value.lower().split())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        context_words.update(item.lower().split())

        if not content_words or not context_words:
            return 0.0

        overlap = len(content_words.intersection(context_words))
        return overlap / len(content_words.union(context_words))

    def _calculate_creative_potential(
        self,
        novelty_score: float,
        novelty_scores: Dict[str, NoveltyScore],
        domain: str,
        prompt_data: Dict[str, Any],
    ) -> float:
        """
        Calculate overall creative potential score

        Args:
            novelty_score: Combined novelty score
            novelty_scores: Individual algorithm scores
            domain: Creative domain
            prompt_data: Original prompt data

        Returns:
            Creative potential score (0-1)
        """
        # Base novelty contribution
        domain_factor = self.domain_factors.get(
            domain, {"novelty_weight": 1.0, "creativity_multiplier": 1.0}
        )
        weighted_novelty = novelty_score * domain_factor["novelty_weight"]

        # Diversity contribution (based on algorithm agreement)
        scores_list = [score.score for score in novelty_scores.values()]
        diversity = np.std(scores_list) if scores_list else 0
        diversity_score = min(diversity, 1.0)  # Cap at 1.0

        # Feasibility contribution (based on prompt structure)
        feasibility = self._assess_feasibility(prompt_data)

        # Impact contribution (based on scope and reach)
        impact = self._assess_impact(prompt_data)

        # Combine components
        creative_potential = (
            self.creative_weights["novelty"] * weighted_novelty
            + self.creative_weights["diversity"] * diversity_score
            + self.creative_weights["feasibility"] * feasibility
            + self.creative_weights["impact"] * impact
        )

        # Apply domain-specific creativity multiplier
        creative_potential *= domain_factor["creativity_multiplier"]

        return min(creative_potential, 1.0)  # Cap at 1.0

    def _assess_feasibility(self, prompt_data: Dict[str, Any]) -> float:
        """Assess how feasible the prompt is to execute"""
        content = prompt_data.get("content", "")
        constraints = prompt_data.get("constraints", [])

        # Length-based feasibility
        word_count = len(content.split())
        if word_count < 10:
            return 0.3  # Too short
        elif word_count > 500:
            return 0.6  # Complex but feasible
        else:
            return 0.8  # Optimal length

    def _assess_impact(self, prompt_data: Dict[str, Any]) -> float:
        """Assess potential impact of the creative output"""
        domain = prompt_data.get("domain", "general")
        scope = prompt_data.get("scope", "individual")

        # Domain impact factors
        domain_impacts = {
            "mythology": 0.9,  # High cultural impact
            "geography": 0.7,  # Educational impact
            "cultural": 0.8,  # Social impact
            "narrative": 0.6,  # Entertainment impact
            "visual": 0.7,  # Aesthetic impact
        }

        # Scope impact factors
        scope_impacts = {"individual": 0.4, "community": 0.7, "global": 0.9}

        domain_impact = domain_impacts.get(domain, 0.5)
        scope_impact = scope_impacts.get(scope, 0.5)

        return (domain_impact + scope_impact) / 2

    def _calculate_diversity_score(self, prompts: List[CreativePrompt]) -> float:
        """Calculate diversity score of the prompt set"""
        if len(prompts) < 2:
            return 0.0

        # Domain diversity
        domains = [p.domain for p in prompts]
        domain_diversity = len(set(domains)) / len(domains)

        # Novelty diversity
        novelty_scores = [p.combined_score for p in prompts]
        novelty_diversity = np.std(novelty_scores) if novelty_scores else 0

        # Content diversity (simple approximation)
        content_lengths = [len(p.content.split()) for p in prompts]
        length_diversity = (
            np.std(content_lengths) / np.mean(content_lengths) if content_lengths else 0
        )

        # Combine diversity measures
        overall_diversity = (
            0.4 * domain_diversity
            + 0.4 * min(novelty_diversity, 1.0)
            + 0.2 * min(length_diversity, 1.0)
        )

        return overall_diversity

    def get_top_prompts(
        self, ranking: PromptRanking, top_n: int = 5, min_score: float = 0.0
    ) -> List[CreativePrompt]:
        """
        Get top N prompts from ranking results

        Args:
            ranking: PromptRanking object
            top_n: Number of top prompts to return
            min_score: Minimum creative potential score

        Returns:
            List of top CreativePrompt objects
        """
        filtered_prompts = [
            p for p in ranking.ranked_prompts if p.creative_potential >= min_score
        ]

        return filtered_prompts[:top_n]

    def generate_prompt_variations(
        self, base_prompt: CreativePrompt, variation_count: int = 3
    ) -> List[CreativePrompt]:
        """
        Generate variations of a base prompt with different creative angles

        Args:
            base_prompt: Base CreativePrompt to vary
            variation_count: Number of variations to generate

        Returns:
            List of CreativePrompt variations
        """
        variations = []

        # Define variation strategies
        strategies = [
            self._apply_mythological_enhancement,
            self._apply_geographical_expansion,
            self._apply_narrative_deepening,
            self._apply_interdisciplinary_fusion,
        ]

        for i in range(min(variation_count, len(strategies))):
            try:
                variation = strategies[i](base_prompt, i)
                variations.append(variation)
            except Exception as e:
                logger.error(f"Failed to generate variation {i}: {e}")

        return variations

    def _apply_mythological_enhancement(
        self, base_prompt: CreativePrompt, index: int
    ) -> CreativePrompt:
        """Apply mythological enhancement to prompt"""
        enhanced_content = (
            f"Explore the mythological dimensions of: {base_prompt.content}"
        )
        return CreativePrompt(
            id=f"{base_prompt.id}_myth_{index}",
            content=enhanced_content,
            domain="mythology",
            context=base_prompt.context,
            novelty_scores={},  # Would be calculated
            combined_score=0.0,  # Would be calculated
            creative_potential=0.0,  # Would be calculated
            timestamp=datetime.utcnow(),
            metadata={
                "variation_type": "mythological_enhancement",
                "base_prompt": base_prompt.id,
            },
        )

    def _apply_geographical_expansion(
        self, base_prompt: CreativePrompt, index: int
    ) -> CreativePrompt:
        """Apply geographical expansion to prompt"""
        enhanced_content = f"Consider the geographical context and spatial relationships in: {base_prompt.content}"
        return CreativePrompt(
            id=f"{base_prompt.id}_geo_{index}",
            content=enhanced_content,
            domain="geography",
            context=base_prompt.context,
            novelty_scores={},
            combined_score=0.0,
            creative_potential=0.0,
            timestamp=datetime.utcnow(),
            metadata={
                "variation_type": "geographical_expansion",
                "base_prompt": base_prompt.id,
            },
        )

    def _apply_narrative_deepening(
        self, base_prompt: CreativePrompt, index: int
    ) -> CreativePrompt:
        """Apply narrative deepening to prompt"""
        enhanced_content = (
            f"Develop a compelling narrative around: {base_prompt.content}"
        )
        return CreativePrompt(
            id=f"{base_prompt.id}_narr_{index}",
            content=enhanced_content,
            domain="narrative",
            context=base_prompt.context,
            novelty_scores={},
            combined_score=0.0,
            creative_potential=0.0,
            timestamp=datetime.utcnow(),
            metadata={
                "variation_type": "narrative_deepening",
                "base_prompt": base_prompt.id,
            },
        )

    def _apply_interdisciplinary_fusion(
        self, base_prompt: CreativePrompt, index: int
    ) -> CreativePrompt:
        """Apply interdisciplinary fusion to prompt"""
        enhanced_content = (
            f"Integrate multiple disciplines to explore: {base_prompt.content}"
        )
        return CreativePrompt(
            id=f"{base_prompt.id}_fusion_{index}",
            content=enhanced_content,
            domain="cultural",
            context=base_prompt.context,
            novelty_scores={},
            combined_score=0.0,
            creative_potential=0.0,
            timestamp=datetime.utcnow(),
            metadata={
                "variation_type": "interdisciplinary_fusion",
                "base_prompt": base_prompt.id,
            },
        )
