"""
Novelty Detection Algorithms

This module implements statistical methods for quantifying "interestingness" including:
- RPAD (Rare Pattern Anomaly Detection)
- Peculiarity/J-Measure
- Belief-Change Measure
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict
import math
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class NoveltyScore:
    """Container for novelty detection results"""

    score: float
    algorithm: str
    confidence: float
    metadata: Dict[str, Any]


class NoveltyAlgorithm(ABC):
    """Abstract base class for novelty detection algorithms"""

    @abstractmethod
    def calculate_novelty(
        self, data: Any, context: Dict[str, Any] = None
    ) -> NoveltyScore:
        """Calculate novelty score for given data"""
        pass

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Return the name of the algorithm"""
        pass


class RPADAlgorithm(NoveltyAlgorithm):
    """
    Rare Pattern Anomaly Detection (RPAD)

    Detects rare patterns by analyzing frequency distributions and identifying
    statistically significant deviations from expected patterns.
    """

    def __init__(self, significance_threshold: float = 0.05, min_support: int = 5):
        self.significance_threshold = significance_threshold
        self.min_support = min_support

    def get_algorithm_name(self) -> str:
        return "RPAD"

    def calculate_novelty(
        self, data: Any, context: Dict[str, Any] = None
    ) -> NoveltyScore:
        """
        Calculate RPAD novelty score based on pattern rarity

        Args:
            data: Input data (list, dict, or pandas DataFrame)
            context: Additional context information

        Returns:
            NoveltyScore with RPAD-based novelty assessment
        """
        if isinstance(data, list):
            patterns = self._extract_patterns_from_list(data)
        elif isinstance(data, dict):
            patterns = self._extract_patterns_from_dict(data)
        elif isinstance(data, pd.DataFrame):
            patterns = self._extract_patterns_from_dataframe(data)
        else:
            patterns = self._extract_patterns_from_generic(data)

        # Calculate pattern frequencies
        pattern_counts = Counter(patterns)
        total_patterns = len(patterns)

        if total_patterns == 0:
            return NoveltyScore(
                0.0, self.get_algorithm_name(), 0.0, {"error": "No patterns found"}
            )

        # Calculate rarity scores
        rarity_scores = {}
        for pattern, count in pattern_counts.items():
            if count >= self.min_support:
                # Expected frequency under uniform distribution
                expected_freq = 1.0 / len(pattern_counts)
                observed_freq = count / total_patterns

                # Calculate statistical significance (chi-square like)
                if expected_freq > 0:
                    chi_square = ((observed_freq - expected_freq) ** 2) / expected_freq
                    rarity_scores[pattern] = chi_square

        # Find most novel pattern
        if rarity_scores:
            max_rarity = max(rarity_scores.values())
            # Normalize to 0-1 scale using exponential decay
            novelty_score = 1 - math.exp(-max_rarity)
        else:
            novelty_score = 0.0

        confidence = min(
            1.0, total_patterns / 100.0
        )  # Higher confidence with more data

        return NoveltyScore(
            novelty_score,
            self.get_algorithm_name(),
            confidence,
            {
                "total_patterns": total_patterns,
                "unique_patterns": len(pattern_counts),
                "max_rarity_score": max(rarity_scores.values()) if rarity_scores else 0,
                "patterns_above_threshold": len(
                    [s for s in rarity_scores.values() if s > 1.0]
                ),
            },
        )

    def _extract_patterns_from_list(self, data: List) -> List[str]:
        """Extract patterns from list data"""
        patterns = []
        for item in data:
            if isinstance(item, (str, int, float)):
                patterns.append(str(item))
            elif isinstance(item, dict):
                # Extract key-value patterns
                for key, value in item.items():
                    patterns.append(f"{key}:{value}")
            else:
                patterns.append(str(type(item).__name__))
        return patterns

    def _extract_patterns_from_dict(self, data: Dict) -> List[str]:
        """Extract patterns from dictionary data"""
        patterns = []
        for key, value in data.items():
            patterns.append(f"{key}:{type(value).__name__}")
            if isinstance(value, (list, tuple)):
                patterns.extend(
                    [f"{key}:{item}" for item in value[:5]]
                )  # Limit to first 5 items
        return patterns

    def _extract_patterns_from_dataframe(self, data: pd.DataFrame) -> List[str]:
        """Extract patterns from DataFrame"""
        patterns = []
        for col in data.columns:
            unique_values = data[col].dropna().unique()
            for val in unique_values[:10]:  # Limit to first 10 unique values per column
                patterns.append(f"{col}:{val}")
        return patterns

    def _extract_patterns_from_generic(self, data: Any) -> List[str]:
        """Extract patterns from generic data"""
        return [str(data)]


class PeculiarityAlgorithm(NoveltyAlgorithm):
    """
    Peculiarity/J-Measure Algorithm

    Measures the peculiarity of patterns based on their deviation from
    expected distributions using information theory.
    """

    def __init__(self, base: float = 2.0):
        self.base = base  # Base for logarithm (2 for bits, e for nats)

    def get_algorithm_name(self) -> str:
        return "Peculiarity_J_Measure"

    def calculate_novelty(
        self, data: Any, context: Dict[str, Any] = None
    ) -> NoveltyScore:
        """
        Calculate peculiarity using J-Measure (information-theoretic approach)

        J-Measure = -log(P(class|feature)) + log(P(class))

        Args:
            data: Input data with features and class labels
            context: Additional context (expected to contain 'class_label' and 'features')

        Returns:
            NoveltyScore with peculiarity assessment
        """
        if context is None or "class_label" not in context or "features" not in context:
            return NoveltyScore(
                0.0,
                self.get_algorithm_name(),
                0.0,
                {"error": "Missing required context: class_label and features"},
            )

        class_label = context["class_label"]
        features = context["features"]

        # Calculate prior probability of class
        if isinstance(data, pd.DataFrame):
            total_samples = len(data)
            class_count = len(
                data[data.iloc[:, -1] == class_label]
            )  # Assume last column is class
            prior_prob = class_count / total_samples if total_samples > 0 else 0
        else:
            # For other data types, assume uniform prior
            prior_prob = 0.5

        # Calculate conditional probabilities for features
        peculiarity_scores = []

        for feature_name, feature_value in features.items():
            # Calculate P(class|feature)
            conditional_prob = self._calculate_conditional_probability(
                data, feature_name, feature_value, class_label
            )

            if conditional_prob > 0 and prior_prob > 0:
                # J-Measure calculation
                j_measure = -math.log(conditional_prob, self.base) + math.log(
                    prior_prob, self.base
                )
                peculiarity_scores.append(j_measure)

        if peculiarity_scores:
            # Average peculiarity across all features
            avg_peculiarity = np.mean(peculiarity_scores)
            # Normalize to 0-1 scale
            novelty_score = 1 / (
                1 + math.exp(-avg_peculiarity)
            )  # Sigmoid normalization
        else:
            novelty_score = 0.0

        confidence = min(
            1.0, len(features) / 10.0
        )  # Higher confidence with more features

        return NoveltyScore(
            novelty_score,
            self.get_algorithm_name(),
            confidence,
            {
                "peculiarity_scores": peculiarity_scores,
                "average_peculiarity": np.mean(peculiarity_scores)
                if peculiarity_scores
                else 0,
                "prior_probability": prior_prob,
                "features_analyzed": len(features),
            },
        )

    def _calculate_conditional_probability(
        self, data: Any, feature_name: str, feature_value: Any, class_label: Any
    ) -> float:
        """Calculate P(class|feature)"""
        if isinstance(data, pd.DataFrame):
            if feature_name not in data.columns:
                return 0.0

            # Count samples with this feature value
            feature_matches = data[data[feature_name] == feature_value]
            if len(feature_matches) == 0:
                return 0.0

            # Count samples with both feature value and class label
            class_matches = feature_matches[feature_matches.iloc[:, -1] == class_label]
            return len(class_matches) / len(feature_matches)

        # For non-DataFrame data, return default
        return 0.5


class BeliefChangeAlgorithm(NoveltyAlgorithm):
    """
    Belief-Change Measure Algorithm

    Measures how much new information changes existing beliefs using
    Bayesian updating principles.
    """

    def __init__(self, prior_belief_strength: float = 0.5):
        self.prior_belief_strength = prior_belief_strength

    def get_algorithm_name(self) -> str:
        return "Belief_Change_Measure"

    def calculate_novelty(
        self, data: Any, context: Dict[str, Any] = None
    ) -> NoveltyScore:
        """
        Calculate belief change using Bayesian updating

        Measures the KL-divergence between prior and posterior beliefs

        Args:
            data: New evidence/data
            context: Context containing prior beliefs and likelihoods

        Returns:
            NoveltyScore with belief change assessment
        """
        if context is None:
            return NoveltyScore(
                0.0,
                self.get_algorithm_name(),
                0.0,
                {"error": "Missing context for belief change calculation"},
            )

        prior_beliefs = context.get("prior_beliefs", {})
        likelihoods = context.get("likelihoods", {})

        if not prior_beliefs or not likelihoods:
            return NoveltyScore(
                0.0,
                self.get_algorithm_name(),
                0.0,
                {"error": "Missing prior_beliefs or likelihoods in context"},
            )

        # Calculate posterior beliefs using Bayes' theorem
        posterior_beliefs = {}
        evidence_strength = 0

        for hypothesis, prior in prior_beliefs.items():
            likelihood = likelihoods.get(hypothesis, 0.1)  # Default likelihood
            evidence = likelihood * prior

            # Normalize by evidence (marginal likelihood)
            if evidence > 0:
                posterior = evidence / sum(
                    likelihoods.get(h, 0.1)
                    * prior_beliefs.get(h, self.prior_belief_strength)
                    for h in prior_beliefs.keys()
                )
                posterior_beliefs[hypothesis] = posterior
                evidence_strength += evidence

        if not posterior_beliefs:
            return NoveltyScore(
                0.0,
                self.get_algorithm_name(),
                0.0,
                {"error": "Could not calculate posterior beliefs"},
            )

        # Calculate KL-divergence between prior and posterior
        kl_divergence = 0
        for hypothesis in prior_beliefs.keys():
            prior = prior_beliefs.get(hypothesis, self.prior_belief_strength)
            posterior = posterior_beliefs.get(hypothesis, prior)

            if prior > 0 and posterior > 0:
                kl_divergence += posterior * math.log(posterior / prior, 2)

        # Normalize KL-divergence to 0-1 scale
        novelty_score = 1 - math.exp(-abs(kl_divergence))

        confidence = min(
            1.0, len(prior_beliefs) / 5.0
        )  # Higher confidence with more hypotheses

        return NoveltyScore(
            novelty_score,
            self.get_algorithm_name(),
            confidence,
            {
                "kl_divergence": kl_divergence,
                "prior_beliefs": prior_beliefs,
                "posterior_beliefs": posterior_beliefs,
                "evidence_strength": evidence_strength,
                "hypotheses_count": len(prior_beliefs),
            },
        )


class NoveltyDetector:
    """
    Main novelty detection orchestrator combining multiple algorithms
    """

    def __init__(self):
        self.algorithms = {
            "rpad": RPADAlgorithm(),
            "peculiarity": PeculiarityAlgorithm(),
            "belief_change": BeliefChangeAlgorithm(),
        }

    def detect_novelty(
        self, data: Any, context: Dict[str, Any] = None, algorithms: List[str] = None
    ) -> Dict[str, NoveltyScore]:
        """
        Detect novelty using specified algorithms

        Args:
            data: Input data to analyze
            context: Additional context for algorithms
            algorithms: List of algorithm names to use (default: all)

        Returns:
            Dictionary mapping algorithm names to NoveltyScore objects
        """
        if algorithms is None:
            algorithms = list(self.algorithms.keys())

        results = {}
        for algo_name in algorithms:
            if algo_name in self.algorithms:
                try:
                    score = self.algorithms[algo_name].calculate_novelty(data, context)
                    results[algo_name] = score
                except Exception as e:
                    results[algo_name] = NoveltyScore(
                        0.0, algo_name, 0.0, {"error": str(e)}
                    )

        return results

    def calculate_combined_score(
        self, scores: Dict[str, NoveltyScore], weights: Dict[str, float] = None
    ) -> float:
        """
        Calculate weighted combination of novelty scores

        Args:
            scores: Dictionary of NoveltyScore objects
            weights: Optional weights for each algorithm

        Returns:
            Combined novelty score (0-1)
        """
        if not scores:
            return 0.0

        if weights is None:
            # Equal weights by default
            weights = {name: 1.0 / len(scores) for name in scores.keys()}

        combined_score = 0.0
        total_weight = 0.0

        for algo_name, score in scores.items():
            weight = weights.get(algo_name, 1.0 / len(scores))
            combined_score += score.score * weight * score.confidence
            total_weight += weight

        return combined_score / total_weight if total_weight > 0 else 0.0
