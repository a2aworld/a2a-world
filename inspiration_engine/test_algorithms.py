"""
Unit tests for Inspiration Engine novelty detection algorithms
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime

from .algorithms import (
    NoveltyDetector,
    RPADAlgorithm,
    PeculiarityAlgorithm,
    BeliefChangeAlgorithm,
    NoveltyScore,
)


class TestRPADAlgorithm(unittest.TestCase):
    """Test cases for RPAD (Rare Pattern Anomaly Detection) algorithm"""

    def setUp(self):
        self.algorithm = RPADAlgorithm()

    def test_get_algorithm_name(self):
        """Test algorithm name retrieval"""
        self.assertEqual(self.algorithm.get_algorithm_name(), "RPAD")

    def test_calculate_novelty_empty_data(self):
        """Test novelty calculation with empty data"""
        result = self.algorithm.calculate_novelty([])
        self.assertIsInstance(result, NoveltyScore)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.algorithm, "RPAD")

    def test_calculate_novelty_list_data(self):
        """Test novelty calculation with list data"""
        data = ["apple", "banana", "apple", "cherry", "apple", "banana", "date"]
        result = self.algorithm.calculate_novelty(data)

        self.assertIsInstance(result, NoveltyScore)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
        self.assertEqual(result.algorithm, "RPAD")
        self.assertIn("total_patterns", result.metadata)

    def test_calculate_novelty_dict_data(self):
        """Test novelty calculation with dictionary data"""
        data = {"name": "test", "type": "entity", "value": 42}
        result = self.algorithm.calculate_novelty(data)

        self.assertIsInstance(result, NoveltyScore)
        self.assertEqual(result.algorithm, "RPAD")

    def test_calculate_novelty_dataframe(self):
        """Test novelty calculation with DataFrame"""
        df = pd.DataFrame(
            {
                "entity": ["god", "hero", "god", "monster"],
                "type": ["divine", "mortal", "divine", "monster"],
            }
        )
        result = self.algorithm.calculate_novelty(df)

        self.assertIsInstance(result, NoveltyScore)
        self.assertEqual(result.algorithm, "RPAD")


class TestPeculiarityAlgorithm(unittest.TestCase):
    """Test cases for Peculiarity/J-Measure algorithm"""

    def setUp(self):
        self.algorithm = PeculiarityAlgorithm()

    def test_get_algorithm_name(self):
        """Test algorithm name retrieval"""
        self.assertEqual(self.algorithm.get_algorithm_name(), "Peculiarity_J_Measure")

    def test_calculate_novelty_without_context(self):
        """Test novelty calculation without required context"""
        data = ["test"]
        result = self.algorithm.calculate_novelty(data)

        self.assertIsInstance(result, NoveltyScore)
        self.assertEqual(result.score, 0.0)
        self.assertIn("error", result.metadata)

    def test_calculate_novelty_with_context(self):
        """Test novelty calculation with proper context"""
        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4],
                "feature2": ["a", "b", "a", "c"],
                "class": ["positive", "negative", "positive", "negative"],
            }
        )

        context = {
            "class_label": "positive",
            "features": {"feature1": 3, "feature2": "a"},
        }

        result = self.algorithm.calculate_novelty(data, context)

        self.assertIsInstance(result, NoveltyScore)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
        self.assertEqual(result.algorithm, "Peculiarity_J_Measure")


class TestBeliefChangeAlgorithm(unittest.TestCase):
    """Test cases for Belief-Change Measure algorithm"""

    def setUp(self):
        self.algorithm = BeliefChangeAlgorithm()

    def test_get_algorithm_name(self):
        """Test algorithm name retrieval"""
        self.assertEqual(self.algorithm.get_algorithm_name(), "Belief_Change_Measure")

    def test_calculate_novelty_without_context(self):
        """Test novelty calculation without required context"""
        data = ["test"]
        result = self.algorithm.calculate_novelty(data)

        self.assertIsInstance(result, NoveltyScore)
        self.assertEqual(result.score, 0.0)
        self.assertIn("error", result.metadata)

    def test_calculate_novelty_with_context(self):
        """Test novelty calculation with proper context"""
        data = ["new_evidence"]

        context = {
            "prior_beliefs": {"hypothesis1": 0.6, "hypothesis2": 0.4},
            "likelihoods": {"hypothesis1": 0.8, "hypothesis2": 0.2},
        }

        result = self.algorithm.calculate_novelty(data, context)

        self.assertIsInstance(result, NoveltyScore)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
        self.assertEqual(result.algorithm, "Belief_Change_Measure")
        self.assertIn("kl_divergence", result.metadata)


class TestNoveltyDetector(unittest.TestCase):
    """Test cases for the main NoveltyDetector class"""

    def setUp(self):
        self.detector = NoveltyDetector()

    def test_detect_novelty_all_algorithms(self):
        """Test novelty detection using all algorithms"""
        data = ["rare_pattern", "common", "common", "rare_pattern"]
        results = self.detector.detect_novelty(data)

        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)

        for algo_name, score in results.items():
            self.assertIsInstance(score, NoveltyScore)
            self.assertEqual(score.algorithm, algo_name)

    def test_detect_novelty_specific_algorithms(self):
        """Test novelty detection using specific algorithms"""
        data = ["test"]
        algorithms = ["rpad", "peculiarity"]
        results = self.detector.detect_novelty(data, algorithms=algorithms)

        self.assertEqual(len(results), 2)
        self.assertIn("rpad", results)
        self.assertIn("peculiarity", results)
        self.assertNotIn("belief_change", results)

    def test_calculate_combined_score(self):
        """Test combined score calculation"""
        # Create mock scores
        scores = {
            "rpad": NoveltyScore(0.8, "rpad", 0.9, {}),
            "peculiarity": NoveltyScore(0.6, "peculiarity", 0.8, {}),
            "belief_change": NoveltyScore(0.4, "belief_change", 0.7, {}),
        }

        combined = self.detector.calculate_combined_score(scores)
        self.assertGreaterEqual(combined, 0.0)
        self.assertLessEqual(combined, 1.0)

    def test_calculate_combined_score_empty(self):
        """Test combined score calculation with empty scores"""
        combined = self.detector.calculate_combined_score({})
        self.assertEqual(combined, 0.0)

    def test_calculate_combined_score_with_weights(self):
        """Test combined score calculation with custom weights"""
        scores = {
            "rpad": NoveltyScore(0.8, "rpad", 0.9, {}),
            "peculiarity": NoveltyScore(0.6, "peculiarity", 0.8, {}),
        }

        weights = {"rpad": 0.7, "peculiarity": 0.3}
        combined = self.detector.calculate_combined_score(scores, weights)

        # Should be closer to RPAD score due to higher weight
        expected = 0.8 * 0.7 * 0.9 + 0.6 * 0.3 * 0.8
        self.assertAlmostEqual(combined, expected, places=3)


class TestNoveltyScore(unittest.TestCase):
    """Test cases for NoveltyScore dataclass"""

    def test_novelty_score_creation(self):
        """Test NoveltyScore object creation"""
        score = NoveltyScore(
            score=0.75,
            algorithm="test_algorithm",
            confidence=0.85,
            metadata={"test": "data"},
        )

        self.assertEqual(score.score, 0.75)
        self.assertEqual(score.algorithm, "test_algorithm")
        self.assertEqual(score.confidence, 0.85)
        self.assertEqual(score.metadata, {"test": "data"})

    def test_novelty_score_default_values(self):
        """Test NoveltyScore with default values"""
        score = NoveltyScore(0.5, "test")

        self.assertEqual(score.score, 0.5)
        self.assertEqual(score.algorithm, "test")
        self.assertEqual(score.confidence, 0.0)  # Default confidence
        self.assertEqual(score.metadata, {})  # Default metadata


if __name__ == "__main__":
    unittest.main()
