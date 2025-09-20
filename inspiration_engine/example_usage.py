#!/usr/bin/env python3
"""
Example usage of the Inspiration Engine

This script demonstrates how to use the Inspiration Engine for novelty detection,
creative inspiration generation, and agent communication.
"""

import asyncio
import logging
import sys
import os

# Add the parent directory to the path to import from terra-constellata
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inspiration_engine import InspirationEngine
from inspiration_engine.algorithms import NoveltyDetector
from inspiration_engine.prompt_ranking import PromptRanker

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def basic_novelty_detection_example():
    """Demonstrate basic novelty detection"""
    print("\n=== Basic Novelty Detection Example ===")

    # Create a novelty detector
    detector = NoveltyDetector()

    # Sample data with some rare patterns
    data = [
        "apple",
        "banana",
        "apple",
        "cherry",
        "apple",
        "banana",
        "dragon_fruit",
        "elderberry",
        "dragon_fruit",
        "fig",
        "dragon_fruit",
        "grapefruit",
        "honeydew",
        "dragon_fruit",
    ]

    print(f"Analyzing data: {data}")

    # Detect novelty
    results = detector.detect_novelty(data)
    combined_score = detector.calculate_combined_score(results)

    print("\nNovelty Analysis Results:")
    print(f"Combined Score: {combined_score:.3f}")

    for algo_name, score in results.items():
        print(f"{algo_name}: {score.score:.3f} (confidence: {score.confidence:.3f})")
        if score.metadata:
            print(f"  Metadata: {score.metadata}")

    return results


async def mythology_inspiration_example():
    """Demonstrate creative inspiration generation for mythology"""
    print("\n=== Mythology Inspiration Example ===")

    # Initialize the inspiration engine
    engine = InspirationEngine()

    try:
        # Initialize connections (this will work if databases are available)
        initialized = await engine.initialize()
        if not initialized:
            print(
                "Warning: Could not initialize database connections. Using mock data."
            )
    except Exception as e:
        print(f"Warning: Could not initialize engine: {e}. Using mock data.")

    # Generate inspiration for mythology domain
    inspiration = await engine.generate_inspiration(
        domain="mythology",
        context={"theme": "transformation", "elements": ["hero", "monster", "quest"]},
        num_prompts=3,
    )

    print(f"Generated {len(inspiration['top_prompts'])} prompts for mythology domain")
    print(f"Diversity Score: {inspiration['ranking']['diversity_score']:.3f}")

    for i, prompt in enumerate(inspiration["top_prompts"], 1):
        print(f"\nPrompt {i}:")
        print(f"Content: {prompt['content']}")
        print(f"Creative Potential: {prompt['creative_potential']:.3f}")
        print(f"Novelty Score: {prompt['combined_novelty']:.3f}")

    # Shutdown the engine
    try:
        await engine.shutdown()
    except:
        pass

    return inspiration


async def geospatial_analysis_example():
    """Demonstrate geospatial data analysis"""
    print("\n=== Geospatial Analysis Example ===")

    # Sample geospatial data (latitude, longitude, entity type)
    geospatial_data = [
        {"lat": 40.7128, "lon": -74.0060, "entity": "city", "name": "New York"},
        {"lat": 34.0522, "lon": -118.2437, "entity": "city", "name": "Los Angeles"},
        {"lat": 41.8781, "lon": -87.6298, "entity": "city", "name": "Chicago"},
        {"lat": 29.7604, "lon": -95.3698, "entity": "city", "name": "Houston"},
        {"lat": 33.4484, "lon": -112.0740, "entity": "city", "name": "Phoenix"},
        {"lat": 25.7617, "lon": -80.1918, "entity": "city", "name": "Miami"},
        {"lat": 47.6062, "lon": -122.3321, "entity": "city", "name": "Seattle"},
        {"lat": 32.7767, "lon": -96.7970, "entity": "city", "name": "Dallas"},
        {"lat": 37.7749, "lon": -122.4194, "entity": "city", "name": "San Francisco"},
        {"lat": 42.3601, "lon": -71.0589, "entity": "city", "name": "Boston"},
        # Add some mythical locations
        {"lat": 36.1699, "lon": -115.1398, "entity": "mythical", "name": "Atlantis"},
        {"lat": 27.1751, "lon": 78.0421, "entity": "mythical", "name": "Shangri-La"},
        {"lat": 55.7558, "lon": 37.6173, "entity": "mythical", "name": "El Dorado"},
    ]

    # Initialize engine for analysis
    engine = InspirationEngine()

    try:
        # Analyze the geospatial data for novelty
        analysis_result = await engine.analyze_novelty(
            geospatial_data, context={"domain": "geography", "data_type": "geospatial"}
        )

        print("Geospatial Data Analysis:")
        print(f"Combined Novelty Score: {analysis_result['combined_score']:.3f}")
        print(f"Is Novel: {analysis_result['is_novel']}")
        print(f"Data Summary: {analysis_result['data_summary']}")

        if analysis_result["novelty_scores"]:
            print("\nAlgorithm Scores:")
            for algo, score_info in analysis_result["novelty_scores"].items():
                print(f"{algo}: {score_info['score']:.3f}")

    except Exception as e:
        print(f"Error in geospatial analysis: {e}")

    try:
        await engine.shutdown()
    except:
        pass


async def prompt_ranking_example():
    """Demonstrate prompt ranking system"""
    print("\n=== Prompt Ranking Example ===")

    # Create sample prompts
    sample_prompts = [
        {
            "id": "myth_1",
            "content": "Explore the mythological significance of mountains in ancient cultures",
            "domain": "mythology",
            "metadata": {"complexity": "high", "scope": "cultural"},
        },
        {
            "id": "geo_1",
            "content": "Describe how the geography of a region shapes its cultural development",
            "domain": "geography",
            "metadata": {"complexity": "medium", "scope": "regional"},
        },
        {
            "id": "narr_1",
            "content": "Create a narrative that bridges ancient myths with modern technology",
            "domain": "narrative",
            "metadata": {"complexity": "high", "scope": "global"},
        },
        {
            "id": "myth_2",
            "content": "Investigate the role of water in creation myths across different civilizations",
            "domain": "mythology",
            "metadata": {"complexity": "medium", "scope": "comparative"},
        },
        {
            "id": "cult_1",
            "content": "Examine how traditional rituals have evolved in contemporary society",
            "domain": "cultural",
            "metadata": {"complexity": "medium", "scope": "temporal"},
        },
    ]

    # Create prompt ranker
    ranker = PromptRanker()

    # Rank the prompts
    ranking = ranker.rank_prompts(sample_prompts)

    print(f"Ranked {len(ranking.ranked_prompts)} prompts")
    print(f"Overall Diversity Score: {ranking.diversity_score:.3f}")
    print(f"Ranking Criteria: {ranking.ranking_criteria}")

    print("\nRanked Prompts:")
    for i, prompt in enumerate(ranking.ranked_prompts, 1):
        print(f"\n{i}. {prompt.content}")
        print(f"   Domain: {prompt.domain}")
        print(f"   Creative Potential: {prompt.creative_potential:.3f}")
        print(f"   Combined Novelty: {prompt.combined_score:.3f}")

    # Get top 3 prompts
    top_prompts = ranker.get_top_prompts(ranking, top_n=3)
    print(f"\nTop 3 Prompts: {len(top_prompts)} selected")


async def collaborative_session_example():
    """Demonstrate collaborative session setup (mock example)"""
    print("\n=== Collaborative Session Example ===")

    # This would normally connect to real A2A agents
    print("Setting up collaborative inspiration session...")

    session_config = {
        "topic": "Ancient Civilizations and Modern Technology",
        "participants": ["mythology_agent", "geography_agent", "narrative_agent"],
        "duration_minutes": 30,
        "objectives": [
            "Identify connections between ancient myths and modern technology",
            "Generate creative prompts for interdisciplinary exploration",
            "Share novel insights across domains",
        ],
    }

    print(f"Session Topic: {session_config['topic']}")
    print(f"Participants: {', '.join(session_config['participants'])}")
    print(f"Duration: {session_config['duration_minutes']} minutes")
    print("Objectives:")
    for i, obj in enumerate(session_config["objectives"], 1):
        print(f"  {i}. {obj}")

    print("\nNote: This is a mock example. In a real implementation,")
    print("this would establish connections with actual A2A agents.")


async def main():
    """Run all examples"""
    print("Inspiration Engine - Usage Examples")
    print("=" * 50)

    try:
        # Run examples
        await basic_novelty_detection_example()
        await mythology_inspiration_example()
        await geospatial_analysis_example()
        await prompt_ranking_example()
        await collaborative_session_example()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    # Run the examples
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
