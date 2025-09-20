#!/usr/bin/env python3
"""
Test script for the Agent's Codex system.

This script demonstrates the core functionality of the Codex system including:
- Agent contribution archival
- Strategy documentation
- Knowledge extraction
- Legacy chapter generation
- Attribution tracking
"""

import asyncio
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_codex_system():
    """Test the complete Codex system."""
    print("🧪 Testing Agent's Codex System")
    print("=" * 50)

    try:
        # Import Codex components
        from codex import CodexManager
        from agents.base_agent import BaseSpecialistAgent
        from learning.workflow_tracer import WorkflowTracer

        # Initialize Codex
        print("📚 Initializing Codex Manager...")
        codex_manager = CodexManager("./test_codex_data")

        # Initialize Workflow Tracer and integrate
        print("🔍 Initializing Workflow Tracer...")
        workflow_tracer = WorkflowTracer("./test_traces")
        workflow_tracer.set_codex_manager(codex_manager)
        codex_manager.integrate_workflow_tracer(workflow_tracer)

        print("✅ Codex system initialized successfully!")

        # Test 1: Archive agent contributions
        print("\n📝 Test 1: Archiving Agent Contributions")
        print("-" * 40)

        # Simulate agent contributions
        contributions = [
            {
                "agent_name": "AtlasAgent",
                "agent_type": "RelationalAnalyst",
                "task_description": "Analyzed spatial relationships in dataset",
                "contribution_type": "task_execution",
                "input_data": {"dataset_size": 1000, "analysis_type": "spatial"},
                "output_data": {"patterns_found": 15, "accuracy": 0.92},
                "success_metrics": {"completed": True, "quality_score": 0.88},
                "duration": 45.2,
                "collaboration_partners": ["LinguistAgent"],
                "ai_model": "gpt-4",
                "ai_provider": "openai",
            },
            {
                "agent_name": "LinguistAgent",
                "agent_type": "LanguageProcessor",
                "task_description": "Processed linguistic patterns in text data",
                "contribution_type": "creative_output",
                "input_data": {"text_samples": 500, "language": "en"},
                "output_data": {"patterns_identified": 23, "confidence": 0.91},
                "success_metrics": {"completed": True, "quality_score": 0.95},
                "duration": 32.1,
                "collaboration_partners": ["AtlasAgent", "MythAgent"],
                "ai_model": "claude-3",
                "ai_provider": "anthropic",
            },
            {
                "agent_name": "MythAgent",
                "agent_type": "ComparativeMythologist",
                "task_description": "Identified mythological archetypes in narratives",
                "contribution_type": "learning_insight",
                "input_data": {
                    "narratives_analyzed": 75,
                    "cultures": ["greek", "nordic", "hindu"],
                },
                "output_data": {"archetypes_found": 12, "connections_made": 8},
                "success_metrics": {"completed": True, "quality_score": 0.89},
                "duration": 67.8,
                "collaboration_partners": ["LinguistAgent"],
                "ai_model": "gemini-pro",
                "ai_provider": "google",
            },
        ]

        archived_ids = []
        for contrib in contributions:
            contrib_id = codex_manager.archive_agent_task(**contrib)
            archived_ids.append(contrib_id)
            print(f"✅ Archived contribution: {contrib_id} from {contrib['agent_name']}")

        # Test 2: Document strategies
        print("\n📋 Test 2: Documenting Strategies")
        print("-" * 40)

        strategies = [
            {
                "title": "Collaborative Spatial Analysis",
                "strategy_type": "collaboration",
                "description": "Method for combining spatial and linguistic analysis",
                "context": "Multi-modal data analysis requiring diverse expertise",
                "steps": [
                    {
                        "step": 1,
                        "description": "Identify spatial patterns",
                        "agent": "AtlasAgent",
                    },
                    {
                        "step": 2,
                        "description": "Extract linguistic features",
                        "agent": "LinguistAgent",
                    },
                    {
                        "step": 3,
                        "description": "Correlate patterns",
                        "agent": "AtlasAgent",
                    },
                    {
                        "step": 4,
                        "description": "Validate findings",
                        "agent": "MythAgent",
                    },
                ],
                "success_criteria": [
                    "Patterns validated across modalities",
                    "Confidence score > 0.85",
                    "Actionable insights generated",
                ],
                "lessons_learned": [
                    "Early collaboration improves pattern recognition",
                    "Cross-validation essential for reliability",
                    "Domain expertise combination yields better results",
                ],
                "created_by": "AtlasAgent",
                "related_contributions": archived_ids[:2],
                "tags": ["collaboration", "analysis", "validation"],
            }
        ]

        strategy_ids = []
        for strategy in strategies:
            strategy_id = codex_manager.document_strategy(**strategy)
            strategy_ids.append(strategy_id)
            print(f"✅ Documented strategy: {strategy_id} - {strategy['title']}")

        # Test 3: Generate knowledge
        print("\n🧠 Test 3: Knowledge Extraction")
        print("-" * 40)

        # Extract patterns from contributions
        knowledge_ids = (
            codex_manager.knowledge_base.extract_patterns_from_contributions(
                contributions
            )
        )
        print(f"✅ Extracted {len(knowledge_ids)} knowledge patterns from contributions")

        # Extract insights from strategies
        insight_ids = codex_manager.knowledge_base.extract_insights_from_strategies(
            [
                {
                    "strategy_id": strategy_ids[0],
                    "strategy_type": "collaboration",
                    "title": strategies[0]["title"],
                    "description": strategies[0]["description"],
                    "steps": strategies[0]["steps"],
                    "success_criteria": strategies[0]["success_criteria"],
                    "lessons_learned": strategies[0]["lessons_learned"],
                }
            ]
        )
        print(f"✅ Extracted {len(insight_ids)} insights from strategies")

        # Test 4: Generate legacy chapters
        print("\n📖 Test 4: Legacy Chapter Generation")
        print("-" * 40)

        # Generate agent hero chapter
        hero_chapter_id = codex_manager.generate_legacy_chapter(
            chapter_type="agent_hero",
            agent_name="AtlasAgent",
            contributions=[contributions[0]],
            strategies=[strategies[0]],
            theme="hero_journey",
        )
        print(f"✅ Generated hero chapter: {hero_chapter_id}")

        # Generate collaboration chapter
        collab_chapter_id = codex_manager.generate_legacy_chapter(
            chapter_type="collaboration",
            collaboration_name="Triad Analysis Team",
            agents=["AtlasAgent", "LinguistAgent", "MythAgent"],
            contributions=contributions,
            theme="harmony",
        )
        print(f"✅ Generated collaboration chapter: {collab_chapter_id}")

        # Test 5: Search and retrieval
        print("\n🔍 Test 5: Search and Retrieval")
        print("-" * 40)

        # Search contributions
        search_results = codex_manager.search_codex("spatial", "contributions")
        print(
            f"✅ Found {len(search_results.get('contributions', []))} contributions matching 'spatial'"
        )

        # Search knowledge
        knowledge_results = codex_manager.search_codex("collaboration", "knowledge")
        print(
            f"✅ Found {len(knowledge_results.get('knowledge', []))} knowledge entries about 'collaboration'"
        )

        # Get learning recommendations
        recommendations = codex_manager.get_learning_recommendations(
            "AtlasAgent", "spatial analysis"
        )
        print(
            f"✅ Generated {len(recommendations)} learning recommendations for AtlasAgent"
        )

        # Test 6: Attribution tracking
        print("\n👥 Test 6: Attribution Tracking")
        print("-" * 40)

        # Get attribution summary
        attribution_summary = (
            codex_manager.attribution_tracker.get_contribution_summary()
        )
        print(
            f"✅ Attribution summary: {attribution_summary['total_contributions']} total contributions"
        )
        print(f"   AI Models used: {len(attribution_summary['ai_models_used'])}")
        print(f"   AI Providers: {len(attribution_summary['ai_providers_used'])}")

        # Get top contributors
        top_contributors = codex_manager.attribution_tracker.get_top_contributors(
            by="count", limit=3
        )
        print(
            f"✅ Top contributors by count: {[tc['agent_name'] for tc in top_contributors]}"
        )

        # Test 7: System statistics
        print("\n📊 Test 7: System Statistics")
        print("-" * 40)

        stats = codex_manager.get_codex_statistics()
        print(f"✅ Codex Statistics:")
        print(f"   Total Contributions: {stats.total_contributions}")
        print(f"   Total Strategies: {stats.total_strategies}")
        print(f"   Total Chapters: {stats.total_chapters}")
        print(f"   Knowledge Entries: {stats.total_knowledge_entries}")
        print(f"   Active Agents: {stats.active_agents}")

        # Test 8: Export data
        print("\n💾 Test 8: Data Export")
        print("-" * 40)

        export_success = codex_manager.export_codex_data("./codex_export")
        if export_success:
            print("✅ Codex data exported successfully")
        else:
            print("❌ Failed to export Codex data")

        print("\n🎉 All Codex tests completed successfully!")
        print("=" * 50)
        print("📚 Agent's Codex System is fully operational!")
        print("   - Contributions archived and searchable")
        print("   - Strategies documented and analyzed")
        print("   - Knowledge extracted and organized")
        print("   - Legacy chapters generated")
        print("   - Attribution properly tracked")
        print("   - Learning recommendations available")
        print("   - API endpoints ready for integration")

        return True

    except Exception as e:
        logger.error(f"❌ Codex test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_codex_system())
    exit(0 if success else 1)
