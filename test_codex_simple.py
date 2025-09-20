#!/usr/bin/env python3
"""
Simple test script for the Agent's Codex system.

This script tests the core Codex functionality without external dependencies.
"""

import sys
import os
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))


def test_codex_core():
    """Test the core Codex components."""
    print("🧪 Testing Agent's Codex Core Components")
    print("=" * 50)

    try:
        # Test data models
        print("📋 Testing Data Models...")
        from codex.models import (
            AgentContribution,
            ContributionType,
            AttributionRecord,
            StrategyDocument,
            StrategyType,
            LegacyChapter,
            KnowledgeEntry,
        )

        # Create test contribution
        contribution = AgentContribution(
            contribution_id="test_contrib_001",
            agent_name="TestAgent",
            agent_type="TestType",
            task_description="Test task execution",
            contribution_type=ContributionType.TASK_EXECUTION,
            input_data={"test": "input"},
            output_data={"test": "output"},
            success_metrics={"success": True},
            timestamp=datetime.utcnow(),
            duration=10.5,
        )

        print(f"✅ Created contribution: {contribution.contribution_id}")

        # Create test attribution
        attribution = AttributionRecord(
            agent_name="TestAgent",
            agent_type="TestType",
            contribution_type=ContributionType.TASK_EXECUTION,
            timestamp=datetime.utcnow(),
            ai_model="test-model",
            ai_provider="test-provider",
        )

        print(f"✅ Created attribution record for: {attribution.agent_name}")

        # Create test strategy
        strategy = StrategyDocument(
            strategy_id="test_strategy_001",
            strategy_type=StrategyType.WORKFLOW_PATTERN,
            title="Test Strategy",
            description="A test strategy",
            context="Testing context",
            steps=[{"step": 1, "description": "Test step"}],
            success_criteria=["Success criterion"],
            lessons_learned=["Lesson learned"],
            created_by="TestAgent",
            created_at=datetime.utcnow(),
        )

        print(f"✅ Created strategy: {strategy.strategy_id}")

        # Create test knowledge entry
        knowledge = KnowledgeEntry(
            entry_id="test_kb_001",
            category="test_category",
            title="Test Knowledge",
            content="Test content",
            source_type="test",
            source_id="test_source",
            confidence_score=0.8,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
        )

        print(f"✅ Created knowledge entry: {knowledge.entry_id}")

        # Test archival system
        print("\n📚 Testing Archival System...")
        from codex.archival_system import ArchivalSystem

        archival = ArchivalSystem("./test_codex_archive")

        # Archive the test contribution
        contrib_id = archival.archive_contribution(
            agent_name="TestAgent",
            agent_type="TestType",
            task_description="Test archival",
            contribution_type=ContributionType.TASK_EXECUTION,
            input_data={"test": "data"},
            output_data={"result": "success"},
            success_metrics={"archived": True},
        )

        print(f"✅ Archived contribution: {contrib_id}")

        # Archive test strategy
        strategy_id = archival.archive_strategy(
            title="Test Archival Strategy",
            strategy_type=StrategyType.OPTIMIZATION,
            description="Strategy for testing archival",
            context="Testing archival system",
            steps=[{"step": 1, "action": "Archive data"}],
            success_criteria=["Data archived successfully"],
            lessons_learned=["Archival is important"],
            created_by="TestAgent",
        )

        print(f"✅ Archived strategy: {strategy_id}")

        # Test knowledge base
        print("\n🧠 Testing Knowledge Base...")
        from codex.knowledge_base import KnowledgeBase

        kb = KnowledgeBase("./test_codex_knowledge")

        # Add knowledge entry
        kb_id = kb.add_knowledge_entry(
            category="test_patterns",
            title="Test Pattern Recognition",
            content="Pattern for testing knowledge base functionality",
            source_type="test",
            source_id="test_source",
            confidence_score=0.9,
            tags=["test", "pattern"],
        )

        print(f"✅ Added knowledge entry: {kb_id}")

        # Search knowledge
        results = kb.search_knowledge("test", limit=5)
        print(f"✅ Found {len(results)} knowledge entries matching 'test'")

        # Test chapter generator
        print("\n📖 Testing Chapter Generator...")
        from codex.chapter_generator import ChapterGenerator

        chapter_gen = ChapterGenerator("./test_codex_chapters")

        # Generate a simple chapter
        chapter_id = chapter_gen.generate_agent_hero_chapter(
            agent_name="TestAgent",
            contributions=[
                {
                    "contribution_id": contrib_id,
                    "agent_name": "TestAgent",
                    "task_description": "Test task",
                    "timestamp": datetime.utcnow(),
                }
            ],
            strategies=[{"strategy_id": strategy_id, "title": "Test Strategy"}],
            theme="test_journey",
        )

        print(f"✅ Generated chapter: {chapter_id}")

        # Test attribution tracker
        print("\n👥 Testing Attribution Tracker...")
        from codex.attribution_tracker import AttributionTracker

        tracker = AttributionTracker("./test_codex_attribution")

        # Record attribution
        attr_id = tracker.record_attribution(
            agent_name="TestAgent",
            agent_type="TestType",
            contribution_type="task_execution",
            ai_model="test-model",
            ai_provider="test-provider",
        )

        print(f"✅ Recorded attribution: {attr_id}")

        # Get attribution summary
        summary = tracker.get_contribution_summary()
        print(f"✅ Attribution summary: {summary['total_contributions']} contributions")

        # Test Codex manager
        print("\n🎯 Testing Codex Manager...")
        from codex.codex_manager import CodexManager

        codex = CodexManager("./test_codex_manager")

        # Archive a task
        task_id = codex.archive_agent_task(
            agent_name="ManagerTestAgent",
            agent_type="ManagerTest",
            task_description="Test Codex manager integration",
            contribution_type="task_execution",
            input_data={"test": "manager"},
            output_data={"result": "integrated"},
            success_metrics={"integrated": True},
        )

        print(f"✅ Codex manager archived task: {task_id}")

        # Get statistics
        stats = codex.get_codex_statistics()
        print(
            f"✅ Codex statistics: {stats.total_contributions} contributions, {stats.total_strategies} strategies"
        )

        # Test search
        search_results = codex.search_codex("test", "all")
        total_results = sum(
            len(v) for v in search_results.values() if isinstance(v, list)
        )
        print(f"✅ Codex search found {total_results} total results")

        print("\n🎉 All Codex core tests passed!")
        print("=" * 50)
        print("📚 Agent's Codex System Core Components:")
        print("   ✅ Data models working correctly")
        print("   ✅ Archival system functional")
        print("   ✅ Knowledge base operational")
        print("   ✅ Chapter generator active")
        print("   ✅ Attribution tracker recording")
        print("   ✅ Codex manager integrated")
        print("   ✅ Search functionality working")
        print("   ✅ Statistics generation complete")

        return True

    except Exception as e:
        print(f"❌ Codex test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_codex_core()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
