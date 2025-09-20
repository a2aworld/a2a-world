#!/usr/bin/env python3
"""
Example Usage of the Collective Learning Loop

This script demonstrates how to use the Collective Learning Loop system
to continuously improve Terra Constellata's agent coordination and creative capabilities.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain.llms.base import BaseLLM
from langchain.llms.fake import FakeListLLM

from ..agents.sentinel.sentinel_orchestrator import SentinelOrchestrator
from .collective_learning_loop import CollectiveLearningLoop
from .workflow_tracer import WorkflowTracer
from .pattern_analyzer import PatternAnalyzer
from .feedback_collector import FeedbackCollector, UserFeedback, CATScore
from .prompt_optimizer import PromptOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MockLLM(FakeListLLM):
    """Mock LLM for testing purposes."""

    def __init__(self):
        responses = [
            "I am analyzing the creative territory and identifying key patterns.",
            "Coordinating with specialized agents for optimal task execution.",
            "Generating innovative solutions based on mythological analysis.",
            "Optimizing workflow based on learned patterns and feedback.",
            "Synthesizing insights from multiple agent perspectives.",
        ]
        super().__init__(responses=responses)


async def basic_learning_loop_example():
    """Demonstrate basic Collective Learning Loop usage."""
    print("\n=== Basic Collective Learning Loop Example ===")

    # Initialize components
    llm = MockLLM()
    orchestrator = SentinelOrchestrator(llm)

    # Create learning system
    learning_config = {
        "learning_interval_minutes": 5,  # Faster for demo
        "min_samples_for_learning": 3,
        "enable_rl_training": False,  # Disable for basic example
        "enable_pattern_discovery": True,
        "enable_prompt_optimization": True,
    }

    learning_loop = CollectiveLearningLoop(orchestrator, learning_config)

    print("Learning system initialized")

    # Start learning
    await learning_loop.start_learning()
    print("Learning loop started")

    # Simulate some workflow executions
    print("Simulating workflow executions...")

    # Create some mock workflow traces
    tracer = learning_loop.workflow_tracer

    for i in range(5):
        trace = tracer.start_trace(f"demo_workflow_{i}", "creative_discovery")

        # Simulate workflow execution
        await asyncio.sleep(0.1)  # Simulate processing time

        # Complete trace with mock data
        final_state = {
            "insights_generated": i + 1,
            "agents_coordinated": min(i + 1, 3),
            "creative_territories_discovered": i * 2,
        }

        success_metrics = {
            "workflow_efficiency": 0.7 + (i * 0.05),
            "agent_coordination_score": 0.6 + (i * 0.08),
            "task_completion_rate": 0.8 + (i * 0.03),
            "avg_agent_response_time": 2.0 - (i * 0.2),
        }

        trace.complete_trace(final_state, success=True)
        tracer.store_trace(trace)

        print(f"Completed workflow {i + 1}/5")

    # Wait for learning cycle to process
    await asyncio.sleep(6)  # Wait for learning cycle

    # Get learning status
    status = await learning_loop.get_learning_status()
    print(f"Learning cycles completed: {status['learning_cycles_completed']}")
    print(f"Workflows processed: {status['metrics']['total_workflows_processed']}")
    print(f"Patterns discovered: {status['metrics']['patterns_discovered']}")

    # Stop learning
    await learning_loop.stop_learning()
    print("Learning loop stopped")

    return learning_loop


async def feedback_integration_example():
    """Demonstrate feedback collection and integration."""
    print("\n=== Feedback Integration Example ===")

    # Initialize feedback collector
    feedback_collector = FeedbackCollector("./demo_feedback")

    # Create mock feedback
    users = ["alice", "bob", "charlie"]
    workflows = ["workflow_1", "workflow_2", "workflow_3"]

    print("Submitting mock feedback...")

    for i, workflow_id in enumerate(workflows):
        for j, user_id in enumerate(users):
            # Create feedback
            feedback = UserFeedback(workflow_id, user_id)

            # Add CAT score
            cat_score = 6 + (i + j) % 4  # Scores between 6-9
            dimensions = {
                "creativity": cat_score,
                "relevance": cat_score - 1,
                "usefulness": cat_score + 1,
            }
            feedback.add_cat_score(
                cat_score, dimensions, f"Great work on {workflow_id}!"
            )

            # Add ratings
            feedback.set_ratings(
                satisfaction=0.7 + (i * 0.1),
                usefulness=0.6 + (j * 0.1),
                novelty=0.8 + ((i + j) * 0.05),
                quality=0.75 + (i * 0.08),
            )

            # Submit feedback
            success = feedback_collector.submit_feedback(feedback)
            print(
                f"Feedback submitted: {workflow_id} by {user_id} - Success: {success}"
            )

    # Get feedback statistics
    stats = feedback_collector.get_feedback_statistics()
    print("\nFeedback Statistics:")
    print(f"Total feedback: {stats['total_feedback']}")
    print(f"Average CAT score: {stats['avg_cat_score']:.2f}")
    print(f"Average composite score: {stats['avg_composite_score']:.3f}")

    # Get high-scoring workflows
    high_scoring = feedback_collector.get_high_scoring_workflows(0.8)
    print(f"High-scoring workflows: {high_scoring}")

    return feedback_collector


async def pattern_analysis_example():
    """Demonstrate pattern analysis capabilities."""
    print("\n=== Pattern Analysis Example ===")

    # Create mock workflow traces
    tracer = WorkflowTracer("./demo_traces")

    # Generate diverse workflow patterns
    workflow_types = ["creative_discovery", "research_analysis", "content_creation"]
    node_sequences = [
        [
            "scan_territories",
            "dispatch_agents",
            "collect_responses",
            "synthesize_insights",
        ],
        [
            "analyze_data",
            "identify_patterns",
            "generate_hypotheses",
            "validate_findings",
        ],
        ["gather_inspiration", "refine_ideas", "create_content", "review_output"],
    ]

    print("Creating mock workflow traces...")

    for i in range(10):
        workflow_type = workflow_types[i % len(workflow_types)]
        trace = tracer.start_trace(f"pattern_demo_{i}", workflow_type)

        # Simulate execution with some variation
        sequence = node_sequences[i % len(node_sequences)]
        if i % 3 == 0:  # Add some variation
            sequence = sequence[:-1] + ["optimize_output"]

        for j, node in enumerate(sequence):
            # Simulate node execution
            input_state = {"step": j}
            output_state = {"step": j + 1, "result": f"completed_{node}"}
            execution_time = 0.5 + (j * 0.2)

            trace.add_node_execution(node, input_state, output_state, execution_time)

        # Complete trace
        final_state = {"pattern_type": workflow_type, "steps_completed": len(sequence)}
        success_metrics = {
            "workflow_efficiency": 0.6 + (i * 0.03),
            "agent_coordination_score": 0.7 + (i * 0.02),
            "task_completion_rate": 0.85 + (i * 0.01),
        }

        trace.complete_trace(final_state, success=True)
        tracer.store_trace(trace)

    # Analyze patterns
    pattern_analyzer = PatternAnalyzer(tracer)
    results = pattern_analyzer.analyze_patterns()

    print("Pattern Analysis Results:")
    print(f"Workflow patterns found: {len(results['workflow_patterns'])}")
    print(f"Agent patterns found: {len(results['agent_patterns'])}")
    print(f"Decision patterns found: {len(results['decision_patterns'])}")
    print(
        f"Overall pattern quality: {results['pattern_quality']['overall_quality']:.3f}"
    )

    # Get recommendations
    recommendations = pattern_analyzer.get_pattern_recommendations(
        {"workflow_type": "creative_discovery", "task_type": "content_creation"}
    )

    print(
        f"Optimization recommendations: {len(recommendations['suggested_workflow_patterns'])} workflow patterns"
    )

    return pattern_analyzer


async def prompt_optimization_example():
    """Demonstrate prompt optimization capabilities."""
    print("\n=== Prompt Optimization Example ===")

    # Initialize components
    feedback_collector = FeedbackCollector("./demo_feedback")
    tracer = WorkflowTracer("./demo_traces")
    pattern_analyzer = PatternAnalyzer(tracer)

    prompt_optimizer = PromptOptimizer(feedback_collector, pattern_analyzer)

    # Generate optimized prompts
    contexts = [
        {"topic": "mythological creatures", "domain": "mythology", "style": "creative"},
        {
            "topic": "cultural symbolism",
            "domain": "anthropology",
            "style": "analytical",
        },
        {"topic": "narrative structures", "domain": "literature", "style": "technical"},
    ]

    print("Generating optimized prompts...")

    for i, context in enumerate(contexts):
        # Generate optimized prompt
        prompt = prompt_optimizer.generate_optimized_prompt(
            "creative_inspiration", **context
        )

        if prompt:
            print(f"Generated prompt {i + 1}:")
            print(f"Length: {len(prompt)} characters")
            print(f"Preview: {prompt[:100]}...")

            # Simulate prompt performance
            success = i % 2 == 0  # Alternate success for demo
            response_time = 1.5 + (i * 0.3)
            feedback_score = 0.7 + (i * 0.1) if success else 0.4

            prompt_optimizer.record_prompt_performance(
                f"demo_prompt_{i}", success, response_time, feedback_score
            )

            print(
                f"Performance recorded - Success: {success}, Score: {feedback_score:.2f}"
            )
        else:
            print(f"Failed to generate prompt {i + 1}")

    # Analyze prompt effectiveness
    effectiveness = prompt_optimizer.analyze_prompt_effectiveness()
    print("\nPrompt Effectiveness Analysis:")
    print(f"Total prompts: {effectiveness['total_prompts']}")
    print(f"Successful prompts: {effectiveness['successful_prompts']}")
    print(f"Average success rate: {effectiveness['avg_success_rate']:.3f}")
    print(f"Average feedback score: {effectiveness['avg_feedback_score']:.3f}")

    return prompt_optimizer


async def comprehensive_integration_example():
    """Demonstrate comprehensive integration of all learning components."""
    print("\n=== Comprehensive Integration Example ===")

    # Initialize core system
    llm = MockLLM()
    orchestrator = SentinelOrchestrator(llm)

    # Initialize learning system
    learning_config = {
        "learning_interval_minutes": 2,  # Very fast for demo
        "min_samples_for_learning": 2,
        "enable_rl_training": False,
        "enable_pattern_discovery": True,
        "enable_prompt_optimization": True,
        "feedback_request_threshold": 0.7,
    }

    learning_loop = CollectiveLearningLoop(orchestrator, learning_config)

    print("Comprehensive learning system initialized")

    # Start learning
    await learning_loop.start_learning()

    # Simulate a complete workflow with feedback
    print("Simulating complete workflow with learning...")

    # 1. Create workflow trace
    tracer = learning_loop.workflow_tracer
    trace = tracer.start_trace("comprehensive_demo", "integrated_workflow")

    # 2. Simulate workflow execution
    workflow_steps = [
        "initialize_orchestration",
        "scan_creative_territories",
        "dispatch_specialist_agents",
        "coordinate_agent_interactions",
        "synthesize_results",
        "generate_final_output",
    ]

    for i, step in enumerate(workflow_steps):
        input_state = {"current_step": i, "progress": i / len(workflow_steps)}
        output_state = {
            "current_step": i + 1,
            "progress": (i + 1) / len(workflow_steps),
        }
        execution_time = 0.8 + (i * 0.2)

        trace.add_node_execution(step, input_state, output_state, execution_time)

        # Simulate agent interaction
        if "dispatch" in step or "coordinate" in step:
            trace.add_agent_interaction(
                "SpecialistAgent",
                f"Execute {step}",
                f"Completed {step} successfully",
                execution_time,
            )

    # 3. Complete trace
    final_state = {
        "workflow_completed": True,
        "insights_generated": 5,
        "agents_coordinated": 3,
        "creative_output_produced": True,
    }

    success_metrics = {
        "workflow_efficiency": 0.85,
        "agent_coordination_score": 0.9,
        "task_completion_rate": 0.95,
        "avg_agent_response_time": 1.2,
    }

    trace.complete_trace(final_state, success=True)
    tracer.store_trace(trace)

    print("Workflow trace completed and stored")

    # 4. Submit feedback
    feedback = UserFeedback("comprehensive_demo", "demo_user")
    feedback.add_cat_score(9, {"creativity": 9, "coordination": 8, "efficiency": 9})
    feedback.set_ratings(satisfaction=0.95, usefulness=0.9, novelty=0.85, quality=0.92)

    learning_loop.feedback_collector.submit_feedback(feedback)
    print("Feedback submitted")

    # 5. Wait for learning cycle
    await asyncio.sleep(3)

    # 6. Get system status
    status = await learning_loop.get_learning_status()
    print("\nLearning System Status:")
    print(f"Learning active: {status['is_learning']}")
    print(f"Cycles completed: {status['learning_cycles_completed']}")
    print(f"Workflows processed: {status['metrics']['total_workflows_processed']}")
    print(f"Feedback collected: {status['metrics']['feedback_collected']}")

    # 7. Get optimization recommendations
    current_state = {
        "active_tasks": 2,
        "agent_utilization": 0.7,
        "workflow_type": "creative_discovery",
    }

    recommendations = await learning_loop.optimize_orchestrator_decision(current_state)
    print(
        f"Optimization recommendations: {len(recommendations['suggested_actions'])} actions suggested"
    )

    # 8. Export learning data
    learning_loop.export_learning_data("comprehensive_demo_export")
    print("Learning data exported")

    # Stop learning
    await learning_loop.stop_learning()
    print("Comprehensive demo completed")

    return learning_loop


async def main():
    """Run all examples."""
    print("Collective Learning Loop - Comprehensive Examples")
    print("=" * 60)

    try:
        # Run examples
        await basic_learning_loop_example()
        await feedback_integration_example()
        await pattern_analysis_example()
        await prompt_optimization_example()
        await comprehensive_integration_example()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ Workflow trace capture and analysis")
        print("✓ Pattern discovery and optimization")
        print("✓ User feedback collection and integration")
        print("✓ Prompt optimization and performance tracking")
        print("✓ Reinforcement learning environment")
        print("✓ Comprehensive learning loop integration")
        print("✓ Real-time optimization recommendations")

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    # Run the examples
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
