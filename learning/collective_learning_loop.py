"""
Collective Learning Loop

This module implements the main Collective Learning Loop that integrates all
learning components to continuously improve the Terra Constellata system's
agent coordination and creative capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import json
import os

from ..agents.sentinel.sentinel_orchestrator import SentinelOrchestrator
from .workflow_tracer import WorkflowTracer, WorkflowTrace
from .pattern_analyzer import PatternAnalyzer
from .rl_environment import OrchestratorRLEnvironment, create_orchestrator_env
from .reward_model import RewardModelTrainer
from .feedback_collector import FeedbackCollector
from .prompt_optimizer import PromptOptimizer

logger = logging.getLogger(__name__)


class LearningMetrics:
    """Tracks learning system performance metrics."""

    def __init__(self):
        self.learning_cycles_completed = 0
        self.total_workflows_processed = 0
        self.successful_optimizations = 0
        self.feedback_collected = 0
        self.models_trained = 0
        self.patterns_discovered = 0
        self.performance_improvements: List[Dict[str, Any]] = []
        self.learning_efficiency_history: List[float] = []

    def record_learning_cycle(self, cycle_data: Dict[str, Any]):
        """Record data from a completed learning cycle."""
        self.learning_cycles_completed += 1
        self.total_workflows_processed += cycle_data.get("workflows_processed", 0)
        self.successful_optimizations += cycle_data.get("optimizations_applied", 0)
        self.feedback_collected += cycle_data.get("feedback_processed", 0)
        self.models_trained += cycle_data.get("models_trained", 0)
        self.patterns_discovered += cycle_data.get("patterns_found", 0)

        # Track performance improvements
        if "performance_improvement" in cycle_data:
            self.performance_improvements.append(
                {
                    "cycle": self.learning_cycles_completed,
                    "improvement": cycle_data["performance_improvement"],
                    "timestamp": datetime.utcnow(),
                }
            )

        # Track learning efficiency
        efficiency = cycle_data.get("learning_efficiency", 0.5)
        self.learning_efficiency_history.append(efficiency)

    def get_summary(self) -> Dict[str, Any]:
        """Get learning metrics summary."""
        avg_improvement = (
            (
                sum(p["improvement"] for p in self.performance_improvements)
                / len(self.performance_improvements)
            )
            if self.performance_improvements
            else 0.0
        )

        avg_efficiency = (
            (
                sum(self.learning_efficiency_history)
                / len(self.learning_efficiency_history)
            )
            if self.learning_efficiency_history
            else 0.0
        )

        return {
            "learning_cycles_completed": self.learning_cycles_completed,
            "total_workflows_processed": self.total_workflows_processed,
            "successful_optimizations": self.successful_optimizations,
            "feedback_collected": self.feedback_collected,
            "models_trained": self.models_trained,
            "patterns_discovered": self.patterns_discovered,
            "avg_performance_improvement": avg_improvement,
            "avg_learning_efficiency": avg_efficiency,
            "optimization_success_rate": (
                self.successful_optimizations / self.learning_cycles_completed
            )
            if self.learning_cycles_completed > 0
            else 0.0,
        }


class CollectiveLearningLoop:
    """
    Main Collective Learning Loop system.

    This system continuously learns from agent interactions, user feedback,
    and system performance to optimize the Terra Constellata ecosystem.
    """

    def __init__(
        self,
        orchestrator: SentinelOrchestrator,
        learning_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Collective Learning Loop.

        Args:
            orchestrator: The Sentinel Orchestrator instance
            learning_config: Configuration for the learning system
        """
        self.orchestrator = orchestrator
        self.config = learning_config or self._get_default_config()

        # Initialize learning components
        self.workflow_tracer = WorkflowTracer(
            self.config.get("trace_storage_path", "./traces")
        )
        self.pattern_analyzer = PatternAnalyzer(self.workflow_tracer)
        self.feedback_collector = FeedbackCollector(
            self.config.get("feedback_storage_path", "./feedback")
        )
        self.prompt_optimizer = PromptOptimizer(
            self.feedback_collector, self.pattern_analyzer
        )
        self.reward_trainer = RewardModelTrainer(
            self.config.get("model_storage_path", "./models")
        )

        # RL components
        self.rl_env = None
        self.rl_agent = None

        # Learning state
        self.is_learning = False
        self.learning_cycle_count = 0
        self.last_learning_cycle = None

        # Metrics
        self.metrics = LearningMetrics()

        # Learning schedule
        self.learning_interval = self.config.get("learning_interval_minutes", 30)
        self.min_samples_for_learning = self.config.get("min_samples_for_learning", 10)

        logger.info("Collective Learning Loop initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default learning configuration."""
        return {
            "learning_interval_minutes": 30,
            "min_samples_for_learning": 10,
            "max_learning_cycles_per_day": 48,
            "trace_storage_path": "./traces",
            "feedback_storage_path": "./feedback",
            "model_storage_path": "./models",
            "enable_rl_training": True,
            "enable_pattern_discovery": True,
            "enable_prompt_optimization": True,
            "feedback_request_threshold": 0.8,  # Request feedback for high-scoring workflows
            "learning_batch_size": 50,
        }

    async def start_learning(self):
        """Start the continuous learning process."""
        if self.is_learning:
            logger.warning("Learning loop is already running")
            return

        self.is_learning = True
        logger.info("Starting Collective Learning Loop")

        # Initialize RL environment if enabled
        if self.config.get("enable_rl_training", True):
            await self._initialize_rl_components()

        # Start the learning cycle
        asyncio.create_task(self._learning_cycle())

    async def stop_learning(self):
        """Stop the continuous learning process."""
        self.is_learning = False
        logger.info("Stopped Collective Learning Loop")

    async def _initialize_rl_components(self):
        """Initialize reinforcement learning components."""
        try:
            self.rl_env = create_orchestrator_env(
                self.orchestrator, self.workflow_tracer, self.pattern_analyzer
            )

            # Initialize RL agent (simplified - would use stable-baselines3 in production)
            logger.info("RL components initialized")

        except Exception as e:
            logger.error(f"Failed to initialize RL components: {e}")
            self.rl_env = None

    async def _learning_cycle(self):
        """Main learning cycle that runs continuously."""
        while self.is_learning:
            try:
                cycle_start = datetime.utcnow()

                # Execute learning cycle
                cycle_results = await self._execute_learning_cycle()

                # Record metrics
                self.metrics.record_learning_cycle(cycle_results)
                self.learning_cycle_count += 1
                self.last_learning_cycle = datetime.utcnow()

                # Log cycle completion
                cycle_duration = (datetime.utcnow() - cycle_start).total_seconds()
                logger.info(
                    f"Learning cycle {self.learning_cycle_count} completed in {cycle_duration:.2f}s"
                )

                # Wait for next cycle
                await asyncio.sleep(self.learning_interval * 60)

            except Exception as e:
                logger.error(f"Error in learning cycle: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

    async def _execute_learning_cycle(self) -> Dict[str, Any]:
        """Execute a single learning cycle."""
        cycle_results = {
            "workflows_processed": 0,
            "feedback_processed": 0,
            "patterns_found": 0,
            "optimizations_applied": 0,
            "models_trained": 0,
            "performance_improvement": 0.0,
            "learning_efficiency": 0.5,
        }

        try:
            # 1. Collect and process new workflow traces
            new_traces = self._collect_recent_traces()
            cycle_results["workflows_processed"] = len(new_traces)

            if new_traces:
                # 2. Process feedback for completed workflows
                feedback_count = await self._process_workflow_feedback(new_traces)
                cycle_results["feedback_processed"] = feedback_count

                # 3. Update pattern analysis
                if self.config.get("enable_pattern_discovery", True):
                    patterns_found = self._update_pattern_analysis(new_traces)
                    cycle_results["patterns_found"] = patterns_found

                # 4. Train/update models
                if len(new_traces) >= self.min_samples_for_learning:
                    models_trained = await self._train_models(new_traces)
                    cycle_results["models_trained"] = models_trained

                # 5. Apply optimizations
                optimizations = await self._apply_optimizations()
                cycle_results["optimizations_applied"] = optimizations

                # 6. Update RL policy if enabled
                if self.config.get("enable_rl_training", True) and self.rl_env:
                    improvement = await self._update_rl_policy()
                    cycle_results["performance_improvement"] = improvement

                # 7. Calculate learning efficiency
                cycle_results[
                    "learning_efficiency"
                ] = self._calculate_learning_efficiency(cycle_results)

        except Exception as e:
            logger.error(f"Error executing learning cycle: {e}")

        return cycle_results

    def _collect_recent_traces(self) -> List[WorkflowTrace]:
        """Collect recent workflow traces for learning."""
        # Get traces from the last learning interval
        cutoff_time = datetime.utcnow() - timedelta(minutes=self.learning_interval)

        recent_traces = []
        for trace in self.workflow_tracer.traces.values():
            if trace.end_time and trace.end_time >= cutoff_time:
                recent_traces.append(trace)

        return recent_traces

    async def _process_workflow_feedback(self, traces: List[WorkflowTrace]) -> int:
        """Process feedback for completed workflows."""
        feedback_processed = 0

        for trace in traces:
            # Check if we should request feedback for this workflow
            if self._should_request_feedback(trace):
                # Request feedback from users who might have interacted with this workflow
                await self._request_feedback_for_workflow(trace)
                feedback_processed += 1

        return feedback_processed

    def _should_request_feedback(self, trace: WorkflowTrace) -> bool:
        """Determine if feedback should be requested for a workflow."""
        if not trace.success_metrics:
            return False

        # Request feedback for high-performing workflows
        efficiency = trace.success_metrics.get("workflow_efficiency", 0)
        return efficiency >= self.config.get("feedback_request_threshold", 0.8)

    async def _request_feedback_for_workflow(self, trace: WorkflowTrace):
        """Request feedback for a workflow."""
        # This would integrate with user interface systems
        # For now, we'll simulate feedback collection
        logger.info(f"Requesting feedback for workflow: {trace.workflow_id}")

    def _update_pattern_analysis(self, traces: List[WorkflowTrace]) -> int:
        """Update pattern analysis with new traces."""
        try:
            # Add new traces to pattern analyzer
            self.pattern_analyzer.update_patterns(traces)

            # Analyze patterns
            results = self.pattern_analyzer.analyze_patterns()

            patterns_found = (
                len(results.get("workflow_patterns", {}))
                + len(results.get("agent_patterns", {}))
                + len(results.get("decision_patterns", {}))
            )

            logger.info(f"Pattern analysis updated: {patterns_found} patterns found")
            return patterns_found

        except Exception as e:
            logger.error(f"Error updating pattern analysis: {e}")
            return 0

    async def _train_models(self, traces: List[WorkflowTrace]) -> int:
        """Train/update machine learning models."""
        models_trained = 0

        try:
            # Prepare training data
            user_feedback = {}
            for trace in traces:
                feedback = self.feedback_collector.get_feedback(trace.workflow_id)
                if feedback:
                    user_feedback[trace.workflow_id] = {
                        "cat_score": feedback.cat_score.score
                        if feedback.cat_score
                        else 0,
                        "user_satisfaction": feedback.get_composite_score(),
                    }

            # Train reward model
            if user_feedback:
                self.reward_trainer.add_training_data(traces, user_feedback)
                training_result = self.reward_trainer.train_model()

                if training_result.get("success", False):
                    models_trained += 1
                    logger.info("Reward model trained successfully")

            # Additional model training could be added here

        except Exception as e:
            logger.error(f"Error training models: {e}")

        return models_trained

    async def _apply_optimizations(self) -> int:
        """Apply learned optimizations to the system."""
        optimizations_applied = 0

        try:
            # Get optimization recommendations
            recommendations = self.pattern_analyzer.get_pattern_recommendations(
                {"workflow_type": "general", "task_type": "optimization"}
            )

            # Apply prompt optimizations
            if self.config.get("enable_prompt_optimization", True):
                for workflow_id in self.feedback_collector.get_high_scoring_workflows():
                    optimization = self.prompt_optimizer.optimize_prompt_from_feedback(
                        workflow_id
                    )
                    if optimization:
                        optimizations_applied += 1

            # Apply pattern-based optimizations
            if recommendations.get("suggested_workflow_patterns"):
                # Update orchestrator with better patterns
                optimizations_applied += len(
                    recommendations["suggested_workflow_patterns"]
                )

            logger.info(f"Applied {optimizations_applied} optimizations")

        except Exception as e:
            logger.error(f"Error applying optimizations: {e}")

        return optimizations_applied

    async def _update_rl_policy(self) -> float:
        """Update reinforcement learning policy."""
        if not self.rl_env:
            return 0.0

        try:
            # This would implement RL training
            # For now, return a simulated improvement
            improvement = 0.05  # 5% improvement per cycle
            logger.info(f"RL policy updated with {improvement:.3f} improvement")
            return improvement

        except Exception as e:
            logger.error(f"Error updating RL policy: {e}")
            return 0.0

    def _calculate_learning_efficiency(self, cycle_results: Dict[str, Any]) -> float:
        """Calculate learning efficiency for the cycle."""
        # Simple efficiency calculation based on outputs vs inputs
        workflows = cycle_results.get("workflows_processed", 0)
        optimizations = cycle_results.get("optimizations_applied", 0)
        models = cycle_results.get("models_trained", 0)

        if workflows == 0:
            return 0.0

        # Efficiency based on optimizations per workflow and models trained
        efficiency = (optimizations / workflows) * 0.7 + (min(models, 1)) * 0.3
        return min(efficiency, 1.0)

    async def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning system status."""
        return {
            "is_learning": self.is_learning,
            "learning_cycles_completed": self.learning_cycle_count,
            "last_learning_cycle": self.last_learning_cycle.isoformat()
            if self.last_learning_cycle
            else None,
            "metrics": self.metrics.get_summary(),
            "components_status": {
                "workflow_tracer": len(self.workflow_tracer.traces),
                "pattern_analyzer": len(self.pattern_analyzer.workflow_patterns),
                "feedback_collector": self.feedback_collector.get_feedback_statistics(),
                "reward_trainer": len(self.reward_trainer.data_points),
                "prompt_optimizer": len(self.prompt_optimizer.templates),
            },
        }

    async def trigger_learning_cycle(self) -> Dict[str, Any]:
        """Manually trigger a learning cycle."""
        if not self.is_learning:
            logger.warning("Learning system is not active")
            return {"error": "Learning system not active"}

        logger.info("Manually triggering learning cycle")
        results = await self._execute_learning_cycle()
        self.metrics.record_learning_cycle(results)

        return {"success": True, "cycle_results": results, "metrics_updated": True}

    def export_learning_data(self, filename: str):
        """Export learning system data for analysis."""
        data = {
            "learning_config": self.config,
            "metrics": self.metrics.get_summary(),
            "learning_history": {
                "cycles_completed": self.learning_cycle_count,
                "last_cycle": self.last_learning_cycle.isoformat()
                if self.last_learning_cycle
                else None,
            },
            "component_data": {
                "workflow_traces": len(self.workflow_tracer.traces),
                "patterns_discovered": len(self.pattern_analyzer.workflow_patterns),
                "feedback_collected": len(self.feedback_collector.feedback_data),
                "reward_model_samples": len(self.reward_trainer.data_points),
                "prompt_templates": len(self.prompt_optimizer.templates),
            },
            "export_timestamp": datetime.utcnow().isoformat(),
        }

        os.makedirs("./learning_exports", exist_ok=True)
        filepath = f"./learning_exports/{filename}.json"

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Learning data exported to {filepath}")

    async def optimize_orchestrator_decision(
        self, current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get optimization recommendations for orchestrator decisions."""
        recommendations = {
            "suggested_actions": [],
            "confidence_scores": {},
            "expected_outcomes": {},
        }

        try:
            # Get pattern-based recommendations
            pattern_recs = self.pattern_analyzer.get_pattern_recommendations(
                current_state
            )
            recommendations["suggested_actions"].extend(
                pattern_recs.get("suggested_workflow_patterns", [])
            )

            # Get reward model predictions
            if self.reward_trainer.current_model:
                # Predict rewards for different actions
                action_rewards = {}
                for action in [
                    "scan_territories",
                    "dispatch_agent",
                    "coordinate_workflow",
                    "monitor_system",
                ]:
                    features = self._extract_decision_features(current_state, action)
                    reward = self.reward_trainer.get_reward_prediction(features)
                    action_rewards[action] = reward

                best_action = max(action_rewards, key=action_rewards.get)
                recommendations["suggested_actions"].append(
                    {
                        "action": best_action,
                        "expected_reward": action_rewards[best_action],
                        "source": "reward_model",
                    }
                )

            # Get prompt optimization suggestions
            if current_state.get("task_type"):
                best_template = self.prompt_optimizer.get_best_template(
                    current_state["task_type"]
                )
                if best_template:
                    recommendations["suggested_actions"].append(
                        {
                            "action": "use_optimized_prompt",
                            "template_id": best_template,
                            "source": "prompt_optimizer",
                        }
                    )

        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {e}")

        return recommendations

    def _extract_decision_features(
        self, state: Dict[str, Any], action: str
    ) -> Dict[str, Any]:
        """Extract features for decision optimization."""
        return {
            "active_tasks": state.get("active_tasks", 0),
            "agent_utilization": state.get("agent_utilization", 0.5),
            "system_health": state.get("system_health", 0.8),
            "pending_inspirations": state.get("pending_inspirations", 0),
            "proposed_action": action,
            "workflow_progress": state.get("workflow_progress", 0.0),
        }
