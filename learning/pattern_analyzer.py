"""
Pattern Analyzer for Workflow Traces

This module analyzes successful workflow traces to identify recurring patterns,
optimal agent combinations, and decision-making strategies that can be learned
and applied to future orchestrations.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
import numpy as np
from datetime import datetime, timedelta

from .workflow_tracer import WorkflowTrace, WorkflowTracer

logger = logging.getLogger(__name__)


class WorkflowPattern:
    """Represents a discovered workflow pattern."""

    def __init__(self, pattern_id: str, pattern_type: str):
        self.pattern_id = pattern_id
        self.pattern_type = pattern_type
        self.node_sequence: List[str] = []
        self.agent_sequence: List[str] = []
        self.conditions: List[Optional[str]] = []
        self.success_rate: float = 0.0
        self.avg_duration: float = 0.0
        self.avg_efficiency: float = 0.0
        self.frequency: int = 0
        self.supporting_traces: List[str] = []
        self.metadata: Dict[str, Any] = {}

    def add_occurrence(self, trace: WorkflowTrace):
        """Add a trace that matches this pattern."""
        self.frequency += 1
        self.supporting_traces.append(trace.workflow_id)

        # Update metrics
        if trace.success_metrics:
            self.avg_efficiency = (
                (self.avg_efficiency * (self.frequency - 1))
                + trace.success_metrics.get("workflow_efficiency", 0)
            ) / self.frequency
            self.avg_duration = (
                (self.avg_duration * (self.frequency - 1)) + (trace.duration or 0)
            ) / self.frequency

        # Update success rate
        successful_traces = sum(
            1
            for tid in self.supporting_traces
            if WorkflowTracer().get_trace(tid)
            and WorkflowTracer().get_trace(tid).error_info is None
        )
        self.success_rate = successful_traces / len(self.supporting_traces)

    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "node_sequence": self.node_sequence,
            "agent_sequence": self.agent_sequence,
            "conditions": self.conditions,
            "success_rate": self.success_rate,
            "avg_duration": self.avg_duration,
            "avg_efficiency": self.avg_efficiency,
            "frequency": self.frequency,
            "supporting_traces": self.supporting_traces,
            "metadata": self.metadata,
        }


class AgentCombinationPattern:
    """Represents optimal agent combinations for specific tasks."""

    def __init__(self, task_type: str):
        self.task_type = task_type
        self.agent_combinations: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        self.best_combination: Optional[Tuple[str, ...]] = None
        self.best_score: float = 0.0

    def add_combination(
        self,
        agents: Tuple[str, ...],
        performance_score: float,
        success_rate: float,
        avg_duration: float,
    ):
        """Add an agent combination with its performance metrics."""
        self.agent_combinations[agents] = {
            "performance_score": performance_score,
            "success_rate": success_rate,
            "avg_duration": avg_duration,
            "frequency": self.agent_combinations.get(agents, {}).get("frequency", 0)
            + 1,
        }

        # Update best combination
        combined_score = (
            performance_score * 0.4
            + success_rate * 0.4
            + (1.0 / max(avg_duration, 0.1)) * 0.2
        )  # Efficiency bonus

        if combined_score > self.best_score:
            self.best_score = combined_score
            self.best_combination = agents

    def get_optimal_combination(self, max_agents: int = 3) -> Tuple[str, ...]:
        """Get the optimal agent combination for this task type."""
        if self.best_combination and len(self.best_combination) <= max_agents:
            return self.best_combination

        # Find best combination within size limit
        candidates = [
            (agents, metrics)
            for agents, metrics in self.agent_combinations.items()
            if len(agents) <= max_agents
        ]

        if not candidates:
            return tuple()

        # Sort by combined score
        candidates.sort(
            key=lambda x: (
                x[1]["performance_score"] * 0.4
                + x[1]["success_rate"] * 0.4
                + (1.0 / max(x[1]["avg_duration"], 0.1)) * 0.2
            ),
            reverse=True,
        )

        return candidates[0][0]


class DecisionPattern:
    """Represents decision-making patterns in workflow orchestration."""

    def __init__(self, decision_point: str):
        self.decision_point = decision_point
        self.conditions_to_actions: Dict[str, Dict[str, Any]] = {}
        self.action_success_rates: Dict[str, float] = {}
        self.context_patterns: Dict[str, List[str]] = defaultdict(list)

    def add_decision(self, condition: str, action: str, success: bool, context: str):
        """Add a decision instance."""
        if condition not in self.conditions_to_actions:
            self.conditions_to_actions[condition] = {
                "actions": defaultdict(int),
                "successes": defaultdict(int),
                "total": 0,
            }

        self.conditions_to_actions[condition]["actions"][action] += 1
        self.conditions_to_actions[condition]["total"] += 1

        if success:
            self.conditions_to_actions[condition]["successes"][action] += 1

        # Update context patterns
        self.context_patterns[condition].append(context)

        # Update success rates
        success_count = self.conditions_to_actions[condition]["successes"][action]
        total_count = self.conditions_to_actions[condition]["actions"][action]
        self.action_success_rates[f"{condition}:{action}"] = success_count / total_count

    def get_best_action(self, condition: str) -> Optional[str]:
        """Get the best action for a given condition."""
        if condition not in self.conditions_to_actions:
            return None

        actions_data = self.conditions_to_actions[condition]

        # Find action with highest success rate
        best_action = None
        best_rate = 0.0

        for action in actions_data["actions"]:
            success_rate = (
                actions_data["successes"][action] / actions_data["actions"][action]
            )
            if success_rate > best_rate:
                best_rate = success_rate
                best_action = action

        return best_action

    def get_action_probabilities(self, condition: str) -> Dict[str, float]:
        """Get action probabilities for a condition."""
        if condition not in self.conditions_to_actions:
            return {}

        actions_data = self.conditions_to_actions[condition]
        total_actions = sum(actions_data["actions"].values())

        return {
            action: count / total_actions
            for action, count in actions_data["actions"].items()
        }


class PatternAnalyzer:
    """Main pattern analysis system for workflow traces."""

    def __init__(self, workflow_tracer: WorkflowTracer):
        self.workflow_tracer = workflow_tracer
        self.workflow_patterns: Dict[str, WorkflowPattern] = {}
        self.agent_patterns: Dict[str, AgentCombinationPattern] = {}
        self.decision_patterns: Dict[str, DecisionPattern] = {}
        self.pattern_similarity_threshold = 0.8

    def analyze_patterns(
        self, workflow_type: Optional[str] = None, min_frequency: int = 2
    ) -> Dict[str, Any]:
        """
        Analyze patterns from workflow traces.

        Args:
            workflow_type: Specific workflow type to analyze (None for all)
            min_frequency: Minimum frequency for pattern consideration

        Returns:
            Dictionary containing discovered patterns
        """
        logger.info(f"Starting pattern analysis for workflow_type: {workflow_type}")

        # Get relevant traces
        if workflow_type:
            traces = self.workflow_tracer.get_successful_traces(workflow_type)
        else:
            traces = self.workflow_tracer.get_successful_traces()

        if not traces:
            logger.warning("No successful traces found for analysis")
            return {}

        # Analyze different types of patterns
        workflow_patterns = self._analyze_workflow_patterns(traces, min_frequency)
        agent_patterns = self._analyze_agent_patterns(traces)
        decision_patterns = self._analyze_decision_patterns(traces)

        # Calculate pattern quality metrics
        pattern_quality = self._calculate_pattern_quality(
            workflow_patterns, agent_patterns
        )

        results = {
            "workflow_patterns": workflow_patterns,
            "agent_patterns": agent_patterns,
            "decision_patterns": decision_patterns,
            "pattern_quality": pattern_quality,
            "analysis_metadata": {
                "total_traces_analyzed": len(traces),
                "workflow_type": workflow_type,
                "min_frequency": min_frequency,
                "analysis_timestamp": datetime.utcnow().isoformat(),
            },
        }

        logger.info(
            f"Pattern analysis completed. Found {len(workflow_patterns)} workflow patterns, "
            f"{len(agent_patterns)} agent patterns, {len(decision_patterns)} decision patterns"
        )

        return results

    def _analyze_workflow_patterns(
        self, traces: List[WorkflowTrace], min_frequency: int
    ) -> Dict[str, WorkflowPattern]:
        """Analyze workflow execution patterns."""
        pattern_groups = defaultdict(list)

        # Group traces by similar node sequences
        for trace in traces:
            if not trace.nodes_executed:
                continue

            # Create sequence signature
            node_sequence = tuple(node["node_name"] for node in trace.nodes_executed)
            pattern_groups[node_sequence].append(trace)

        # Create patterns from groups that meet minimum frequency
        workflow_patterns = {}
        for sequence, trace_group in pattern_groups.items():
            if len(trace_group) >= min_frequency:
                pattern_id = f"workflow_pattern_{len(workflow_patterns)}"
                pattern = WorkflowPattern(pattern_id, "node_sequence")
                pattern.node_sequence = list(sequence)

                # Add all occurrences
                for trace in trace_group:
                    pattern.add_occurrence(trace)

                workflow_patterns[pattern_id] = pattern

        return workflow_patterns

    def _analyze_agent_patterns(
        self, traces: List[WorkflowTrace]
    ) -> Dict[str, AgentCombinationPattern]:
        """Analyze optimal agent combinations."""
        task_patterns = defaultdict(AgentCombinationPattern)

        for trace in traces:
            if not trace.agent_interactions:
                continue

            # Group by task types (simplified - using agent names as proxy)
            agent_names = tuple(
                sorted(
                    set(
                        interaction["agent_name"]
                        for interaction in trace.agent_interactions
                    )
                )
            )

            if not agent_names:
                continue

            # Use first agent as task type indicator (simplified)
            task_type = (
                trace.agent_interactions[0]["agent_name"].split("_")[0]
                if "_" in trace.agent_interactions[0]["agent_name"]
                else "general"
            )

            if task_type not in task_patterns:
                task_patterns[task_type] = AgentCombinationPattern(task_type)

            # Calculate performance metrics for this combination
            performance_score = trace.success_metrics.get("workflow_efficiency", 0.5)
            success_rate = 1.0 if trace.error_info is None else 0.0
            avg_duration = trace.duration or 60.0

            task_patterns[task_type].add_combination(
                agent_names, performance_score, success_rate, avg_duration
            )

        return dict(task_patterns)

    def _analyze_decision_patterns(
        self, traces: List[WorkflowTrace]
    ) -> Dict[str, DecisionPattern]:
        """Analyze decision-making patterns."""
        decision_patterns = {}

        for trace in traces:
            # Extract decision points from edges with conditions
            for edge in trace.edges_traversed:
                if edge.get("condition"):
                    decision_point = f"{edge['from_node']}_to_{edge['to_node']}"

                    if decision_point not in decision_patterns:
                        decision_patterns[decision_point] = DecisionPattern(
                            decision_point
                        )

                    # Determine success based on final state
                    success = trace.error_info is None

                    # Use condition as context
                    context = edge["condition"]

                    decision_patterns[decision_point].add_decision(
                        edge["condition"], edge["to_node"], success, context
                    )

        return decision_patterns

    def _calculate_pattern_quality(
        self,
        workflow_patterns: Dict[str, WorkflowPattern],
        agent_patterns: Dict[str, AgentCombinationPattern],
    ) -> Dict[str, Any]:
        """Calculate overall quality metrics for discovered patterns."""
        if not workflow_patterns and not agent_patterns:
            return {"overall_quality": 0.0}

        # Calculate workflow pattern quality
        workflow_quality = 0.0
        if workflow_patterns:
            avg_success_rate = np.mean(
                [p.success_rate for p in workflow_patterns.values()]
            )
            avg_frequency = np.mean([p.frequency for p in workflow_patterns.values()])
            avg_efficiency = np.mean(
                [p.avg_efficiency for p in workflow_patterns.values()]
            )

            workflow_quality = (
                avg_success_rate * 0.4
                + avg_efficiency * 0.4
                + min(avg_frequency / 10, 1.0) * 0.2
            )

        # Calculate agent pattern quality
        agent_quality = 0.0
        if agent_patterns:
            best_scores = [
                p.best_score for p in agent_patterns.values() if p.best_score > 0
            ]
            if best_scores:
                agent_quality = np.mean(best_scores)

        overall_quality = (workflow_quality + agent_quality) / 2

        return {
            "overall_quality": overall_quality,
            "workflow_pattern_quality": workflow_quality,
            "agent_pattern_quality": agent_quality,
            "total_patterns": len(workflow_patterns) + len(agent_patterns),
        }

    def get_optimal_workflow_pattern(
        self, workflow_type: str
    ) -> Optional[WorkflowPattern]:
        """Get the optimal workflow pattern for a given type."""
        patterns = [
            p
            for p in self.workflow_patterns.values()
            if p.pattern_type == workflow_type
        ]

        if not patterns:
            return None

        # Return pattern with highest combined score
        return max(
            patterns,
            key=lambda p: p.success_rate * 0.5
            + p.avg_efficiency * 0.3
            + min(p.frequency / 10, 1.0) * 0.2,
        )

    def get_optimal_agent_combination(
        self, task_type: str, max_agents: int = 3
    ) -> Tuple[str, ...]:
        """Get the optimal agent combination for a task type."""
        if task_type not in self.agent_patterns:
            return tuple()

        return self.agent_patterns[task_type].get_optimal_combination(max_agents)

    def predict_workflow_success(
        self, workflow_pattern: List[str], agent_combination: Tuple[str, ...]
    ) -> float:
        """Predict success probability for a proposed workflow configuration."""
        # Find similar patterns
        similar_patterns = []
        for pattern in self.workflow_patterns.values():
            if pattern.node_sequence == workflow_pattern:
                similar_patterns.append(pattern)

        if not similar_patterns:
            return 0.5  # Default probability

        # Use the most similar pattern's success rate
        best_pattern = max(similar_patterns, key=lambda p: p.frequency)
        base_probability = best_pattern.success_rate

        # Adjust based on agent combination
        if agent_combination and best_pattern.agent_sequence:
            agent_overlap = len(
                set(agent_combination) & set(best_pattern.agent_sequence)
            )
            agent_bonus = agent_overlap / max(
                len(agent_combination), len(best_pattern.agent_sequence)
            )
            base_probability = min(1.0, base_probability + (agent_bonus * 0.2))

        return base_probability

    def get_pattern_recommendations(
        self, current_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get pattern-based recommendations for current context."""
        recommendations = {
            "suggested_workflow_patterns": [],
            "suggested_agent_combinations": [],
            "decision_recommendations": {},
            "confidence_scores": {},
        }

        # Get workflow type from context
        workflow_type = current_context.get("workflow_type", "general")

        # Recommend optimal workflow pattern
        optimal_pattern = self.get_optimal_workflow_pattern(workflow_type)
        if optimal_pattern:
            recommendations["suggested_workflow_patterns"].append(
                {
                    "pattern_id": optimal_pattern.pattern_id,
                    "node_sequence": optimal_pattern.node_sequence,
                    "success_rate": optimal_pattern.success_rate,
                    "avg_efficiency": optimal_pattern.avg_efficiency,
                }
            )

        # Recommend optimal agent combination
        task_type = current_context.get("task_type", "general")
        optimal_agents = self.get_optimal_agent_combination(task_type)
        if optimal_agents:
            recommendations["suggested_agent_combinations"].append(
                {
                    "task_type": task_type,
                    "agents": list(optimal_agents),
                    "expected_performance": self.agent_patterns[task_type].best_score,
                }
            )

        # Get decision recommendations
        for decision_point, pattern in self.decision_patterns.items():
            condition = current_context.get("current_condition", "default")
            best_action = pattern.get_best_action(condition)
            if best_action:
                recommendations["decision_recommendations"][decision_point] = {
                    "condition": condition,
                    "recommended_action": best_action,
                    "success_rate": pattern.action_success_rates.get(
                        f"{condition}:{best_action}", 0.0
                    ),
                }

        return recommendations

    def update_patterns(self, new_traces: List[WorkflowTrace]):
        """Update patterns with new trace data."""
        logger.info(f"Updating patterns with {len(new_traces)} new traces")

        # Re-analyze patterns with new data
        self.analyze_patterns()

        logger.info("Pattern update completed")
