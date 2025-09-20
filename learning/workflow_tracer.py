"""
Workflow Tracer for LangGraph Executions

This module captures and analyzes workflow traces from LangGraph executions,
enabling the learning system to understand successful patterns and optimize
future agent coordination.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from collections import defaultdict
import hashlib

from langgraph.graph import StateGraph
from langchain.callbacks.base import BaseCallbackHandler

logger = logging.getLogger(__name__)


class WorkflowTrace:
    """Represents a complete workflow execution trace."""

    def __init__(self, workflow_id: str, workflow_type: str):
        self.workflow_id = workflow_id
        self.workflow_type = workflow_type
        self.start_time = datetime.utcnow()
        self.end_time: Optional[datetime] = None
        self.duration: Optional[float] = None
        self.nodes_executed: List[Dict[str, Any]] = []
        self.edges_traversed: List[Dict[str, Any]] = []
        self.state_transitions: List[Dict[str, Any]] = []
        self.agent_interactions: List[Dict[str, Any]] = []
        self.final_state: Optional[Dict[str, Any]] = None
        self.success_metrics: Dict[str, Any] = {}
        self.error_info: Optional[Dict[str, Any]] = None
        self.metadata: Dict[str, Any] = {}

    def add_node_execution(
        self,
        node_name: str,
        input_state: Dict[str, Any],
        output_state: Dict[str, Any],
        execution_time: float,
    ):
        """Add a node execution to the trace."""
        self.nodes_executed.append(
            {
                "node_name": node_name,
                "input_state": input_state,
                "output_state": output_state,
                "execution_time": execution_time,
                "timestamp": datetime.utcnow(),
            }
        )

    def add_edge_traversal(
        self, from_node: str, to_node: str, condition: Optional[str] = None
    ):
        """Add an edge traversal to the trace."""
        self.edges_traversed.append(
            {
                "from_node": from_node,
                "to_node": to_node,
                "condition": condition,
                "timestamp": datetime.utcnow(),
            }
        )

    def add_agent_interaction(
        self, agent_name: str, task: str, response: Any, response_time: float
    ):
        """Add an agent interaction to the trace."""
        self.agent_interactions.append(
            {
                "agent_name": agent_name,
                "task": task,
                "response": str(response)[:500],  # Truncate for storage
                "response_time": response_time,
                "timestamp": datetime.utcnow(),
            }
        )

    def complete_trace(
        self,
        final_state: Dict[str, Any],
        success: bool = True,
        error_info: Optional[Dict[str, Any]] = None,
    ):
        """Complete the workflow trace."""
        self.end_time = datetime.utcnow()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.final_state = final_state

        if success:
            self._calculate_success_metrics()
        else:
            self.error_info = error_info

    def _calculate_success_metrics(self):
        """Calculate success metrics for the workflow."""
        if not self.final_state:
            return

        # Calculate various success metrics
        self.success_metrics = {
            "total_nodes": len(self.nodes_executed),
            "total_edges": len(self.edges_traversed),
            "total_agent_interactions": len(self.agent_interactions),
            "avg_node_execution_time": sum(
                n["execution_time"] for n in self.nodes_executed
            )
            / max(len(self.nodes_executed), 1),
            "avg_agent_response_time": sum(
                a["response_time"] for a in self.agent_interactions
            )
            / max(len(self.agent_interactions), 1),
            "workflow_efficiency": self._calculate_efficiency_score(),
            "agent_coordination_score": self._calculate_coordination_score(),
        }

    def _calculate_efficiency_score(self) -> float:
        """Calculate workflow efficiency score."""
        if not self.duration or self.duration == 0:
            return 0.0

        # Efficiency based on nodes executed vs time
        base_efficiency = len(self.nodes_executed) / self.duration

        # Penalize for excessive agent interactions
        interaction_penalty = (
            max(0, len(self.agent_interactions) - len(self.nodes_executed)) * 0.1
        )

        return max(0, base_efficiency - interaction_penalty)

    def _calculate_coordination_score(self) -> float:
        """Calculate agent coordination score."""
        if not self.agent_interactions:
            return 1.0  # Perfect if no external agents needed

        # Score based on response times and interaction patterns
        avg_response_time = sum(
            a["response_time"] for a in self.agent_interactions
        ) / len(self.agent_interactions)

        # Lower response times are better
        time_score = max(0, 1.0 - (avg_response_time / 60.0))  # Penalize > 60 seconds

        # Diversity in agent usage (using multiple agents is good)
        unique_agents = len(set(a["agent_name"] for a in self.agent_interactions))
        diversity_score = min(1.0, unique_agents / 3.0)  # Max score at 3+ agents

        return (time_score + diversity_score) / 2.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for storage."""
        return {
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "nodes_executed": self.nodes_executed,
            "edges_traversed": self.edges_traversed,
            "state_transitions": self.state_transitions,
            "agent_interactions": self.agent_interactions,
            "final_state": self.final_state,
            "success_metrics": self.success_metrics,
            "error_info": self.error_info,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowTrace":
        """Create trace from dictionary."""
        trace = cls(data["workflow_id"], data["workflow_type"])
        trace.start_time = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            trace.end_time = datetime.fromisoformat(data["end_time"])
        trace.duration = data.get("duration")
        trace.nodes_executed = data.get("nodes_executed", [])
        trace.edges_traversed = data.get("edges_traversed", [])
        trace.state_transitions = data.get("state_transitions", [])
        trace.agent_interactions = data.get("agent_interactions", [])
        trace.final_state = data.get("final_state")
        trace.success_metrics = data.get("success_metrics", {})
        trace.error_info = data.get("error_info")
        trace.metadata = data.get("metadata", {})
        return trace


class LangGraphCallbackHandler(BaseCallbackHandler):
    """Callback handler for capturing LangGraph workflow executions."""

    def __init__(self, tracer: "WorkflowTracer"):
        self.tracer = tracer
        self.current_trace: Optional[WorkflowTrace] = None
        self.node_start_times: Dict[str, datetime] = {}

    def on_workflow_start(self, workflow_id: str, workflow_type: str, **kwargs):
        """Called when a workflow starts."""
        self.current_trace = WorkflowTrace(workflow_id, workflow_type)
        logger.info(f"Started tracing workflow: {workflow_id}")

    def on_node_start(self, node_name: str, input_state: Dict[str, Any], **kwargs):
        """Called when a node execution starts."""
        if self.current_trace:
            self.node_start_times[node_name] = datetime.utcnow()

    def on_node_end(self, node_name: str, output_state: Dict[str, Any], **kwargs):
        """Called when a node execution ends."""
        if self.current_trace and node_name in self.node_start_times:
            start_time = self.node_start_times[node_name]
            execution_time = (datetime.utcnow() - start_time).total_seconds()

            # Get input state from previous state or kwargs
            input_state = kwargs.get("input_state", {})

            self.current_trace.add_node_execution(
                node_name, input_state, output_state, execution_time
            )

            del self.node_start_times[node_name]

    def on_edge_traversal(
        self, from_node: str, to_node: str, condition: Optional[str] = None, **kwargs
    ):
        """Called when an edge is traversed."""
        if self.current_trace:
            self.current_trace.add_edge_traversal(from_node, to_node, condition)

    def on_agent_interaction(
        self, agent_name: str, task: str, response: Any, response_time: float, **kwargs
    ):
        """Called when an agent interaction occurs."""
        if self.current_trace:
            self.current_trace.add_agent_interaction(
                agent_name, task, response, response_time
            )

    def on_workflow_end(
        self,
        final_state: Dict[str, Any],
        success: bool = True,
        error_info: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Called when a workflow ends."""
        if self.current_trace:
            self.current_trace.complete_trace(final_state, success, error_info)
            self.tracer.store_trace(self.current_trace)
            logger.info(f"Completed tracing workflow: {self.current_trace.workflow_id}")
            self.current_trace = None


class WorkflowTracer:
    """Main workflow tracing system."""

    def __init__(self, storage_path: str = "./traces"):
        self.storage_path = storage_path
        self.traces: Dict[str, WorkflowTrace] = {}
        self.callback_handlers: Dict[str, LangGraphCallbackHandler] = {}
        self.trace_index: Dict[str, List[str]] = defaultdict(
            list
        )  # workflow_type -> trace_ids

        # Codex integration
        self.codex_manager = None

        # Create storage directory if it doesn't exist
        import os

        os.makedirs(storage_path, exist_ok=True)

    def create_callback_handler(self, workflow_type: str) -> LangGraphCallbackHandler:
        """Create a callback handler for a specific workflow type."""
        handler = LangGraphCallbackHandler(self)
        self.callback_handlers[workflow_type] = handler
        return handler

    def start_trace(self, workflow_id: str, workflow_type: str) -> WorkflowTrace:
        """Start tracing a new workflow."""
        trace = WorkflowTrace(workflow_id, workflow_type)
        self.traces[workflow_id] = trace
        self.trace_index[workflow_type].append(workflow_id)

        # Notify callback handler if it exists
        if workflow_type in self.callback_handlers:
            self.callback_handlers[workflow_type].on_workflow_start(
                workflow_id, workflow_type
            )

        logger.info(f"Started trace for workflow: {workflow_id}")
        return trace

    def set_codex_manager(self, codex_manager):
        """
        Set the Codex manager for archival integration.

        Args:
            codex_manager: CodexManager instance
        """
        self.codex_manager = codex_manager
        logger.info("WorkflowTracer integrated with Codex")

    def store_trace(self, trace: WorkflowTrace):
        """Store a completed trace."""
        self.traces[trace.workflow_id] = trace

        # Save to file
        self._save_trace_to_file(trace)

        # Archive to Codex if available
        if self.codex_manager:
            try:
                self.codex_manager.archive_workflow_trace(trace.to_dict())
            except Exception as e:
                logger.warning(f"Failed to archive workflow trace to Codex: {e}")

        logger.info(f"Stored trace: {trace.workflow_id}")

    def get_trace(self, workflow_id: str) -> Optional[WorkflowTrace]:
        """Get a trace by ID."""
        return self.traces.get(workflow_id)

    def get_traces_by_type(self, workflow_type: str) -> List[WorkflowTrace]:
        """Get all traces for a specific workflow type."""
        trace_ids = self.trace_index.get(workflow_type, [])
        return [self.traces[tid] for tid in trace_ids if tid in self.traces]

    def get_successful_traces(
        self, workflow_type: Optional[str] = None, min_efficiency: float = 0.0
    ) -> List[WorkflowTrace]:
        """Get traces that completed successfully with minimum efficiency."""
        traces = []
        candidates = self.traces.values()

        if workflow_type:
            candidates = self.get_traces_by_type(workflow_type)

        for trace in candidates:
            if (
                trace.error_info is None
                and trace.success_metrics.get("workflow_efficiency", 0)
                >= min_efficiency
            ):
                traces.append(trace)

        return traces

    def get_workflow_patterns(self, workflow_type: str) -> Dict[str, Any]:
        """Extract common patterns from workflow traces."""
        traces = self.get_traces_by_type(workflow_type)
        if not traces:
            return {}

        patterns = {
            "avg_duration": sum(t.duration or 0 for t in traces) / len(traces),
            "avg_nodes": sum(len(t.nodes_executed) for t in traces) / len(traces),
            "avg_agent_interactions": sum(len(t.agent_interactions) for t in traces)
            / len(traces),
            "common_node_sequences": self._extract_common_sequences(traces),
            "success_rate": len([t for t in traces if t.error_info is None])
            / len(traces),
        }

        return patterns

    def _extract_common_sequences(self, traces: List[WorkflowTrace]) -> List[List[str]]:
        """Extract common node execution sequences."""
        sequences = []
        for trace in traces:
            if trace.nodes_executed:
                seq = [node["node_name"] for node in trace.nodes_executed]
                sequences.append(seq)

        # Find most common sequences (simplified implementation)
        if not sequences:
            return []

        # For now, return the most common sequence
        from collections import Counter

        seq_counter = Counter(tuple(seq) for seq in sequences)
        most_common = seq_counter.most_common(1)

        return [list(most_common[0][0])] if most_common else []

    def _save_trace_to_file(self, trace: WorkflowTrace):
        """Save trace to JSON file."""
        import os

        filename = f"{trace.workflow_id}.json"
        filepath = os.path.join(self.storage_path, filename)

        with open(filepath, "w") as f:
            json.dump(trace.to_dict(), f, indent=2, default=str)

    def load_traces_from_files(self):
        """Load traces from storage files."""
        import os
        import glob

        pattern = os.path.join(self.storage_path, "*.json")
        for filepath in glob.glob(pattern):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    trace = WorkflowTrace.from_dict(data)
                    self.traces[trace.workflow_id] = trace
                    self.trace_index[trace.workflow_type].append(trace.workflow_id)
            except Exception as e:
                logger.error(f"Error loading trace from {filepath}: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall tracing statistics."""
        total_traces = len(self.traces)
        successful_traces = len(
            [t for t in self.traces.values() if t.error_info is None]
        )

        return {
            "total_traces": total_traces,
            "successful_traces": successful_traces,
            "success_rate": successful_traces / max(total_traces, 1),
            "workflow_types": list(self.trace_index.keys()),
            "avg_duration": sum((t.duration or 0) for t in self.traces.values())
            / max(total_traces, 1),
        }
