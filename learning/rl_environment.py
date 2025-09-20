"""
Reinforcement Learning Environment for Sentinel Orchestrator

This module implements a Gymnasium environment that models the decision-making
process of the Sentinel Orchestrator, allowing reinforcement learning agents
to learn optimal orchestration strategies.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import logging

from ..agents.sentinel.sentinel_orchestrator import SentinelOrchestrator
from .workflow_tracer import WorkflowTracer
from .pattern_analyzer import PatternAnalyzer

logger = logging.getLogger(__name__)


class OrchestrationAction(Enum):
    """Possible actions the orchestrator can take."""

    SCAN_TERRITORIES = 0
    DISPATCH_SINGLE_AGENT = 1
    DISPATCH_MULTIPLE_AGENTS = 2
    COORDINATE_WORKFLOW = 3
    MONITOR_SYSTEM = 4
    OPTIMIZE_RESOURCES = 5
    REQUEST_INSPIRATION = 6
    COMPLETE_TASK = 7


class OrchestratorRLEnvironment(gym.Env):
    """
    Reinforcement Learning Environment for Sentinel Orchestrator.

    The environment models the orchestration decision-making process where:
    - State: Current system status, active tasks, agent availability, workflow progress
    - Actions: Different orchestration decisions (scan, dispatch, coordinate, etc.)
    - Rewards: Based on task completion, efficiency, agent utilization, user satisfaction
    """

    def __init__(
        self,
        orchestrator: SentinelOrchestrator,
        workflow_tracer: WorkflowTracer,
        pattern_analyzer: PatternAnalyzer,
        max_steps: int = 100,
    ):
        """
        Initialize the RL environment.

        Args:
            orchestrator: The Sentinel Orchestrator instance
            workflow_tracer: Workflow tracing system
            pattern_analyzer: Pattern analysis system
            max_steps: Maximum steps per episode
        """
        super().__init__()

        self.orchestrator = orchestrator
        self.workflow_tracer = workflow_tracer
        self.pattern_analyzer = pattern_analyzer
        self.max_steps = max_steps

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(len(OrchestrationAction))

        # Observation space: system state vector
        # [active_tasks, agent_utilization, workflow_progress, system_health,
        #  pending_inspirations, coordination_complexity, time_pressure]
        obs_dim = 7
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Environment state
        self.current_step = 0
        self.episode_reward = 0.0
        self.current_state = self._get_initial_state()
        self.pending_tasks = []
        self.completed_tasks = []
        self.active_workflows = {}

        # Performance tracking
        self.task_completion_rate = 0.0
        self.agent_utilization = 0.0
        self.system_efficiency = 0.0

        logger.info("Initialized Orchestrator RL Environment")

    def _get_initial_state(self) -> np.ndarray:
        """Get the initial environment state."""
        return np.array(
            [
                0.0,  # active_tasks (0-1 normalized)
                0.5,  # agent_utilization (0-1)
                0.0,  # workflow_progress (0-1)
                0.8,  # system_health (0-1)
                0.0,  # pending_inspirations (0-1)
                0.0,  # coordination_complexity (0-1)
                0.0,  # time_pressure (0-1)
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.episode_reward = 0.0
        self.current_state = self._get_initial_state()
        self.pending_tasks = []
        self.completed_tasks = []
        self.active_workflows = {}

        # Reset performance metrics
        self.task_completion_rate = 0.0
        self.agent_utilization = 0.0
        self.system_efficiency = 0.0

        logger.info("Environment reset")
        return self.current_state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (OrchestrationAction enum value)

        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        self.current_step += 1

        # Execute the action
        action_enum = OrchestrationAction(action)
        reward = self._execute_action(action_enum)

        # Update environment state
        next_state = self._update_state(action_enum)

        # Check termination conditions
        terminated = self._is_terminal()
        truncated = self.current_step >= self.max_steps

        # Calculate final reward components
        reward += self._calculate_step_reward()

        self.episode_reward += reward

        info = {
            "step": self.current_step,
            "action": action_enum.name,
            "task_completion_rate": self.task_completion_rate,
            "agent_utilization": self.agent_utilization,
            "system_efficiency": self.system_efficiency,
        }

        return next_state, reward, terminated, truncated, info

    def _execute_action(self, action: OrchestrationAction) -> float:
        """Execute the chosen orchestration action."""
        base_reward = 0.0

        try:
            if action == OrchestrationAction.SCAN_TERRITORIES:
                base_reward = self._action_scan_territories()
            elif action == OrchestrationAction.DISPATCH_SINGLE_AGENT:
                base_reward = self._action_dispatch_single_agent()
            elif action == OrchestrationAction.DISPATCH_MULTIPLE_AGENTS:
                base_reward = self._action_dispatch_multiple_agents()
            elif action == OrchestrationAction.COORDINATE_WORKFLOW:
                base_reward = self._action_coordinate_workflow()
            elif action == OrchestrationAction.MONITOR_SYSTEM:
                base_reward = self._action_monitor_system()
            elif action == OrchestrationAction.OPTIMIZE_RESOURCES:
                base_reward = self._action_optimize_resources()
            elif action == OrchestrationAction.REQUEST_INSPIRATION:
                base_reward = self._action_request_inspiration()
            elif action == OrchestrationAction.COMPLETE_TASK:
                base_reward = self._action_complete_task()

        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
            base_reward = -0.5  # Penalty for errors

        return base_reward

    def _action_scan_territories(self) -> float:
        """Execute scan territories action."""
        # Simulate scanning for creative territories
        scan_result = self.orchestrator.tools[3]._run(
            "comprehensive scan"
        )  # CreativeTerritoryScannerTool

        # Reward based on territories found
        territories_found = len(scan_result.split("\n")) - 1  # Rough estimate
        reward = min(territories_found * 0.1, 1.0)

        # Add new pending tasks based on scan
        if territories_found > 0:
            self.pending_tasks.extend(
                [f"territory_{i}" for i in range(territories_found)]
            )

        return reward

    def _action_dispatch_single_agent(self) -> float:
        """Execute dispatch single agent action."""
        if not self.pending_tasks:
            return -0.2  # Penalty for no tasks to dispatch

        task = self.pending_tasks.pop(0)

        # Get pattern recommendations for optimal agent
        recommendations = self.pattern_analyzer.get_pattern_recommendations(
            {"task_type": task.split("_")[0], "workflow_type": "single_agent"}
        )

        if recommendations["suggested_agent_combinations"]:
            optimal_agents = recommendations["suggested_agent_combinations"][0][
                "agents"
            ]
            agent_name = (
                optimal_agents[0] if optimal_agents else "Atlas_Relational_Analyst"
            )
        else:
            agent_name = "Atlas_Relational_Analyst"  # Default

        # Simulate agent dispatch
        try:
            # This would normally dispatch via A2A protocol
            self.active_workflows[task] = {
                "agent": agent_name,
                "start_time": self.current_step,
                "status": "active",
            }
            return 0.3  # Reward for successful dispatch
        except Exception:
            return -0.3  # Penalty for failed dispatch

    def _action_dispatch_multiple_agents(self) -> float:
        """Execute dispatch multiple agents action."""
        if len(self.pending_tasks) < 2:
            return -0.2  # Need multiple tasks for this action

        tasks = self.pending_tasks[:2]
        self.pending_tasks = self.pending_tasks[2:]

        # Get pattern recommendations
        recommendations = self.pattern_analyzer.get_pattern_recommendations(
            {"task_type": "multi_agent", "workflow_type": "coordination"}
        )

        if recommendations["suggested_agent_combinations"]:
            optimal_agents = recommendations["suggested_agent_combinations"][0][
                "agents"
            ]
        else:
            optimal_agents = ["Atlas_Relational_Analyst", "ComparativeMythologyAgent"]

        # Simulate multi-agent dispatch
        for i, task in enumerate(tasks):
            agent_name = optimal_agents[i % len(optimal_agents)]
            self.active_workflows[task] = {
                "agent": agent_name,
                "start_time": self.current_step,
                "status": "active",
            }

        return 0.5  # Higher reward for multi-agent coordination

    def _action_coordinate_workflow(self) -> float:
        """Execute coordinate workflow action."""
        if not self.active_workflows:
            return -0.1  # No active workflows to coordinate

        # Simulate workflow coordination
        coordination_result = self.orchestrator.tools[1]._run(
            "optimize active workflows"
        )  # WorkflowManagementTool

        # Reward based on coordination success
        if "optimization" in coordination_result.lower():
            return 0.4
        else:
            return 0.1

    def _action_monitor_system(self) -> float:
        """Execute monitor system action."""
        # Simulate system monitoring
        health_report = self.orchestrator.tools[2]._run(
            "generate health report"
        )  # SystemMonitoringTool

        # Reward for monitoring (always positive, encourages regular monitoring)
        return 0.2

    def _action_optimize_resources(self) -> float:
        """Execute optimize resources action."""
        # Simulate resource optimization
        if len(self.active_workflows) > 3:
            # High load - optimization needed
            return 0.6
        else:
            # Low load - less benefit
            return 0.1

    def _action_request_inspiration(self) -> float:
        """Execute request inspiration action."""
        if not self.pending_tasks:
            return -0.1

        # Simulate inspiration request
        try:
            # This would normally use A2A protocol
            inspiration_received = np.random.random() > 0.3  # 70% success rate
            if inspiration_received:
                return 0.4
            else:
                return 0.1
        except Exception:
            return -0.2

    def _action_complete_task(self) -> float:
        """Execute complete task action."""
        if not self.active_workflows:
            return -0.3  # No active tasks to complete

        # Complete a random active task
        task_to_complete = np.random.choice(list(self.active_workflows.keys()))
        workflow = self.active_workflows.pop(task_to_complete)

        # Calculate completion reward based on efficiency
        duration = self.current_step - workflow["start_time"]
        efficiency_bonus = max(0, 1.0 - (duration / 20.0))  # Bonus for quick completion

        self.completed_tasks.append(task_to_complete)
        return 0.8 + efficiency_bonus  # High reward for task completion

    def _update_state(self, action: OrchestrationAction) -> np.ndarray:
        """Update the environment state based on the action taken."""
        # Update state components
        active_tasks = len(self.pending_tasks) + len(self.active_workflows)
        agent_utilization = min(
            len(self.active_workflows) / 5.0, 1.0
        )  # Max 5 concurrent tasks
        workflow_progress = len(self.completed_tasks) / max(
            len(self.completed_tasks) + active_tasks, 1
        )
        system_health = 0.8 + np.random.normal(0, 0.1)  # Base health with noise
        pending_inspirations = len(
            [t for t in self.pending_tasks if "inspiration" in t]
        ) / max(len(self.pending_tasks), 1)
        coordination_complexity = min(len(self.active_workflows) / 3.0, 1.0)
        time_pressure = min(self.current_step / self.max_steps, 1.0)

        # Normalize values
        system_health = np.clip(system_health, 0.0, 1.0)

        new_state = np.array(
            [
                min(active_tasks / 10.0, 1.0),  # Normalize active tasks
                agent_utilization,
                workflow_progress,
                system_health,
                pending_inspirations,
                coordination_complexity,
                time_pressure,
            ],
            dtype=np.float32,
        )

        self.current_state = new_state
        return new_state

    def _calculate_step_reward(self) -> float:
        """Calculate additional reward components for the current step."""
        reward = 0.0

        # Efficiency reward
        if self.agent_utilization > 0.7:
            reward += 0.1  # Bonus for high utilization
        elif self.agent_utilization < 0.3:
            reward -= 0.1  # Penalty for low utilization

        # Task completion reward
        if self.completed_tasks and self.current_step > 0:
            completion_rate = len(self.completed_tasks) / self.current_step
            if completion_rate > 0.5:
                reward += 0.2

        # System health maintenance
        if self.current_state[3] > 0.8:  # system_health
            reward += 0.05

        return reward

    def _is_terminal(self) -> bool:
        """Check if the episode should terminate."""
        # Terminate if all tasks completed or system health critical
        all_tasks_done = (
            len(self.pending_tasks) == 0 and len(self.active_workflows) == 0
        )
        system_critical = self.current_state[3] < 0.2  # system_health

        return all_tasks_done or system_critical

    def render(self, mode="human"):
        """Render the current environment state."""
        print(f"Step: {self.current_step}")
        print(f"Active Tasks: {len(self.pending_tasks) + len(self.active_workflows)}")
        print(f"Completed Tasks: {len(self.completed_tasks)}")
        print(f"Agent Utilization: {self.agent_utilization:.2f}")
        print(f"System Health: {self.current_state[3]:.2f}")
        print(f"Episode Reward: {self.episode_reward:.2f}")
        print("-" * 40)

    def close(self):
        """Clean up environment resources."""
        logger.info("Environment closed")


class OrchestratorRewardWrapper(gym.RewardWrapper):
    """
    Reward wrapper that modifies rewards based on learning objectives.

    This wrapper can amplify certain reward signals or add custom reward shaping
    based on the learning goals of the orchestrator.
    """

    def __init__(self, env: OrchestratorRLEnvironment, reward_config: Dict[str, float]):
        super().__init__(env)
        self.reward_config = reward_config

    def reward(self, reward: float) -> float:
        """Modify the reward based on configuration."""
        # Apply reward shaping based on config
        shaped_reward = reward

        # Amplify task completion rewards
        if reward > 0.5:  # Likely a task completion
            shaped_reward *= self.reward_config.get("task_completion_multiplier", 1.2)

        # Penalize inefficient actions more heavily
        if reward < -0.1:
            shaped_reward *= self.reward_config.get(
                "inefficiency_penalty_multiplier", 1.5
            )

        return shaped_reward


def create_orchestrator_env(
    orchestrator: SentinelOrchestrator,
    workflow_tracer: WorkflowTracer,
    pattern_analyzer: PatternAnalyzer,
    reward_config: Optional[Dict[str, float]] = None,
) -> gym.Env:
    """
    Factory function to create a configured orchestrator environment.

    Args:
        orchestrator: The Sentinel Orchestrator instance
        workflow_tracer: Workflow tracing system
        pattern_analyzer: Pattern analysis system
        reward_config: Optional reward shaping configuration

    Returns:
        Configured Gymnasium environment
    """
    env = OrchestratorRLEnvironment(orchestrator, workflow_tracer, pattern_analyzer)

    if reward_config:
        env = OrchestratorRewardWrapper(env, reward_config)

    return env
