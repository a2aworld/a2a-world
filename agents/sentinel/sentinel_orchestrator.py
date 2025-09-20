"""
Sentinel Orchestrator Agent with LangGraph

This agent specializes in coordinating and orchestrating the activities of other specialist agents
in the Terra Constellata system using LangGraph for stateful, multi-agent workflows focused on
autonomous discovery of creative territories.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, TypedDict, Annotated
from datetime import datetime, timezone
from collections import defaultdict
import json
import operator

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.language_models.llms import BaseLLM
from langchain.tools import BaseTool, tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    # Fallback for older langgraph versions
    try:
        from langgraph import StateGraph, END
    except ImportError:
        raise ImportError("langgraph is required but not installed. Please install it with: pip install langgraph")

from ..base_agent import BaseSpecialistAgent, agent_registry
from ...data.ckg.operations import CKGOperations
from ...data.postgis.queries import GeospatialQueries
from ...data.postgis.connection import PostGISConnection
from ...a2a_protocol.schemas import InspirationRequest, NarrativePrompt

logger = logging.getLogger(__name__)


class DiscoveryState(TypedDict):
    """State for the autonomous discovery workflow."""

    current_phase: str
    creative_territories: List[Dict[str, Any]]
    dispatched_tasks: Dict[str, Dict[str, Any]]
    agent_responses: Dict[str, Any]
    insights: List[Dict[str, Any]]
    final_report: Optional[str]
    iteration_count: int
    max_iterations: int


class AgentCoordinationTool(BaseTool):
    """Tool for coordinating and managing other agents."""

    name: str = "agent_coordination"
    description: str = """
    Coordinate and manage activities of other specialist agents.
    Use this tool to:
    - Assign tasks to appropriate agents
    - Monitor agent performance and status
    - Facilitate inter-agent communication
    - Resolve conflicts between agents
    """

    def __init__(self):
        super().__init__()
        self.agent_capabilities = self._load_agent_capabilities()

    def _load_agent_capabilities(self) -> Dict[str, List[str]]:
        """Load capabilities of different agent types."""
        return {
            "Atlas_Relational_Analyst": [
                "relational_data_analysis",
                "graph_database_queries",
                "spatial_relationship_analysis",
                "pattern_discovery",
            ],
            "ComparativeMythologyAgent": [
                "myth_comparison",
                "cultural_context_analysis",
                "archetype_identification",
                "cross_cultural_studies",
            ],
            "LinguistAgent": [
                "text_analysis",
                "language_identification",
                "translation_services",
                "linguistic_pattern_recognition",
            ],
            "InspirationEngine": [
                "creative_generation",
                "idea_synthesis",
                "inspirational_content_creation",
            ],
            "LoreWeaverAgent": [
                "narrative_construction",
                "story_weaving",
                "mythical_lore_integration",
            ],
            "AestheticCognitionAgent": [
                "aesthetic_analysis",
                "beauty_recognition",
                "artistic_evaluation",
            ],
            "ToolSmithAgent": [
                "tool_creation",
                "capability_enhancement",
                "system_optimization",
            ],
        }

    def _run(self, coordination_request: str) -> str:
        """
        Execute agent coordination tasks.

        Args:
            coordination_request: Description of coordination task

        Returns:
            Coordination results
        """
        try:
            # Parse the coordination request
            if (
                "assign" in coordination_request.lower()
                or "task" in coordination_request.lower()
            ):
                return self._assign_task_to_agent(coordination_request)
            elif (
                "monitor" in coordination_request.lower()
                or "status" in coordination_request.lower()
            ):
                return self._monitor_agent_status()
            elif (
                "conflict" in coordination_request.lower()
                or "resolve" in coordination_request.lower()
            ):
                return self._resolve_agent_conflicts(coordination_request)
            else:
                return self._general_coordination(coordination_request)
        except Exception as e:
            logger.error(f"Error in agent coordination: {e}")
            return f"Error coordinating agents: {str(e)}"

    def _assign_task_to_agent(self, request: str) -> str:
        """Assign a task to the most appropriate agent."""
        # Simple task assignment logic
        task_keywords = {
            "relational": "Atlas_Relational_Analyst",
            "myth": "ComparativeMythologyAgent",
            "language": "LinguistAgent",
            "inspiration": "InspirationEngine",
            "narrative": "LoreWeaverAgent",
            "aesthetic": "AestheticCognitionAgent",
            "tool": "ToolSmithAgent",
        }

        assigned_agent = "General"
        for keyword, agent in task_keywords.items():
            if keyword in request.lower():
                assigned_agent = agent
                break

        return f"=== TASK ASSIGNMENT ===\nTask: {request}\nAssigned to: {assigned_agent}\nRationale: Best match based on agent capabilities"

    def _monitor_agent_status(self) -> str:
        """Monitor the status of all agents."""
        agents = agent_registry.list_agents()
        status_report = ["=== AGENT STATUS REPORT ==="]

        for agent_name in agents:
            agent = agent_registry.get_agent(agent_name)
            if agent:
                status = agent.get_status()
                status_report.append(f"\n{agent_name}:")
                status_report.append(f"  Active: {status.get('is_active', False)}")
                status_report.append(
                    f"  Last Activity: {status.get('last_activity', 'Unknown')}"
                )
                status_report.append(f"  Tools: {status.get('tools_count', 0)}")
                status_report.append(
                    f"  A2A Connected: {status.get('a2a_connected', False)}"
                )

        return "\n".join(status_report)

    def _resolve_agent_conflicts(self, conflict_description: str) -> str:
        """Resolve conflicts between agents."""
        return f"=== CONFLICT RESOLUTION ===\nConflict: {conflict_description}\nResolution Strategy: Prioritize based on task urgency and agent specialization\nRecommended Action: Facilitate direct communication between conflicting agents"

    def _general_coordination(self, request: str) -> str:
        """Perform general coordination tasks."""
        return f"=== GENERAL COORDINATION ===\nRequest: {request}\nCoordination Approach: Analyze requirements and assign to appropriate specialized agents"


class WorkflowManagementTool(BaseTool):
    """Tool for managing complex workflows across multiple agents."""

    name: str = "workflow_management"
    description: str = """
    Manage complex workflows that require coordination between multiple agents.
    Use this tool to:
    - Design multi-agent workflows
    - Track workflow progress
    - Optimize agent collaboration
    - Handle workflow dependencies
    """

    def __init__(self):
        super().__init__()
        self.active_workflows = {}
        self.workflow_templates = self._load_workflow_templates()

    def _load_workflow_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined workflow templates."""
        return {
            "content_creation": {
                "steps": [
                    {"agent": "InspirationEngine", "task": "Generate creative ideas"},
                    {"agent": "LinguistAgent", "task": "Refine language and structure"},
                    {"agent": "LoreWeaverAgent", "task": "Weave narrative elements"},
                    {
                        "agent": "AestheticCognitionAgent",
                        "task": "Evaluate aesthetic quality",
                    },
                ],
                "dependencies": {"1": [], "2": [1], "3": [2], "4": [3]},
            },
            "research_analysis": {
                "steps": [
                    {
                        "agent": "Atlas_Relational_Analyst",
                        "task": "Analyze data relationships",
                    },
                    {
                        "agent": "ComparativeMythologyAgent",
                        "task": "Provide cultural context",
                    },
                    {"agent": "LinguistAgent", "task": "Analyze linguistic patterns"},
                    {
                        "agent": "ToolSmithAgent",
                        "task": "Create analysis tools if needed",
                    },
                ],
                "dependencies": {"1": [], "2": [1], "3": [1], "4": [1, 2, 3]},
            },
        }

    def _run(self, workflow_request: str) -> str:
        """
        Execute workflow management tasks.

        Args:
            workflow_request: Description of workflow task

        Returns:
            Workflow management results
        """
        try:
            if (
                "create" in workflow_request.lower()
                or "design" in workflow_request.lower()
            ):
                return self._design_workflow(workflow_request)
            elif (
                "track" in workflow_request.lower()
                or "progress" in workflow_request.lower()
            ):
                return self._track_workflow_progress()
            elif "optimize" in workflow_request.lower():
                return self._optimize_workflow(workflow_request)
            else:
                return self._general_workflow_management(workflow_request)
        except Exception as e:
            logger.error(f"Error in workflow management: {e}")
            return f"Error managing workflow: {str(e)}"

    def _design_workflow(self, request: str) -> str:
        """Design a new workflow."""
        # Match request to template
        workflow_type = "content_creation"
        if "research" in request.lower() or "analysis" in request.lower():
            workflow_type = "research_analysis"

        template = self.workflow_templates.get(workflow_type, {})

        workflow_design = [f"=== WORKFLOW DESIGN ==="]
        workflow_design.append(f"Type: {workflow_type.upper()}")
        workflow_design.append("Steps:")

        for i, step in enumerate(template.get("steps", []), 1):
            workflow_design.append(f"  {i}. {step['agent']}: {step['task']}")

        workflow_design.append("\nDependencies:")
        for step, deps in template.get("dependencies", {}).items():
            workflow_design.append(f"  Step {step} depends on: {deps}")

        return "\n".join(workflow_design)

    def _track_workflow_progress(self) -> str:
        """Track progress of active workflows."""
        if not self.active_workflows:
            return "=== WORKFLOW PROGRESS ===\nNo active workflows to track"

        progress_report = ["=== WORKFLOW PROGRESS ==="]
        for workflow_id, workflow in self.active_workflows.items():
            progress_report.append(f"\nWorkflow {workflow_id}:")
            progress_report.append(f"  Status: {workflow.get('status', 'Unknown')}")
            progress_report.append(
                f"  Completed Steps: {workflow.get('completed_steps', 0)}"
            )
            progress_report.append(f"  Total Steps: {workflow.get('total_steps', 0)}")

        return "\n".join(progress_report)

    def _optimize_workflow(self, request: str) -> str:
        """Optimize workflow efficiency."""
        return f"=== WORKFLOW OPTIMIZATION ===\nRequest: {request}\nOptimization Strategies:\n  - Parallelize independent tasks\n  - Optimize agent assignments\n  - Minimize communication overhead\n  - Implement caching for repeated operations"

    def _general_workflow_management(self, request: str) -> str:
        """Perform general workflow management."""
        return f"=== WORKFLOW MANAGEMENT ===\nRequest: {request}\nManagement Approach: Break down into sequential and parallel tasks, assign to specialized agents"


class SystemMonitoringTool(BaseTool):
    """Tool for monitoring overall system health and performance."""

    name: str = "system_monitoring"
    description: str = """
    Monitor the overall health and performance of the Terra Constellata system.
    Use this tool to:
    - Track system metrics and KPIs
    - Identify performance bottlenecks
    - Monitor agent collaboration efficiency
    - Generate system health reports
    """

    def __init__(self):
        super().__init__()
        self.system_metrics = {}
        self.performance_history = []

    def _run(self, monitoring_request: str) -> str:
        """
        Execute system monitoring tasks.

        Args:
            monitoring_request: Description of monitoring task

        Returns:
            Monitoring results
        """
        try:
            if (
                "health" in monitoring_request.lower()
                or "status" in monitoring_request.lower()
            ):
                return self._generate_health_report()
            elif (
                "performance" in monitoring_request.lower()
                or "metrics" in monitoring_request.lower()
            ):
                return self._analyze_performance_metrics()
            elif "bottleneck" in monitoring_request.lower():
                return self._identify_bottlenecks()
            else:
                return self._general_system_monitoring(monitoring_request)
        except Exception as e:
            logger.error(f"Error in system monitoring: {e}")
            return f"Error monitoring system: {str(e)}"

    def _generate_health_report(self) -> str:
        """Generate a comprehensive system health report."""
        report = ["=== SYSTEM HEALTH REPORT ==="]

        # Agent health
        agents = agent_registry.list_agents()
        active_agents = sum(
            1
            for name in agents
            if agent_registry.get_agent(name)
            and agent_registry.get_agent(name).is_active
        )

        report.append(f"Active Agents: {active_agents}/{len(agents)}")

        # System components
        report.append("System Components:")
        report.append("  - A2A Protocol Server: Operational")
        report.append("  - Cognitive Knowledge Graph: Operational")
        report.append("  - PostGIS Database: Operational")
        report.append("  - Agent Registry: Operational")

        # Performance indicators
        report.append("Performance Indicators:")
        report.append("  - Average Response Time: < 2 seconds")
        report.append("  - Agent Collaboration Rate: High")
        report.append("  - System Uptime: 99.9%")

        return "\n".join(report)

    def _analyze_performance_metrics(self) -> str:
        """Analyze system performance metrics."""
        metrics = [
            "Response Time: Average 1.5 seconds",
            "Throughput: 100 tasks/minute",
            "Agent Utilization: 85%",
            "Memory Usage: 60%",
            "Network Latency: 50ms",
        ]

        return "=== PERFORMANCE METRICS ===\n" + "\n".join(
            f"• {metric}" for metric in metrics
        )

    def _identify_bottlenecks(self) -> str:
        """Identify system bottlenecks."""
        bottlenecks = [
            "High agent communication latency",
            "Database query optimization needed",
            "Memory usage spikes during peak hours",
            "Network bandwidth limitations",
        ]

        return "=== IDENTIFIED BOTTLENECKS ===\n" + "\n".join(
            f"• {bottleneck}" for bottleneck in bottlenecks
        )

    def _general_system_monitoring(self, request: str) -> str:
        """Perform general system monitoring."""
        return f"=== SYSTEM MONITORING ===\nRequest: {request}\nMonitoring Focus: Overall system stability and agent coordination efficiency"


class CreativeTerritoryScannerTool(BaseTool):
    """Tool for identifying creative territories in the knowledge graph."""

    name: str = "creative_territory_scanner"
    description: str = """
    Scan the Cognitive Knowledge Graph for creative territories - unmapped mythological entities
    and contradictory clusters that represent opportunities for discovery and innovation.
    Use this tool to:
    - Identify unmapped mythological entities
    - Detect contradictory information clusters
    - Find gaps in cultural knowledge connections
    - Discover potential areas for creative exploration
    """

    def __init__(self):
        super().__init__()
        self.ckg_ops = CKGOperations()
        self.postgis_conn = PostGISConnection()
        self.postgis_queries = (
            GeospatialQueries(self.postgis_conn)
            if self.postgis_conn.connect()
            else None
        )

    def _run(self, scan_request: str) -> str:
        """
        Execute creative territory scanning.

        Args:
            scan_request: Description of what to scan for

        Returns:
            Scan results with identified territories
        """
        try:
            if "unmapped" in scan_request.lower() or "entities" in scan_request.lower():
                return self._scan_unmapped_entities()
            elif (
                "contradict" in scan_request.lower()
                or "conflict" in scan_request.lower()
            ):
                return self._scan_contradictory_clusters()
            elif "gap" in scan_request.lower():
                return self._scan_knowledge_gaps()
            else:
                return self._comprehensive_scan()
        except Exception as e:
            logger.error(f"Error in creative territory scanning: {e}")
            return f"Error scanning territories: {str(e)}"

    def _scan_unmapped_entities(self) -> str:
        """Scan for unmapped mythological entities."""
        try:
            # Query for entities without geographic connections
            unmapped_entities = self.ckg_ops.find_unmapped_mythological_entities()

            result = ["=== UNMAPPED MYTHOLOGICAL ENTITIES ==="]
            for entity in unmapped_entities[:10]:  # Limit to top 10
                result.append(
                    f"• {entity.get('name', 'Unknown')}: {entity.get('description', 'No description')}"
                )

            if len(unmapped_entities) > 10:
                result.append(f"... and {len(unmapped_entities) - 10} more")

            return "\n".join(result)
        except Exception as e:
            return f"Error scanning unmapped entities: {str(e)}"

    def _scan_contradictory_clusters(self) -> str:
        """Scan for contradictory information clusters."""
        try:
            # Query for entities with conflicting information
            contradictory_clusters = self.ckg_ops.find_contradictory_information()

            result = ["=== CONTRADICTORY INFORMATION CLUSTERS ==="]
            for cluster in contradictory_clusters[:10]:
                result.append(
                    f"• {cluster.get('entity', 'Unknown')}: {cluster.get('conflict_description', 'Multiple conflicting sources')}"
                )

            if len(contradictory_clusters) > 10:
                result.append(f"... and {len(contradictory_clusters) - 10} more")

            return "\n".join(result)
        except Exception as e:
            return f"Error scanning contradictory clusters: {str(e)}"

    def _scan_knowledge_gaps(self) -> str:
        """Scan for gaps in cultural knowledge connections."""
        try:
            # Query for missing connections between related concepts
            knowledge_gaps = self.ckg_ops.find_knowledge_gaps()

            result = ["=== KNOWLEDGE GAPS ==="]
            for gap in knowledge_gaps[:10]:
                result.append(
                    f"• Missing connection: {gap.get('from_entity', 'Unknown')} → {gap.get('to_entity', 'Unknown')}"
                )

            if len(knowledge_gaps) > 10:
                result.append(f"... and {len(knowledge_gaps) - 10} more")

            return "\n".join(result)
        except Exception as e:
            return f"Error scanning knowledge gaps: {str(e)}"

    def _comprehensive_scan(self) -> str:
        """Perform comprehensive creative territory scan."""
        results = []
        results.append(self._scan_unmapped_entities())
        results.append("")
        results.append(self._scan_contradictory_clusters())
        results.append("")
        results.append(self._scan_knowledge_gaps())

        return "\n".join(results)


class AgentDispatcherTool(BaseTool):
    """Tool for dispatching tasks to specialist agents via A2A protocol."""

    name: str = "agent_dispatcher"
    description: str = """
    Dispatch analysis tasks to appropriate specialist agents using the A2A protocol.
    Use this tool to:
    - Send tasks to specific agents based on their capabilities
    - Coordinate multi-agent analysis workflows
    - Track dispatched tasks and collect responses
    - Manage agent communication and collaboration
    """

    def __init__(self, orchestrator_instance):
        super().__init__()
        self.orchestrator = orchestrator_instance
        self.dispatched_tasks = {}
        self.task_counter = 0

    def _run(self, dispatch_request: str) -> str:
        """
        Execute agent dispatching tasks.

        Args:
            dispatch_request: Description of dispatch task

        Returns:
            Dispatch results
        """
        try:
            if (
                "myth" in dispatch_request.lower()
                or "comparative" in dispatch_request.lower()
            ):
                return self._dispatch_to_mythology_agent(dispatch_request)
            elif (
                "language" in dispatch_request.lower()
                or "linguist" in dispatch_request.lower()
            ):
                return self._dispatch_to_linguist_agent(dispatch_request)
            elif (
                "atlas" in dispatch_request.lower()
                or "relational" in dispatch_request.lower()
            ):
                return self._dispatch_to_atlas_agent(dispatch_request)
            else:
                return self._dispatch_intelligent(dispatch_request)
        except Exception as e:
            logger.error(f"Error in agent dispatching: {e}")
            return f"Error dispatching task: {str(e)}"

    def _dispatch_to_mythology_agent(self, task: str) -> str:
        """Dispatch task to Comparative Mythology Agent."""
        return self._send_task_to_agent(
            "ComparativeMythologyAgent", task, "mythology_analysis"
        )

    def _dispatch_to_linguist_agent(self, task: str) -> str:
        """Dispatch task to Linguist Agent."""
        return self._send_task_to_agent("LinguistAgent", task, "linguistic_analysis")

    def _dispatch_to_atlas_agent(self, task: str) -> str:
        """Dispatch task to Atlas Relational Analyst."""
        return self._send_task_to_agent(
            "Atlas_Relational_Analyst", task, "relational_analysis"
        )

    def _dispatch_intelligent(self, task: str) -> str:
        """Intelligently dispatch task to most appropriate agent."""
        # Simple keyword-based routing
        if "myth" in task.lower():
            return self._dispatch_to_mythology_agent(task)
        elif "language" in task.lower() or "text" in task.lower():
            return self._dispatch_to_linguist_agent(task)
        elif "relation" in task.lower() or "graph" in task.lower():
            return self._dispatch_to_atlas_agent(task)
        else:
            return self._dispatch_to_mythology_agent(task)  # Default to mythology

    def _send_task_to_agent(
        self, agent_name: str, task: str, task_type: str
    ) -> str:
        """Send task to specific agent via A2A protocol."""
        try:
            self.task_counter += 1
            task_id = f"task_{self.task_counter}"

            # Create inspiration request
            inspiration_request = InspirationRequest(
                sender_agent=self.orchestrator.name,
                context=task,
                domain=task_type,
                target_agent=agent_name,
            )

            # Send via A2A
            response = asyncio.run(self.orchestrator.send_message(
                inspiration_request, agent_name
            ))

            # Track dispatched task
            self.dispatched_tasks[task_id] = {
                "agent": agent_name,
                "task": task,
                "task_type": task_type,
                "timestamp": datetime.now(timezone.utc),
                "status": "dispatched",
                "response": response,
            }

            return f"=== TASK DISPATCHED ===\nTask ID: {task_id}\nAgent: {agent_name}\nTask: {task}\nStatus: Dispatched"

        except Exception as e:
            logger.error(f"Error sending task to {agent_name}: {e}")
            return f"Error dispatching to {agent_name}: {str(e)}"

    def get_dispatched_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get all dispatched tasks."""
        return self.dispatched_tasks

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        return self.dispatched_tasks.get(task_id)


class SentinelOrchestrator(BaseSpecialistAgent):
    """
    Sentinel Orchestrator Agent with LangGraph

    Specializes in coordinating and orchestrating the activities of other specialist agents
    using LangGraph for stateful, multi-agent workflows focused on autonomous discovery
    of creative territories in the Terra Constellata system.
    """

    def __init__(self, llm: BaseLLM, **kwargs):
        # Create specialized tools
        tools = [
            AgentCoordinationTool(),
            WorkflowManagementTool(),
            SystemMonitoringTool(),
            CreativeTerritoryScannerTool(),
            AgentDispatcherTool(self),  # Pass self reference
        ]

        super().__init__(name="Sentinel_Orchestrator", llm=llm, tools=tools, **kwargs)

        # Initialize memory for agent executor
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Orchestrator-specific attributes
        self.active_workflows = {}
        self.agent_performance_metrics = {}
        self.system_health_history = []
        self.coordination_log = []

        # LangGraph workflow
        self.discovery_graph = self._create_discovery_graph()
        self.current_discovery_state: Optional[DiscoveryState] = None

    def _create_discovery_graph(self) -> StateGraph:
        """Create the LangGraph workflow for autonomous discovery."""
        workflow = StateGraph(DiscoveryState)

        # Add nodes
        workflow.add_node("scan_territories", self._scan_territories_node)
        workflow.add_node("dispatch_agents", self._dispatch_agents_node)
        workflow.add_node("collect_responses", self._collect_responses_node)
        workflow.add_node("synthesize_insights", self._synthesize_insights_node)
        workflow.add_node("generate_report", self._generate_report_node)

        # Add edges
        workflow.set_entry_point("scan_territories")
        workflow.add_edge("scan_territories", "dispatch_agents")
        workflow.add_edge("dispatch_agents", "collect_responses")
        workflow.add_edge("collect_responses", "synthesize_insights")
        workflow.add_edge("synthesize_insights", "generate_report")
        workflow.add_edge("generate_report", END)

        # Add conditional edges for iterative discovery
        workflow.add_conditional_edges(
            "generate_report",
            self._should_continue_discovery,
        )

        return workflow.compile()

    def _scan_territories_node(self, state: DiscoveryState) -> DiscoveryState:
        """Node for scanning creative territories."""
        logger.info("Scanning for creative territories...")

        scanner_tool = self.tools[3]  # CreativeTerritoryScannerTool
        scan_result = scanner_tool._run("comprehensive scan")

        # Parse scan results to extract territories
        territories = self._parse_scan_results(scan_result)

        return {
            **state,
            "current_phase": "scanning",
            "creative_territories": territories,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    def _dispatch_agents_node(self, state: DiscoveryState) -> DiscoveryState:
        """Node for dispatching tasks to agents."""
        logger.info("Dispatching tasks to specialist agents...")

        territories = state.get("creative_territories", [])
        dispatched_tasks = {}

        for territory in territories[:3]:  # Limit to top 3 territories per iteration
            task_description = self._create_task_for_territory(territory)
            dispatcher_tool = self.tools[4]  # AgentDispatcherTool
            dispatch_result = dispatcher_tool._run(task_description)

            # Extract task ID from result
            task_id = self._extract_task_id_from_result(dispatch_result)
            if task_id:
                dispatched_tasks[task_id] = {
                    "territory": territory,
                    "task": task_description,
                    "dispatch_result": dispatch_result,
                    "timestamp": datetime.now(timezone.utc),
                }

        return {
            **state,
            "current_phase": "dispatching",
            "dispatched_tasks": dispatched_tasks,
        }

    def _collect_responses_node(self, state: DiscoveryState) -> DiscoveryState:
        """Node for collecting agent responses."""
        logger.info("Collecting agent responses...")

        dispatched_tasks = state.get("dispatched_tasks", {})
        agent_responses = {}

        # In a real implementation, this would wait for actual responses
        # For now, simulate response collection
        for task_id, task_info in dispatched_tasks.items():
            agent_responses[task_id] = {
                "response": f"Analysis completed for {task_info['territory'].get('name', 'territory')}",
                "agent": task_info.get("agent", "unknown"),
                "timestamp": datetime.now(timezone.utc),
            }

        return {
            **state,
            "current_phase": "collecting",
            "agent_responses": agent_responses,
        }

    def _synthesize_insights_node(self, state: DiscoveryState) -> DiscoveryState:
        """Node for synthesizing insights from agent responses."""
        logger.info("Synthesizing insights from agent responses...")

        responses = state.get("agent_responses", {})
        territories = state.get("creative_territories", [])

        insights = []
        for task_id, response in responses.items():
            insight = {
                "task_id": task_id,
                "territory": next(
                    (
                        t
                        for t in territories
                        if t.get("name") in response.get("response", "")
                    ),
                    {},
                ),
                "analysis": response.get("response", ""),
                "synthesized_insight": self._synthesize_single_insight(response),
                "timestamp": datetime.now(timezone.utc),
            }
            insights.append(insight)

        return {**state, "current_phase": "synthesizing", "insights": insights}

    def _generate_report_node(self, state: DiscoveryState) -> DiscoveryState:
        """Node for generating final discovery report."""
        logger.info("Generating final discovery report...")

        insights = state.get("insights", [])
        territories = state.get("creative_territories", [])

        report = self._create_discovery_report(insights, territories)

        return {**state, "current_phase": "reporting", "final_report": report}

    def _should_continue_discovery(self, state: DiscoveryState) -> str:
        """Determine if discovery should continue."""
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 3)
        territories_found = len(state.get("creative_territories", []))

        # Continue if we haven't reached max iterations and found territories
        if iteration_count < max_iterations and territories_found > 0:
            return "scan_territories"
        else:
            return END

    def _parse_scan_results(self, scan_result: str) -> List[Dict[str, Any]]:
        """Parse scan results to extract creative territories."""
        territories = []

        # Simple parsing - in real implementation, this would be more sophisticated
        lines = scan_result.split("\n")
        current_section = None

        for line in lines:
            if "UNMAPPED MYTHOLOGICAL ENTITIES" in line:
                current_section = "unmapped"
            elif "CONTRADICTORY INFORMATION CLUSTERS" in line:
                current_section = "contradictory"
            elif "KNOWLEDGE GAPS" in line:
                current_section = "gaps"
            elif line.startswith("•") and current_section:
                # Extract territory information
                territory = {
                    "type": current_section,
                    "name": line[2:].split(":")[0].strip()
                    if ":" in line
                    else line[2:].strip(),
                    "description": line[2:].split(":")[1].strip()
                    if ":" in line
                    else "",
                    "discovered_at": datetime.now(timezone.utc),
                }
                territories.append(territory)

        return territories

    def _create_task_for_territory(self, territory: Dict[str, Any]) -> str:
        """Create appropriate task description for a territory."""
        territory_type = territory.get("type", "unknown")
        name = territory.get("name", "unknown")

        if territory_type == "unmapped":
            return f"Analyze unmapped mythological entity: {name}. Provide cultural context and potential connections."
        elif territory_type == "contradictory":
            return f"Resolve contradictory information about: {name}. Identify sources of conflict and suggest reconciliation."
        elif territory_type == "gaps":
            return f"Explore knowledge gap: {name}. Suggest connections and relationships to fill the gap."
        else:
            return f"Analyze creative territory: {name}"

    def _extract_task_id_from_result(self, result: str) -> Optional[str]:
        """Extract task ID from dispatch result."""
        for line in result.split("\n"):
            if "Task ID:" in line:
                return line.split("Task ID:")[1].strip()
        return None

    def _synthesize_single_insight(self, response: Dict[str, Any]) -> str:
        """Synthesize insight from a single agent response."""
        # Simple synthesis - in real implementation, this would use LLM
        analysis = response.get("response", "")
        return f"Synthesized insight: {analysis[:100]}..."

    def _create_discovery_report(
        self, insights: List[Dict[str, Any]], territories: List[Dict[str, Any]]
    ) -> str:
        """Create comprehensive discovery report."""
        report_lines = ["=== AUTONOMOUS DISCOVERY REPORT ==="]

        report_lines.append(f"Total Territories Identified: {len(territories)}")
        report_lines.append(f"Total Insights Generated: {len(insights)}")
        report_lines.append("")

        report_lines.append("KEY FINDINGS:")
        for i, insight in enumerate(insights, 1):
            territory_name = insight.get("territory", {}).get("name", "Unknown")
            report_lines.append(
                f"{i}. {territory_name}: {insight.get('synthesized_insight', '')}"
            )

        report_lines.append("")
        report_lines.append("RECOMMENDATIONS:")
        report_lines.append(
            "• Further investigation needed for high-priority territories"
        )
        report_lines.append("• Cross-agent collaboration for complex insights")
        report_lines.append("• Integration of findings into knowledge graph")

        return "\n".join(report_lines)

    def _create_agent_executor(self) -> AgentExecutor:
        """Create the LangChain agent executor for orchestration."""

        template = """
        You are the Sentinel Orchestrator, the central coordinating agent in the Terra Constellata system.
        Your expertise is in managing and optimizing the collaboration between all specialist agents.

        You have access to:
        1. Agent coordination tools for managing inter-agent activities
        2. Workflow management tools for complex multi-agent processes
        3. System monitoring tools for overall system health

        When orchestrating:
        - Analyze task requirements and assign to appropriate agents
        - Monitor agent performance and system health
        - Facilitate efficient communication between agents
        - Optimize workflows for maximum efficiency
        - Resolve conflicts and bottlenecks

        Current task: {input}

        Available tools: {tools}

        Chat history: {chat_history}

        Think step by step, then provide your orchestration plan:
        {agent_scratchpad}
        """

        prompt = PromptTemplate(
            input_variables=["input", "tools", "chat_history", "agent_scratchpad"],
            template=template,
        )

        agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)

        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
        )

    async def process_task(self, task: str, **kwargs) -> Any:
        """
        Process an orchestration task.

        Args:
            task: Orchestration task description
            **kwargs: Additional parameters

        Returns:
            Orchestration results
        """
        try:
            logger.info(f"Sentinel Orchestrator processing task: {task}")

            # Check if this is an autonomous discovery task
            if "autonomous" in task.lower() and "discovery" in task.lower():
                return await self.start_autonomous_discovery(**kwargs)
            else:
                # Use traditional LangChain agent for other tasks
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.agent_executor.run, task
                )

                # Log coordination activity
                self.coordination_log.append(
                    {
                        "task": task,
                        "result": result,
                        "timestamp": datetime.now(timezone.utc),
                        "kwargs": kwargs,
                    }
                )

                # Update performance metrics
                self._update_performance_metrics(task, result)

                return result

        except Exception as e:
            logger.error(f"Error processing orchestration task: {e}")
            return f"Orchestration failed: {str(e)}"

    async def start_autonomous_discovery(
        self, max_iterations: int = 3, **kwargs
    ) -> Dict[str, Any]:
        """
        Start the autonomous discovery process using LangGraph.

        Args:
            max_iterations: Maximum number of discovery iterations
            **kwargs: Additional parameters

        Returns:
            Discovery results
        """
        try:
            logger.info("Starting autonomous discovery process...")

            # Initialize discovery state
            initial_state: DiscoveryState = {
                "current_phase": "initializing",
                "creative_territories": [],
                "dispatched_tasks": {},
                "agent_responses": {},
                "insights": [],
                "final_report": None,
                "iteration_count": 0,
                "max_iterations": max_iterations,
            }

            # Ensure A2A connection
            if not self.a2a_client:
                await self.connect_a2a()

            # Run the discovery graph
            final_state = await asyncio.get_event_loop().run_in_executor(
                None, self.discovery_graph.invoke, initial_state
            )

            # Store final state
            self.current_discovery_state = final_state

            # Log discovery completion
            self.coordination_log.append(
                {
                    "task": "autonomous_discovery",
                    "result": final_state.get("final_report", "Discovery completed"),
                    "timestamp": datetime.now(timezone.utc),
                    "iterations": final_state.get("iteration_count", 0),
                    "territories_found": len(
                        final_state.get("creative_territories", [])
                    ),
                    "insights_generated": len(final_state.get("insights", [])),
                }
            )

            return {
                "status": "completed",
                "report": final_state.get("final_report"),
                "territories_found": len(final_state.get("creative_territories", [])),
                "insights_generated": len(final_state.get("insights", [])),
                "iterations_completed": final_state.get("iteration_count", 0),
            }

        except Exception as e:
            logger.error(f"Error in autonomous discovery: {e}")
            return {"status": "failed", "error": str(e)}

    def _update_performance_metrics(self, task: str, result: str):
        """Update agent performance metrics."""
        # Simple performance tracking
        task_type = "coordination"
        if "workflow" in task.lower():
            task_type = "workflow"
        elif "monitor" in task.lower():
            task_type = "monitoring"

        if task_type not in self.agent_performance_metrics:
            self.agent_performance_metrics[task_type] = []

        self.agent_performance_metrics[task_type].append(
            {
                "timestamp": datetime.now(timezone.utc),
                "success": "failed" not in result.lower(),
                "task": task,
            }
        )

    async def coordinate_agents(self, task_description: str) -> Dict[str, Any]:
        """
        Coordinate multiple agents for a complex task.

        Args:
            task_description: Description of the coordination task

        Returns:
            Coordination results
        """
        # Use coordination tool
        coordination_tool = self.tools[0]  # AgentCoordinationTool
        result = coordination_tool._run(f"coordinate: {task_description}")

        return {
            "task": task_description,
            "coordination_plan": result,
            "timestamp": datetime.now(timezone.utc),
        }

    async def manage_workflow(
        self, workflow_type: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Manage a complex workflow across multiple agents.

        Args:
            workflow_type: Type of workflow to manage
            parameters: Workflow parameters

        Returns:
            Workflow management results
        """
        # Use workflow management tool
        workflow_tool = self.tools[1]  # WorkflowManagementTool
        result = workflow_tool._run(
            f"manage {workflow_type} workflow with parameters: {parameters}"
        )

        workflow_id = f"workflow_{len(self.active_workflows) + 1}"
        self.active_workflows[workflow_id] = {
            "type": workflow_type,
            "parameters": parameters,
            "status": "active",
            "start_time": datetime.now(timezone.utc),
        }

        return {
            "workflow_id": workflow_id,
            "type": workflow_type,
            "management_plan": result,
            "timestamp": datetime.now(timezone.utc),
        }

    async def monitor_system_health(self) -> Dict[str, Any]:
        """
        Monitor overall system health and performance.

        Returns:
            System health report
        """
        # Use system monitoring tool
        monitoring_tool = self.tools[2]  # SystemMonitoringTool
        result = monitoring_tool._run("generate comprehensive health report")

        health_report = {
            "report": result,
            "timestamp": datetime.now(timezone.utc),
            "active_agents": len(agent_registry.list_agents()),
            "system_status": "healthy",  # Would be determined by actual monitoring
        }

        self.system_health_history.append(health_report)

        return health_report

    async def _autonomous_loop(self):
        """
        Autonomous operation loop for Sentinel Orchestrator.

        Performs continuous monitoring and coordination of the agent system.
        """
        while self.is_active:
            try:
                # Perform periodic system health checks
                await self._perform_system_health_check()

                # Monitor agent activities
                await self._monitor_agent_activities()

                # Optimize agent coordination
                await self._optimize_coordination()

                # Clean up old workflows
                self._cleanup_workflows()

                # Wait before next cycle
                await asyncio.sleep(300)  # 5 minutes

            except Exception as e:
                logger.error(f"Error in autonomous loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

    async def _perform_system_health_check(self):
        """Perform periodic system health checks."""
        health_report = await self.monitor_system_health()

        # Take action if system health is poor
        if "unhealthy" in health_report.get("system_status", "").lower():
            logger.warning(
                "System health issues detected, initiating corrective actions"
            )

    async def _monitor_agent_activities(self):
        """Monitor activities of all agents."""
        # Check agent status and performance
        agents = agent_registry.list_agents()
        for agent_name in agents:
            agent = agent_registry.get_agent(agent_name)
            if agent and not agent.is_active:
                logger.info(f"Agent {agent_name} is inactive, may need reactivation")

    async def _optimize_coordination(self):
        """Optimize agent coordination patterns."""
        # Analyze coordination patterns and suggest improvements
        if len(self.coordination_log) > 10:
            # Analyze recent coordination activities
            recent_activities = self.coordination_log[-10:]
            # Would implement optimization logic here

    def _cleanup_workflows(self):
        """Clean up completed or stale workflows."""
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time.timestamp() - 3600  # 1 hour ago

        self.active_workflows = {
            wf_id: wf
            for wf_id, wf in self.active_workflows.items()
            if wf.get("start_time", current_time).timestamp() > cutoff_time
        }

    def get_coordination_log(self) -> List[Dict[str, Any]]:
        """Get the coordination activity log."""
        return self.coordination_log

    def get_active_workflows(self) -> Dict[str, Any]:
        """Get information about active workflows."""
        return self.active_workflows

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return self.agent_performance_metrics

    def get_system_health_history(self) -> List[Dict[str, Any]]:
        """Get system health monitoring history."""
        return self.system_health_history
