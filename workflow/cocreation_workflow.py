"""
Co-Creation Workflow for Human-AI Partnership

This module implements the unified workflow that connects all components
in the doubt -> discovery -> art -> wisdom -> knowledge cycle.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

try:
    from langgraph import StateGraph, END
except ImportError:
    # Fallback for different langgraph versions
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        # Create a simple mock for testing
        class StateGraph:
            def __init__(self, state_class):
                self.state_class = state_class
                self.nodes = {}
                self.edges = []

            def add_node(self, name, func):
                self.nodes[name] = func

            def add_conditional_edges(self, source, condition, targets):
                self.edges.append((source, condition, targets))

            def set_entry_point(self, node):
                self.entry_point = node

            def compile(self):
                return self

            def invoke(self, state):
                # Simple synchronous execution for testing
                return state

        END = "END"
from langchain.llms.base import BaseLLM

from ..agents.sentinel.sentinel_orchestrator import SentinelOrchestrator
from ..agents.apprentice.apprentice_agent import ApprenticeAgent
from ..codex.codex_manager import CodexManager
from ..chatbot.backend import LoreWeaverRAG
from ..a2a_protocol.schemas import (
    A2AMessage,
    InspirationRequest,
    CreationFeedback,
    WorkflowTrigger,
)

logger = logging.getLogger(__name__)


class WorkflowStage(Enum):
    """Stages in the co-creation workflow."""

    DOUBT = "doubt"
    DISCOVERY = "discovery"
    ART = "art"
    WISDOM = "wisdom"
    KNOWLEDGE = "knowledge"
    PUBLICATION = "publication"


@dataclass
class CoCreationState:
    """State for the co-creation workflow."""

    current_stage: WorkflowStage
    trigger_source: str  # "human" or "autonomous"
    human_input: Optional[str] = None
    discovery_results: Optional[Dict[str, Any]] = None
    artistic_outputs: Optional[List[Dict[str, Any]]] = None
    wisdom_insights: Optional[Dict[str, Any]] = None
    knowledge_entries: Optional[List[Dict[str, Any]]] = None
    publication_ready: bool = False
    human_feedback: Optional[Dict[str, Any]] = None
    workflow_metadata: Dict[str, Any] = None


class CoCreationWorkflow:
    """
    Unified workflow orchestrator for human-AI co-creation.

    Implements the complete cycle: doubt -> discovery -> art -> wisdom -> knowledge
    with seamless integration between all Terra Constellata components.
    """

    def __init__(
        self,
        sentinel: SentinelOrchestrator,
        apprentice: ApprenticeAgent,
        codex: CodexManager,
        chatbot: Optional[LoreWeaverRAG] = None,
        llm: Optional[BaseLLM] = None,
    ):
        self.sentinel = sentinel
        self.apprentice = apprentice
        self.codex = codex
        self.chatbot = chatbot
        self.llm = llm

        # Workflow state
        self.current_workflow_id = None
        self.workflow_history = []

        # Create the workflow graph
        self.workflow_graph = self._create_workflow_graph()

        logger.info("Co-Creation Workflow initialized")

    def _create_workflow_graph(self) -> StateGraph:
        """Create the state graph for the co-creation workflow."""

        def doubt_stage(state: CoCreationState) -> CoCreationState:
            """Handle doubt/trigger stage."""
            return self._process_doubt_stage(state)

        def discovery_stage(state: CoCreationState) -> CoCreationState:
            """Handle discovery stage."""
            return self._process_discovery_stage(state)

        def art_stage(state: CoCreationState) -> CoCreationState:
            """Handle art generation stage."""
            return self._process_art_stage(state)

        def wisdom_stage(state: CoCreationState) -> CoCreationState:
            """Handle wisdom synthesis stage."""
            return self._process_wisdom_stage(state)

        def knowledge_stage(state: CoCreationState) -> CoCreationState:
            """Handle knowledge preservation stage."""
            return self._process_knowledge_stage(state)

        def publication_stage(state: CoCreationState) -> CoCreationState:
            """Handle publication stage."""
            return self._process_publication_stage(state)

        def should_continue_to_discovery(state: CoCreationState) -> bool:
            """Determine if we should proceed to discovery."""
            return (
                state.trigger_source in ["human", "autonomous"]
                and state.human_input is not None
            )

        def should_continue_to_art(state: CoCreationState) -> bool:
            """Determine if we should proceed to art generation."""
            return (
                state.discovery_results is not None
                and len(state.discovery_results.get("insights", [])) > 0
            )

        def should_continue_to_wisdom(state: CoCreationState) -> bool:
            """Determine if we should proceed to wisdom synthesis."""
            return (
                state.artistic_outputs is not None and len(state.artistic_outputs) > 0
            )

        def should_continue_to_knowledge(state: CoCreationState) -> bool:
            """Determine if we should proceed to knowledge preservation."""
            return state.wisdom_insights is not None

        def should_publish(state: CoCreationState) -> bool:
            """Determine if we should proceed to publication."""
            return state.publication_ready and state.human_feedback is not None

        # Create the graph
        workflow = StateGraph(CoCreationState)

        # Add nodes
        workflow.add_node("doubt", doubt_stage)
        workflow.add_node("discovery", discovery_stage)
        workflow.add_node("art", art_stage)
        workflow.add_node("wisdom", wisdom_stage)
        workflow.add_node("knowledge", knowledge_stage)
        workflow.add_node("publication", publication_stage)

        # Add edges with conditions
        workflow.add_conditional_edges(
            "doubt", should_continue_to_discovery, {"discovery": "discovery", END: END}
        )

        workflow.add_conditional_edges(
            "discovery", should_continue_to_art, {"art": "art", END: END}
        )

        workflow.add_conditional_edges(
            "art", should_continue_to_wisdom, {"wisdom": "wisdom", END: END}
        )

        workflow.add_conditional_edges(
            "wisdom", should_continue_to_knowledge, {"knowledge": "knowledge", END: END}
        )

        workflow.add_conditional_edges(
            "knowledge", should_publish, {"publication": "publication", END: END}
        )

        # Set entry point
        workflow.set_entry_point("doubt")

        return workflow.compile()

    def _process_doubt_stage(self, state: CoCreationState) -> CoCreationState:
        """Process the doubt/trigger stage."""
        logger.info(f"Processing doubt stage for workflow {self.current_workflow_id}")

        # Initialize workflow metadata
        if state.workflow_metadata is None:
            state.workflow_metadata = {
                "workflow_id": self.current_workflow_id,
                "start_time": datetime.utcnow(),
                "stages_completed": [],
                "human_interactions": [],
            }

        # If human input, analyze and prepare for discovery
        if state.trigger_source == "human" and state.human_input:
            # Use LLM to analyze human doubt/question
            if self.llm:
                analysis = self.llm(
                    f"""
                Analyze this human input for creative potential and research directions:
                {state.human_input}

                Provide:
                1. Key themes and concepts
                2. Potential research directions
                3. Artistic interpretation opportunities
                4. Cultural/historical connections
                """
                )
                state.workflow_metadata["human_analysis"] = analysis

        state.current_stage = WorkflowStage.DOUBT
        state.workflow_metadata["stages_completed"].append("doubt")

        return state

    def _process_discovery_stage(self, state: CoCreationState) -> CoCreationState:
        """Process the discovery stage using Sentinel Orchestrator."""
        logger.info(
            f"Processing discovery stage for workflow {self.current_workflow_id}"
        )

        try:
            # Use Sentinel for autonomous discovery
            discovery_task = (
                state.human_input or "Explore creative territories in geomythology"
            )

            # Run discovery process
            discovery_results = asyncio.run(
                self.sentinel.start_autonomous_discovery(max_iterations=2)
            )

            state.discovery_results = discovery_results
            state.current_stage = WorkflowStage.DISCOVERY
            state.workflow_metadata["stages_completed"].append("discovery")

            logger.info(
                f"Discovery completed with {len(discovery_results.get('insights', []))} insights"
            )

        except Exception as e:
            logger.error(f"Error in discovery stage: {e}")
            state.discovery_results = {"error": str(e), "insights": []}

        return state

    def _process_art_stage(self, state: CoCreationState) -> CoCreationState:
        """Process the art generation stage using Apprentice Agent."""
        logger.info(f"Processing art stage for workflow {self.current_workflow_id}")

        try:
            artistic_outputs = []

            # Extract key insights from discovery
            insights = state.discovery_results.get("insights", [])

            for insight in insights[:3]:  # Limit to 3 artworks for now
                # Generate artistic representation
                art_description = f"Create artistic representation of: {insight.get('description', '')}"

                # Use Apprentice Agent for style transfer/art generation
                art_result = asyncio.run(self.apprentice.process_task(art_description))

                artistic_outputs.append(
                    {
                        "insight_id": insight.get("id"),
                        "description": art_description,
                        "artwork": art_result,
                        "timestamp": datetime.utcnow(),
                    }
                )

            state.artistic_outputs = artistic_outputs
            state.current_stage = WorkflowStage.ART
            state.workflow_metadata["stages_completed"].append("art")

            logger.info(
                f"Art generation completed with {len(artistic_outputs)} artworks"
            )

        except Exception as e:
            logger.error(f"Error in art stage: {e}")
            state.artistic_outputs = []

        return state

    def _process_wisdom_stage(self, state: CoCreationState) -> CoCreationState:
        """Process the wisdom synthesis stage using Codex."""
        logger.info(f"Processing wisdom stage for workflow {self.current_workflow_id}")

        try:
            # Archive the entire workflow in Codex
            workflow_contribution = self.codex.archive_agent_task(
                agent_name="CoCreationWorkflow",
                agent_type="orchestrator",
                task_description=f"Complete co-creation cycle: {state.human_input}",
                contribution_type="workflow_orchestration",
                input_data={
                    "trigger": state.human_input,
                    "discovery_results": state.discovery_results,
                    "artistic_outputs": state.artistic_outputs,
                },
                output_data={
                    "workflow_metadata": state.workflow_metadata,
                    "stage": "wisdom_synthesis",
                },
                success_metrics={
                    "stages_completed": len(
                        state.workflow_metadata.get("stages_completed", [])
                    ),
                    "artworks_generated": len(state.artistic_outputs or []),
                    "insights_discovered": len(
                        state.discovery_results.get("insights", [])
                    )
                    if state.discovery_results
                    else 0,
                },
                workflow_context=self.current_workflow_id,
            )

            # Extract wisdom insights
            wisdom_insights = {
                "contribution_id": workflow_contribution,
                "patterns_identified": self._extract_patterns_from_workflow(state),
                "learning_opportunities": self._identify_learning_opportunities(state),
                "collaboration_insights": self._analyze_collaboration_patterns(state),
            }

            state.wisdom_insights = wisdom_insights
            state.current_stage = WorkflowStage.WISDOM
            state.workflow_metadata["stages_completed"].append("wisdom")

            logger.info(
                f"Wisdom synthesis completed, archived as {workflow_contribution}"
            )

        except Exception as e:
            logger.error(f"Error in wisdom stage: {e}")
            state.wisdom_insights = {"error": str(e)}

        return state

    def _process_knowledge_stage(self, state: CoCreationState) -> CoCreationState:
        """Process the knowledge preservation stage using Codex."""
        logger.info(
            f"Processing knowledge stage for workflow {self.current_workflow_id}"
        )

        try:
            # Generate legacy chapter for the Galactic Storybook
            chapter_id = self.codex.generate_legacy_chapter(
                chapter_type="collaboration",
                collaboration_name=f"CoCreation_{self.current_workflow_id}",
                agents=[
                    "SentinelOrchestrator",
                    "ApprenticeAgent",
                    "CoCreationWorkflow",
                ],
                contributions=[state.wisdom_insights.get("contribution_id")],
                theme="human_ai_partnership",
            )

            # Document strategy from this workflow
            strategy_id = self.codex.document_strategy(
                title=f"Co-Creation Pattern: {state.human_input[:50]}...",
                strategy_type="collaboration",
                description=f"Successful human-AI co-creation workflow pattern",
                context="Posthuman creativity and geomythological exploration",
                steps=[
                    {
                        "step": "doubt",
                        "description": "Human input or autonomous trigger",
                    },
                    {
                        "step": "discovery",
                        "description": "Autonomous exploration and insight generation",
                    },
                    {
                        "step": "art",
                        "description": "Artistic representation of discoveries",
                    },
                    {
                        "step": "wisdom",
                        "description": "Knowledge synthesis and archiving",
                    },
                    {
                        "step": "knowledge",
                        "description": "Legacy preservation and learning",
                    },
                ],
                success_criteria=[
                    "Human satisfaction with creative output",
                    "Successful knowledge preservation",
                    "Positive collaboration metrics",
                ],
                lessons_learned=[
                    "Importance of human-AI feedback loops",
                    "Value of multi-stage creative processes",
                    "Benefits of autonomous discovery integration",
                ],
                created_by="CoCreationWorkflow",
                related_contributions=[state.wisdom_insights.get("contribution_id")],
                tags=["co-creation", "human-ai", "posthuman-creativity"],
            )

            state.knowledge_entries = [
                {
                    "type": "chapter",
                    "id": chapter_id,
                    "title": f"CoCreation_{self.current_workflow_id}",
                },
                {
                    "type": "strategy",
                    "id": strategy_id,
                    "title": f"Co-Creation Pattern",
                },
            ]

            state.current_stage = WorkflowStage.KNOWLEDGE
            state.workflow_metadata["stages_completed"].append("knowledge")

            logger.info(
                f"Knowledge preservation completed: chapter {chapter_id}, strategy {strategy_id}"
            )

        except Exception as e:
            logger.error(f"Error in knowledge stage: {e}")
            state.knowledge_entries = []

        return state

    def _process_publication_stage(self, state: CoCreationState) -> CoCreationState:
        """Process the publication stage."""
        logger.info(
            f"Processing publication stage for workflow {self.current_workflow_id}"
        )

        try:
            # Publish the chapter to Galactic Storybook
            for entry in state.knowledge_entries or []:
                if entry["type"] == "chapter":
                    success = self.codex.chapter_generator.publish_chapter(entry["id"])
                    if success:
                        logger.info(
                            f"Chapter {entry['id']} published to Galactic Storybook"
                        )

            # Update workflow metadata
            state.workflow_metadata["publication_time"] = datetime.utcnow()
            state.workflow_metadata["stages_completed"].append("publication")

            state.current_stage = WorkflowStage.PUBLICATION
            state.publication_ready = True

            logger.info(
                f"Publication stage completed for workflow {self.current_workflow_id}"
            )

        except Exception as e:
            logger.error(f"Error in publication stage: {e}")

        return state

    def _extract_patterns_from_workflow(self, state: CoCreationState) -> List[str]:
        """Extract patterns from the completed workflow."""
        patterns = []

        if state.discovery_results and state.discovery_results.get("insights"):
            patterns.append("Autonomous discovery successfully generated insights")

        if state.artistic_outputs:
            patterns.append("Artistic representation enhanced discovery insights")

        if state.human_feedback:
            patterns.append("Human feedback integrated into creative process")

        return patterns

    def _identify_learning_opportunities(self, state: CoCreationState) -> List[str]:
        """Identify learning opportunities from the workflow."""
        opportunities = []

        # Analyze success metrics
        success_metrics = state.workflow_metadata.get("stages_completed", [])
        if len(success_metrics) >= 5:
            opportunities.append(
                "Full workflow completion indicates robust integration"
            )

        # Analyze human feedback
        if state.human_feedback and state.human_feedback.get("rating", 0) >= 4:
            opportunities.append(
                "High human satisfaction suggests effective co-creation"
            )

        return opportunities

    def _analyze_collaboration_patterns(self, state: CoCreationState) -> Dict[str, Any]:
        """Analyze collaboration patterns in the workflow."""
        return {
            "agents_involved": [
                "SentinelOrchestrator",
                "ApprenticeAgent",
                "CodexManager",
            ],
            "communication_channels": ["A2A_Protocol", "Direct_API"],
            "feedback_loops": bool(state.human_feedback),
            "autonomous_elements": ["discovery", "art_generation"],
            "human_elements": ["input", "feedback", "curation"],
        }

    async def start_cocreation_workflow(
        self,
        trigger_source: str,
        human_input: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Start a new co-creation workflow.

        Args:
            trigger_source: "human" or "autonomous"
            human_input: Human input text (if applicable)
            workflow_id: Optional workflow ID

        Returns:
            Workflow results
        """
        self.current_workflow_id = (
            workflow_id or f"cocreation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

        # Initialize state
        initial_state = CoCreationState(
            current_stage=WorkflowStage.DOUBT,
            trigger_source=trigger_source,
            human_input=human_input,
            workflow_metadata={"workflow_id": self.current_workflow_id},
        )

        try:
            logger.info(f"Starting co-creation workflow {self.current_workflow_id}")

            # Execute the workflow
            final_state = await asyncio.get_event_loop().run_in_executor(
                None, self.workflow_graph.invoke, initial_state
            )

            # Store in history
            workflow_result = {
                "workflow_id": self.current_workflow_id,
                "final_state": final_state,
                "completion_time": datetime.utcnow(),
                "success": final_state.current_stage == WorkflowStage.PUBLICATION,
            }

            self.workflow_history.append(workflow_result)

            logger.info(
                f"Co-creation workflow {self.current_workflow_id} completed at stage {final_state.current_stage}"
            )

            return workflow_result

        except Exception as e:
            logger.error(f"Error in co-creation workflow: {e}")
            return {
                "workflow_id": self.current_workflow_id,
                "error": str(e),
                "success": False,
            }

    async def submit_human_feedback(
        self, workflow_id: str, feedback: Dict[str, Any]
    ) -> bool:
        """
        Submit human feedback for a workflow.

        Args:
            workflow_id: Workflow ID
            feedback: Feedback data

        Returns:
            Success status
        """
        try:
            # Find the workflow in history
            for workflow in self.workflow_history:
                if workflow["workflow_id"] == workflow_id:
                    workflow["final_state"].human_feedback = feedback

                    # If workflow is in knowledge stage, trigger publication
                    if workflow["final_state"].current_stage == WorkflowStage.KNOWLEDGE:
                        await self._continue_to_publication(workflow["final_state"])

                    logger.info(f"Human feedback submitted for workflow {workflow_id}")
                    return True

            logger.warning(f"Workflow {workflow_id} not found for feedback submission")
            return False

        except Exception as e:
            logger.error(f"Error submitting human feedback: {e}")
            return False

    async def _continue_to_publication(self, state: CoCreationState):
        """Continue workflow to publication stage after human feedback."""
        try:
            # Update state with feedback
            state.publication_ready = True

            # Process publication
            final_state = self._process_publication_stage(state)

            logger.info(
                f"Workflow {state.workflow_metadata['workflow_id']} advanced to publication"
            )

        except Exception as e:
            logger.error(f"Error continuing to publication: {e}")

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow."""
        for workflow in self.workflow_history:
            if workflow["workflow_id"] == workflow_id:
                return {
                    "workflow_id": workflow_id,
                    "current_stage": workflow["final_state"].current_stage.value,
                    "stages_completed": workflow["final_state"].workflow_metadata.get(
                        "stages_completed", []
                    ),
                    "success": workflow.get("success", False),
                    "completion_time": workflow.get("completion_time"),
                }
        return None

    def get_workflow_history(self) -> List[Dict[str, Any]]:
        """Get the history of all workflows."""
        return [
            {
                "workflow_id": w["workflow_id"],
                "current_stage": w["final_state"].current_stage.value,
                "success": w.get("success", False),
                "completion_time": w.get("completion_time"),
                "stages_completed": len(
                    w["final_state"].workflow_metadata.get("stages_completed", [])
                ),
            }
            for w in self.workflow_history
        ]
