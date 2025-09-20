"""
Base Agent Class for Terra Constellata Specialist Agents

This module provides the base class for all specialist agents in the Terra Constellata
system, integrating LangChain agents with the A2A protocol for inter-agent communication.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from langchain.agents import AgentExecutor, BaseSingleActionAgent
from langchain.llms.base import BaseLLM
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from ..a2a_protocol.client import A2AClient
from ..a2a_protocol.schemas import (
    A2AMessage,
    InspirationRequest,
    CreationFeedback,
    ToolProposal,
    NarrativePrompt,
    CertificationRequest,
)

logger = logging.getLogger(__name__)


class BaseSpecialistAgent(ABC):
    """
    Base class for all specialist agents in Terra Constellata.

    This class provides:
    - LangChain agent integration
    - A2A protocol communication
    - Tool management
    - Memory and state management
    - Autonomous operation capabilities
    """

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        tools: List[BaseTool],
        a2a_server_url: str = "http://localhost:8080",
        memory_size: int = 1000,
        **kwargs,
    ):
        """
        Initialize the specialist agent.

        Args:
            name: Unique name for this agent
            llm: Language model for the agent
            tools: List of tools available to the agent
            a2a_server_url: URL of the A2A protocol server
            memory_size: Maximum memory buffer size
            **kwargs: Additional agent-specific parameters
        """
        self.name = name
        self.llm = llm
        self.tools = tools
        self.a2a_server_url = a2a_server_url
        self.memory_size = memory_size

        # Initialize A2A client
        self.a2a_client: Optional[A2AClient] = None

        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, max_token_limit=memory_size
        )

        # Agent state
        self.is_active = False
        self.last_activity = datetime.utcnow()

        # Codex integration
        self.codex_manager = None

        # Store additional parameters
        self.kwargs = kwargs

        # Initialize the LangChain agent
        self.agent_executor = self._create_agent_executor()

        logger.info(f"Initialized {self.__class__.__name__}: {name}")

    async def connect_a2a(self):
        """Connect to the A2A protocol server."""
        if self.a2a_client is None:
            self.a2a_client = A2AClient(self.a2a_server_url, self.name)
            await self.a2a_client.connect()
            logger.info(f"{self.name} connected to A2A server")

    async def disconnect_a2a(self):
        """Disconnect from the A2A protocol server."""
        if self.a2a_client:
            await self.a2a_client.disconnect()
            self.a2a_client = None
            logger.info(f"{self.name} disconnected from A2A server")

    @abstractmethod
    def _create_agent_executor(self) -> AgentExecutor:
        """
        Create the LangChain agent executor.

        This method should be implemented by subclasses to define
        the specific agent type and behavior.
        """
        pass

    @abstractmethod
    async def process_task(self, task: str, **kwargs) -> Any:
        """
        Process a task autonomously.

        Args:
            task: Task description
            **kwargs: Additional task parameters

        Returns:
            Task result
        """
        pass

    async def process_task_with_archival(self, task: str, **kwargs) -> Any:
        """
        Process a task with automatic Codex archival.

        Args:
            task: Task description
            **kwargs: Additional task parameters

        Returns:
            Task result
        """
        start_time = datetime.utcnow()

        try:
            # Process the task
            result = await self.process_task(task, **kwargs)
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            # Archive to Codex if available
            if self.codex_manager:
                try:
                    success_metrics = kwargs.get("success_metrics", {"completed": True})
                    collaboration_partners = kwargs.get("collaboration_partners", [])

                    self.codex_manager.archive_agent_task(
                        agent_name=self.name,
                        agent_type=self.__class__.__name__,
                        task_description=task,
                        contribution_type="task_execution",
                        input_data=kwargs.get("input_data", {}),
                        output_data={
                            "result": str(result)[:1000]
                        },  # Truncate for storage
                        success_metrics=success_metrics,
                        duration=duration,
                        workflow_context=kwargs.get("workflow_context"),
                        collaboration_partners=collaboration_partners,
                        ai_model=getattr(self.llm, "model_name", None)
                        if hasattr(self.llm, "model_name")
                        else None,
                        ai_provider=getattr(self.llm, "provider", None)
                        if hasattr(self.llm, "provider")
                        else None,
                    )
                except Exception as e:
                    logger.warning(f"Failed to archive task to Codex: {e}")

            return result

        except Exception as e:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            # Archive failed task if Codex is available
            if self.codex_manager:
                try:
                    self.codex_manager.archive_agent_task(
                        agent_name=self.name,
                        agent_type=self.__class__.__name__,
                        task_description=task,
                        contribution_type="task_execution",
                        input_data=kwargs.get("input_data", {}),
                        output_data={"error": str(e)},
                        success_metrics={"completed": False, "error": str(e)},
                        duration=duration,
                        workflow_context=kwargs.get("workflow_context"),
                    )
                except Exception as archive_error:
                    logger.warning(
                        f"Failed to archive failed task to Codex: {archive_error}"
                    )

            raise

    async def send_message(
        self, message: A2AMessage, target_agent: Optional[str] = None
    ) -> Any:
        """
        Send a message via A2A protocol.

        Args:
            message: A2A message to send
            target_agent: Target agent name (optional)

        Returns:
            Response from target agent
        """
        if not self.a2a_client:
            await self.connect_a2a()

        message.target_agent = target_agent
        return await self.a2a_client.send_request(
            method=self._get_message_method(message), message=message
        )

    async def send_notification(
        self, message: A2AMessage, target_agent: Optional[str] = None
    ):
        """
        Send a notification via A2A protocol.

        Args:
            message: A2A message to send
            target_agent: Target agent name (optional)
        """
        if not self.a2a_client:
            await self.connect_a2a()

        message.target_agent = target_agent
        await self.a2a_client.send_notification(
            method=self._get_message_method(message), message=message
        )

    def _get_message_method(self, message: A2AMessage) -> str:
        """Get the JSON-RPC method name for a message type."""
        message_type_map = {
            InspirationRequest: "INSPIRATION_REQUEST",
            CreationFeedback: "CREATION_FEEDBACK",
            ToolProposal: "TOOL_PROPOSAL",
            NarrativePrompt: "NARRATIVE_PROMPT",
            CertificationRequest: "CERTIFICATION_REQUEST",
        }
        return message_type_map.get(type(message), "UNKNOWN_MESSAGE")

    async def request_inspiration(
        self, context: str, domain: str, target_agent: Optional[str] = None
    ) -> Any:
        """
        Request inspiration from another agent.

        Args:
            context: Context for inspiration
            domain: Domain of inspiration
            target_agent: Target agent name

        Returns:
            Inspiration response
        """
        message = InspirationRequest(
            context=context, domain=domain, target_agent=target_agent
        )
        return await self.send_message(message, target_agent)

    async def provide_feedback(
        self,
        original_request_id: str,
        feedback_type: str,
        content: str,
        target_agent: str,
        rating: Optional[int] = None,
    ):
        """
        Provide feedback on creative output.

        Args:
            original_request_id: ID of original request
            feedback_type: Type of feedback
            content: Feedback content
            target_agent: Target agent name
            rating: Optional rating (1-5)
        """
        message = CreationFeedback(
            original_request_id=original_request_id,
            feedback_type=feedback_type,
            content=content,
            rating=rating,
            target_agent=target_agent,
        )
        await self.send_notification(message, target_agent)

    async def propose_tool(
        self,
        tool_name: str,
        description: str,
        capabilities: List[str],
        target_agent: str,
        use_case: str,
    ):
        """
        Propose a new tool to another agent.

        Args:
            tool_name: Name of proposed tool
            description: Tool description
            capabilities: Tool capabilities
            target_agent: Target agent name
            use_case: Use case for the tool
        """
        message = ToolProposal(
            tool_name=tool_name,
            description=description,
            capabilities=capabilities,
            use_case=use_case,
            target_agent=target_agent,
        )
        await self.send_notification(message, target_agent)

    async def request_narrative(
        self,
        theme: str,
        elements: List[str],
        target_agent: str,
        style: str = "narrative",
    ) -> Any:
        """
        Request narrative generation.

        Args:
            theme: Narrative theme
            elements: Key elements to include
            target_agent: Target agent name
            style: Narrative style

        Returns:
            Narrative response
        """
        message = NarrativePrompt(
            theme=theme, elements=elements, style=style, target_agent=target_agent
        )
        return await self.send_message(message, target_agent)

    async def start_autonomous_operation(self):
        """Start autonomous operation mode."""
        self.is_active = True
        logger.info(f"{self.name} started autonomous operation")

        # Connect to A2A if not connected
        if not self.a2a_client:
            await self.connect_a2a()

        # Start the main operation loop
        await self._autonomous_loop()

    async def stop_autonomous_operation(self):
        """Stop autonomous operation mode."""
        self.is_active = False
        logger.info(f"{self.name} stopped autonomous operation")

    @abstractmethod
    async def _autonomous_loop(self):
        """
        Main autonomous operation loop.

        This method should be implemented by subclasses to define
        the agent's autonomous behavior.
        """
        pass

    def set_codex_manager(self, codex_manager):
        """
        Set the Codex manager for archival and learning integration.

        Args:
            codex_manager: CodexManager instance
        """
        self.codex_manager = codex_manager
        logger.info(f"{self.name} integrated with Codex")

    def update_memory(self, input_text: str, output_text: str):
        """Update agent memory with conversation."""
        self.memory.save_context({"input": input_text}, {"output": output_text})
        self.last_activity = datetime.utcnow()

        # Archive to Codex if available
        if self.codex_manager:
            try:
                self.codex_manager.archive_agent_task(
                    agent_name=self.name,
                    agent_type=self.__class__.__name__,
                    task_description=f"Memory update: {input_text[:100]}...",
                    contribution_type="learning_update",
                    input_data={"input": input_text},
                    output_data={"output": output_text},
                    success_metrics={"memory_updated": True},
                    workflow_context="memory_management",
                )
            except Exception as e:
                logger.warning(f"Failed to archive memory update to Codex: {e}")

    def get_memory_context(self) -> str:
        """Get current memory context."""
        return self.memory.buffer_as_str

    def get_status(self) -> Dict[str, Any]:
        """Get agent status information."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "is_active": self.is_active,
            "last_activity": self.last_activity.isoformat(),
            "memory_size": len(self.memory.chat_memory.messages),
            "tools_count": len(self.tools),
            "a2a_connected": self.a2a_client is not None,
        }


class SpecialistAgentRegistry:
    """Registry for managing specialist agents."""

    def __init__(self):
        self.agents: Dict[str, BaseSpecialistAgent] = {}

    def register_agent(self, agent: BaseSpecialistAgent):
        """Register an agent."""
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")

    def get_agent(self, name: str) -> Optional[BaseSpecialistAgent]:
        """Get an agent by name."""
        return self.agents.get(name)

    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self.agents.keys())

    def get_agents_by_type(self, agent_type: str) -> List[BaseSpecialistAgent]:
        """Get agents by type."""
        return [
            agent
            for agent in self.agents.values()
            if agent.__class__.__name__ == agent_type
        ]

    async def broadcast_message(
        self, message: A2AMessage, exclude_agent: Optional[str] = None
    ):
        """Broadcast a message to all agents except the sender."""
        for agent_name, agent in self.agents.items():
            if agent_name != exclude_agent:
                try:
                    await agent.send_notification(message, agent_name)
                except Exception as e:
                    logger.error(f"Failed to send message to {agent_name}: {e}")


# Global agent registry
agent_registry = SpecialistAgentRegistry()
