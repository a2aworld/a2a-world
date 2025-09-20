"""
Apprentice Agent

The Apprentice Agent is a generative AI agent specialized in artistic style transfer
using CycleGAN architecture. It learns and replicates user artistic styles to generate
novel artwork from various inputs including satellite imagery.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

import torch
from langchain.agents import AgentExecutor, create_react_agent
from langchain.llms.base import BaseLLM
from langchain.tools import BaseTool, tool
from langchain.prompts import PromptTemplate

from ..base_agent import BaseSpecialistAgent
from ...a2a_protocol.schemas import (
    A2AMessage,
    InspirationRequest,
    CreationFeedback,
    ToolProposal,
    NarrativePrompt,
    CertificationRequest,
    GeospatialAnomalyIdentified,
)
from .cyclegan_model import CycleGAN
from .data_loader import ImageDataset, DataLoader
from .training_pipeline import TrainingPipeline
from .style_transfer import StyleTransfer

logger = logging.getLogger(__name__)


class StyleTransferTool(BaseTool):
    """Tool for performing artistic style transfer using CycleGAN."""

    name: str = "style_transfer"
    description: str = """
    Perform artistic style transfer on images using CycleGAN.
    Use this tool to:
    - Apply learned artistic styles to new images
    - Generate artwork in user's signature style
    - Transform satellite imagery into artistic representations
    - Create novel artistic compositions
    """

    def __init__(self, apprentice_agent):
        super().__init__()
        self.apprentice_agent = apprentice_agent

    def _run(self, input_description: str) -> str:
        """
        Execute style transfer on input images.

        Args:
            input_description: Description of input images and desired style

        Returns:
            Style transfer results
        """
        try:
            return self.apprentice_agent.perform_style_transfer(input_description)
        except Exception as e:
            logger.error(f"Error in style transfer: {e}")
            return f"Style transfer failed: {str(e)}"


class ModelTrainingTool(BaseTool):
    """Tool for training and fine-tuning CycleGAN models."""

    name: str = "train_model"
    description: str = """
    Train or fine-tune CycleGAN models for style transfer.
    Use this tool to:
    - Train new models on user artwork datasets
    - Fine-tune existing models with new styles
    - Update models with additional training data
    - Monitor training progress and performance
    """

    def __init__(self, apprentice_agent):
        super().__init__()
        self.apprentice_agent = apprentice_agent

    def _run(self, training_config: str) -> str:
        """
        Execute model training with specified configuration.

        Args:
            training_config: JSON string with training parameters

        Returns:
            Training results and status
        """
        try:
            return self.apprentice_agent.train_model(training_config)
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return f"Model training failed: {str(e)}"


class ApprenticeAgent(BaseSpecialistAgent):
    """
    Apprentice Agent for artistic style transfer using CycleGAN.

    This agent specializes in learning artistic styles and generating novel artwork
    by applying those styles to various inputs, including satellite imagery.
    """

    def __init__(
        self,
        llm: BaseLLM,
        model_dir: str = "./models/apprentice",
        data_dir: str = "./data/apprentice",
        device: str = "auto",
        **kwargs,
    ):
        # Initialize model components
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.device = torch.device(
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CycleGAN components
        self.cyclegan = CycleGAN(device=self.device)
        self.training_pipeline = TrainingPipeline(self.cyclegan, self.model_dir)
        self.style_transfer = StyleTransfer(self.cyclegan, self.model_dir)

        # Training state
        self.current_training_job = None
        self.training_history = []

        # Create specialized tools
        tools = [
            StyleTransferTool(self),
            ModelTrainingTool(self),
        ]

        super().__init__(name="Apprentice_Agent", llm=llm, tools=tools, **kwargs)

        # Agent-specific attributes
        self.artwork_history = []
        self.style_library = {}

        logger.info(f"Apprentice Agent initialized on device: {self.device}")

    def _create_agent_executor(self) -> AgentExecutor:
        """Create the LangChain agent executor for artistic tasks."""

        template = """
        You are the Apprentice Agent, a master of artistic style transfer in the Terra Constellata system.
        Your expertise is in learning artistic styles and generating novel artwork using CycleGAN technology.

        You have access to:
        1. Style transfer tools for applying learned artistic styles
        2. Model training capabilities for learning new artistic styles
        3. Image processing for various input types (satellite imagery, photos, etc.)
        4. A2A protocol integration for collaborative artistic creation

        When working with artistic tasks:
        - Consider the artistic vision and intent behind each piece
        - Apply appropriate styles that enhance the visual narrative
        - Collaborate with other agents for inspiration and feedback
        - Maintain artistic quality and coherence in generated works
        - Document the creative process for the Codex

        Current task: {input}

        Available tools: {tools}

        Chat history: {chat_history}

        Think step by step, then provide your artistic analysis:
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
        Process an artistic task using CycleGAN and style transfer.

        Args:
            task: Artistic task description
            **kwargs: Additional parameters

        Returns:
            Artistic results
        """
        try:
            logger.info(f"Apprentice processing artistic task: {task}")

            # Execute the artistic task
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.agent_executor.run, task
            )

            # Store in artwork history
            self.artwork_history.append(
                {
                    "task": task,
                    "result": result,
                    "timestamp": datetime.utcnow(),
                    "kwargs": kwargs,
                }
            )

            return result

        except Exception as e:
            logger.error(f"Error processing artistic task: {e}")
            return f"Artistic task failed: {str(e)}"

    def perform_style_transfer(self, input_description: str) -> str:
        """
        Perform style transfer on input images.

        Args:
            input_description: Description of input and desired style

        Returns:
            Style transfer results
        """
        try:
            # Parse input description (simplified for now)
            # In a full implementation, this would parse more complex descriptions
            style_name = "default_style"
            input_path = self._parse_input_path(input_description)

            if not input_path or not os.path.exists(input_path):
                return f"Input image not found: {input_path}"

            # Perform style transfer
            result_path = self.style_transfer.apply_style(input_path, style_name)

            # Store result
            self._store_artwork_result(input_description, result_path)

            return f"Style transfer completed. Result saved to: {result_path}"

        except Exception as e:
            return f"Style transfer failed: {str(e)}"

    def train_model(self, training_config: str) -> str:
        """
        Train or fine-tune a CycleGAN model.

        Args:
            training_config: JSON string with training parameters

        Returns:
            Training results
        """
        try:
            # Parse training configuration
            import json

            config = json.loads(training_config)

            # Start training job
            job_id = f"training_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            # This would start async training in a real implementation
            self.current_training_job = {
                "job_id": job_id,
                "config": config,
                "status": "started",
                "start_time": datetime.utcnow(),
            }

            # Store in training history
            self.training_history.append(self.current_training_job)

            return f"Training job started: {job_id}. Status: {self.current_training_job['status']}"

        except Exception as e:
            return f"Training failed: {str(e)}"

    def _parse_input_path(self, input_description: str) -> Optional[str]:
        """Parse input path from description (simplified)."""
        # This is a placeholder - in a real implementation, this would
        # use NLP to extract file paths from natural language descriptions
        if "satellite" in input_description.lower():
            return str(self.data_dir / "satellite_images" / "sample.jpg")
        elif "photo" in input_description.lower():
            return str(self.data_dir / "photos" / "sample.jpg")
        else:
            return str(self.data_dir / "inputs" / "default.jpg")

    def _store_artwork_result(self, description: str, result_path: str):
        """Store artwork result in history."""
        self.artwork_history.append(
            {
                "description": description,
                "result_path": result_path,
                "timestamp": datetime.utcnow(),
            }
        )


    def get_artwork_history(self) -> List[Dict[str, Any]]:
        """Get the history of generated artwork."""
        return self.artwork_history

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get the history of training jobs."""
        return self.training_history

    def get_model_status(self) -> Dict[str, Any]:
        """Get current model and training status."""
        return {
            "device": str(self.device),
            "current_training_job": self.current_training_job,
            "model_dir": str(self.model_dir),
            "data_dir": str(self.data_dir),
            "training_jobs_count": len(self.training_history),
            "artwork_count": len(self.artwork_history),
        }

    # A2A Protocol Integration Methods

    async def request_artistic_inspiration(
        self, theme: str, medium: str = "visual", target_agent: Optional[str] = None
    ) -> Any:
        """
        Request artistic inspiration from other agents.

        Args:
            theme: Artistic theme or concept
            medium: Artistic medium (visual, conceptual, etc.)
            target_agent: Specific agent to request from

        Returns:
            Inspiration response
        """
        context = f"Artistic inspiration request for theme: {theme} in medium: {medium}"
        domain = f"artistic_{medium}"

        message = InspirationRequest(
            context=context,
            domain=domain,
            constraints=[f"medium:{medium}", f"theme:{theme}"],
            inspiration_type="visual",
        )

        return await self.send_message(message, target_agent)

    async def share_artwork(
        self, artwork_path: str, description: str, style_used: str, target_agent: str
    ):
        """
        Share generated artwork with another agent.

        Args:
            artwork_path: Path to the artwork file
            description: Description of the artwork
            style_used: Style/model used to generate it
            target_agent: Agent to share with
        """
        # Create a custom message for artwork sharing
        message = A2AMessage(sender_agent=self.name, target_agent=target_agent)
        # Add custom fields for artwork sharing
        message.artwork_path = artwork_path
        message.description = description
        message.style_used = style_used
        message.message_type = "ARTWORK_SHARE"

        await self.send_notification(message, target_agent)
        logger.info(f"Shared artwork with {target_agent}: {description}")

    async def request_style_collaboration(
        self,
        input_description: str,
        target_agent: str,
        collaboration_type: str = "style_blend",
    ) -> Any:
        """
        Request collaboration on style transfer tasks.

        Args:
            input_description: Description of input for collaboration
            collaboration_type: Type of collaboration requested
            target_agent: Agent to collaborate with

        Returns:
            Collaboration response
        """
        message = ToolProposal(
            tool_name=f"style_collaboration_{collaboration_type}",
            description=f"Collaborative style transfer: {input_description}",
            capabilities=["style_transfer", "artistic_creation", collaboration_type],
            use_case=f"Create novel artwork through {collaboration_type} collaboration",
            target_agent=target_agent,
        )

        return await self.send_message(message, target_agent)

    async def provide_creative_feedback(
        self,
        original_artwork_id: str,
        feedback_content: str,
        rating: int,
        target_agent: str,
        suggestions: Optional[List[str]] = None,
    ):
        """
        Provide creative feedback on artwork.

        Args:
            original_artwork_id: ID of the original artwork
            feedback_content: Feedback content
            rating: Rating from 1-5
            target_agent: Agent that created the artwork
            suggestions: Optional improvement suggestions
        """
        message = CreationFeedback(
            original_request_id=original_artwork_id,
            feedback_type="creative_feedback",
            content=feedback_content,
            rating=rating,
            suggestions=suggestions or [],
            target_agent=target_agent,
        )

        await self.send_notification(message, target_agent)
        logger.info(f"Provided creative feedback to {target_agent}")

    async def request_narrative_for_artwork(
        self,
        artwork_description: str,
        target_agent: str,
        narrative_style: str = "descriptive",
    ) -> Any:
        """
        Request narrative generation for artwork.

        Args:
            artwork_description: Description of the artwork
            narrative_style: Style of narrative to generate
            target_agent: Agent to request narrative from

        Returns:
            Narrative response
        """
        theme = f"Artwork: {artwork_description}"
        elements = ["visual_description", "emotional_impact", "artistic_technique"]

        message = NarrativePrompt(
            theme=theme,
            elements=elements,
            style=narrative_style,
            target_agent=target_agent,
        )

        return await self.send_message(message, target_agent)

    async def certify_artwork_quality(
        self, artwork_path: str, quality_criteria: List[str], target_agent: str
    ) -> Any:
        """
        Request certification/validation of artwork quality.

        Args:
            artwork_path: Path to the artwork
            quality_criteria: List of quality criteria to evaluate
            target_agent: Agent to request certification from

        Returns:
            Certification response
        """
        evidence = {
            "artwork_path": artwork_path,
            "generated_by": self.name,
            "quality_criteria": quality_criteria,
            "timestamp": datetime.utcnow().isoformat(),
        }

        message = CertificationRequest(
            subject=f"Artwork quality certification: {artwork_path}",
            certification_type="artwork_quality",
            evidence=evidence,
            criteria=quality_criteria,
            target_agent=target_agent,
        )

        return await self.send_message(message, target_agent)

    async def report_artistic_anomaly(
        self,
        anomaly_description: str,
        location_data: Optional[Dict[str, float]] = None,
        confidence: float = 0.8,
        target_agent: Optional[str] = None,
    ):
        """
        Report an artistic or creative anomaly discovered during generation.

        Args:
            anomaly_description: Description of the anomaly
            location_data: Optional geospatial location data
            confidence: Confidence level of the anomaly detection
            target_agent: Agent to notify (optional broadcast)
        """
        message = GeospatialAnomalyIdentified(
            anomaly_type="artistic_anomaly",
            location=location_data or {"lat": 0.0, "lon": 0.0},
            confidence=confidence,
            description=anomaly_description,
            data_source=f"Apprentice_Agent_{self.name}",
            target_agent=target_agent,
        )

        await self.send_notification(message, target_agent)
        logger.info(f"Reported artistic anomaly: {anomaly_description}")

    async def _autonomous_loop(self):
        """
        Autonomous operation loop for Apprentice Agent.

        Performs continuous artistic creation and collaboration.
        """
        while self.is_active:
            try:
                # Generate autonomous artwork
                await self._generate_autonomous_artwork()

                # Seek inspiration from other agents
                await self._seek_inspiration()

                # Collaborate on creative tasks
                await self._seek_collaboration()

                # Clean up old results
                self._cleanup_old_results()

                # Wait before next cycle
                await asyncio.sleep(900)  # 15 minutes

            except Exception as e:
                logger.error(f"Error in autonomous loop: {e}")
                await asyncio.sleep(120)  # Wait 2 minutes before retry

    async def _generate_autonomous_artwork(self):
        """Generate artwork autonomously."""
        try:
            # This would generate new artistic pieces based on learned patterns
            # For now, it's a placeholder
            logger.info("Generating autonomous artwork...")
            # In a full implementation, this would create new pieces
        except Exception as e:
            logger.warning(f"Failed to generate autonomous artwork: {e}")

    async def _seek_inspiration(self):
        """Seek inspiration from other agents."""
        try:
            inspiration = await self.request_artistic_inspiration(
                theme="emergent_creativity", medium="visual"
            )
            logger.info(f"Received artistic inspiration: {inspiration}")
        except Exception as e:
            logger.warning(f"Failed to seek artistic inspiration: {e}")

    async def _seek_collaboration(self):
        """Seek collaboration opportunities."""
        try:
            # This would look for collaboration opportunities
            # For now, it's a placeholder
            logger.info("Seeking collaboration opportunities...")
        except Exception as e:
            logger.warning(f"Failed to seek collaboration: {e}")
