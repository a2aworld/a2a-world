"""
A2A Protocol Integration for Inspiration Engine

This module provides integration with the A2A protocol for agent-to-agent
communication about inspiration, novelty detection, and creative outputs.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json

from ..a2a_protocol.client import A2AClient
from ..a2a_protocol.schemas import (
    InspirationRequest,
    CreationFeedback,
    NarrativePrompt,
    ToolProposal,
    CertificationRequest,
    A2AMessage,
)
from .algorithms import NoveltyScore
from .prompt_ranking import CreativePrompt, PromptRanking

logger = logging.getLogger(__name__)


class A2AInspirationClient:
    """
    A2A client specialized for inspiration engine communication
    """

    def __init__(
        self,
        server_url: str,
        agent_name: str = "inspiration_engine",
        timeout: float = 30.0,
    ):
        """
        Initialize A2A inspiration client

        Args:
            server_url: A2A server URL
            agent_name: Name of this agent
            timeout: Request timeout in seconds
        """
        self.server_url = server_url
        self.agent_name = agent_name
        self.timeout = timeout
        self.client = A2AClient(server_url, agent_name, timeout)

    async def __aenter__(self):
        """Async context manager entry"""
        await self.client.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.client.disconnect()

    async def request_inspiration(
        self,
        context: str,
        domain: str,
        constraints: Optional[List[str]] = None,
        target_agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Request creative inspiration from other agents

        Args:
            context: Context for inspiration request
            domain: Creative domain (mythology, geography, etc.)
            constraints: Optional constraints for the inspiration
            target_agent: Specific agent to request from

        Returns:
            Inspiration response from target agent
        """
        message = InspirationRequest(
            sender_agent=self.agent_name,
            target_agent=target_agent,
            context=context,
            domain=domain,
            constraints=constraints or [],
            inspiration_type="creative_novelty",
        )

        try:
            response = await self.client.send_request("INSPIRATION_REQUEST", message)
            logger.info(
                f"Received inspiration response from {target_agent or 'any agent'}"
            )
            return response
        except Exception as e:
            logger.error(f"Failed to request inspiration: {e}")
            return {"error": str(e)}

    async def share_novelty_findings(
        self, findings: Dict[str, Any], target_agents: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Share novelty detection findings with other agents

        Args:
            findings: Novelty findings to share
            target_agents: List of agents to share with (None = broadcast)

        Returns:
            List of responses from recipient agents
        """
        # Create narrative prompt based on findings
        prompt_content = self._create_narrative_prompt_from_findings(findings)

        message = NarrativePrompt(
            sender_agent=self.agent_name,
            target_agent=target_agents[0]
            if target_agents and len(target_agents) == 1
            else None,
            theme="novelty_discovery",
            elements=findings.get("key_elements", []),
            style="analytical",
            length="medium",
            constraints=["focus_on_novelty", "include_quantitative_scores"],
        )

        try:
            if target_agents and len(target_agents) > 1:
                # Send to multiple agents
                requests = [("NARRATIVE_PROMPT", message) for _ in target_agents]
                responses = await self.client.batch_request(requests)
            else:
                # Send to single agent or broadcast
                response = await self.client.send_request("NARRATIVE_PROMPT", message)
                responses = [response]

            logger.info(f"Shared novelty findings with {len(responses)} agents")
            return responses
        except Exception as e:
            logger.error(f"Failed to share novelty findings: {e}")
            return [{"error": str(e)}]

    async def propose_inspiration_tools(
        self, tool_ideas: List[Dict[str, Any]], target_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Propose new tools or capabilities for inspiration generation

        Args:
            tool_ideas: List of tool proposals
            target_agent: Agent to propose tools to

        Returns:
            Response from target agent
        """
        if not tool_ideas:
            return {"error": "No tool ideas provided"}

        # Use the first tool idea for the proposal
        tool_idea = tool_ideas[0]

        message = ToolProposal(
            sender_agent=self.agent_name,
            target_agent=target_agent,
            tool_name=tool_idea.get("name", "InspirationEnhancementTool"),
            description=tool_idea.get(
                "description", "Tool for enhancing creative inspiration"
            ),
            capabilities=tool_idea.get("capabilities", []),
            requirements=tool_idea.get("requirements", []),
            use_case=tool_idea.get("use_case", "Creative inspiration enhancement"),
            priority=tool_idea.get("priority", "medium"),
        )

        try:
            response = await self.client.send_request("TOOL_PROPOSAL", message)
            logger.info(f"Proposed inspiration tool to {target_agent or 'coordinator'}")
            return response
        except Exception as e:
            logger.error(f"Failed to propose inspiration tools: {e}")
            return {"error": str(e)}

    async def send_creation_feedback(
        self,
        original_request_id: str,
        feedback_type: str,
        content: str,
        rating: Optional[int] = None,
        target_agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send feedback on creative outputs or inspiration requests

        Args:
            original_request_id: ID of the original request
            feedback_type: Type of feedback (positive, negative, suggestion)
            content: Feedback content
            rating: Optional numerical rating (1-5)
            target_agent: Agent that provided the original inspiration

        Returns:
            Response from feedback recipient
        """
        message = CreationFeedback(
            sender_agent=self.agent_name,
            target_agent=target_agent,
            original_request_id=original_request_id,
            feedback_type=feedback_type,
            content=content,
            rating=rating,
        )

        try:
            response = await self.client.send_request("CREATION_FEEDBACK", message)
            logger.info(f"Sent creation feedback to {target_agent or 'agent'}")
            return response
        except Exception as e:
            logger.error(f"Failed to send creation feedback: {e}")
            return {"error": str(e)}

    async def request_certification(
        self,
        subject: str,
        certification_type: str,
        evidence: Dict[str, Any],
        criteria: List[str],
        target_agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Request certification/validation of inspiration findings

        Args:
            subject: Subject to be certified
            certification_type: Type of certification needed
            evidence: Evidence supporting the certification
            criteria: Certification criteria
            target_agent: Agent to request certification from

        Returns:
            Certification response
        """
        message = CertificationRequest(
            sender_agent=self.agent_name,
            target_agent=target_agent,
            subject=subject,
            certification_type=certification_type,
            evidence=evidence,
            criteria=criteria,
        )

        try:
            response = await self.client.send_request("CERTIFICATION_REQUEST", message)
            logger.info(
                f"Requested certification for {subject} from {target_agent or 'certification_agent'}"
            )
            return response
        except Exception as e:
            logger.error(f"Failed to request certification: {e}")
            return {"error": str(e)}

    async def broadcast_inspiration_update(
        self, update_type: str, update_data: Dict[str, Any]
    ) -> None:
        """
        Broadcast inspiration engine status updates to all agents

        Args:
            update_type: Type of update (e.g., 'novelty_discovered', 'model_updated')
            update_data: Update data to broadcast
        """
        # Create a custom message for broadcasting
        message = A2AMessage(
            sender_agent=self.agent_name,
            message_id=f"broadcast_{update_type}_{datetime.utcnow().isoformat()}",
            timestamp=datetime.utcnow(),
        )

        # Add custom fields
        message.__dict__.update(
            {"update_type": update_type, "update_data": update_data}
        )

        try:
            await self.client.send_notification("INSPIRATION_UPDATE", message)
            logger.info(f"Broadcasted {update_type} update to all agents")
        except Exception as e:
            logger.error(f"Failed to broadcast inspiration update: {e}")

    def _create_narrative_prompt_from_findings(self, findings: Dict[str, Any]) -> str:
        """
        Create a narrative prompt based on novelty findings

        Args:
            findings: Novelty detection findings

        Returns:
            Narrative prompt string
        """
        key_elements = findings.get("key_elements", [])
        novelty_scores = findings.get("novelty_scores", {})
        domain = findings.get("domain", "general")

        prompt_parts = [
            f"Explore the novel discovery in the {domain} domain:",
            f"Key elements: {', '.join(key_elements[:5])}",  # Limit to 5 elements
        ]

        if novelty_scores:
            top_score = max(
                novelty_scores.values(),
                key=lambda x: x.score if hasattr(x, "score") else 0,
            )
            if hasattr(top_score, "algorithm"):
                prompt_parts.append(
                    f"Novelty detected using {top_score.algorithm} algorithm"
                )

        prompt_parts.append(
            "Develop a creative narrative that captures this unique insight."
        )

        return " ".join(prompt_parts)

    async def collaborative_inspiration_session(
        self,
        session_topic: str,
        participating_agents: List[str],
        duration_minutes: int = 30,
    ) -> Dict[str, Any]:
        """
        Initiate a collaborative inspiration session with multiple agents

        Args:
            session_topic: Topic for the inspiration session
            participating_agents: List of agents to participate
            duration_minutes: Session duration in minutes

        Returns:
            Session results and contributions
        """
        session_data = {
            "session_id": f"session_{datetime.utcnow().isoformat()}",
            "topic": session_topic,
            "participants": participating_agents,
            "duration_minutes": duration_minutes,
            "start_time": datetime.utcnow().isoformat(),
        }

        # Send session invitation to all participants
        invitation_message = A2AMessage(
            sender_agent=self.agent_name,
            message_id=f"invitation_{session_data['session_id']}",
            timestamp=datetime.utcnow(),
        )
        invitation_message.__dict__.update(
            {"invitation_type": "inspiration_session", "session_data": session_data}
        )

        try:
            # Send invitations
            requests = [
                ("SESSION_INVITATION", invitation_message) for _ in participating_agents
            ]
            responses = await self.client.batch_request(requests)

            session_results = {
                "session_data": session_data,
                "invitations_sent": len(requests),
                "responses_received": len(
                    [r for r in responses if not isinstance(r, Exception)]
                ),
                "participants_joined": [
                    agent
                    for agent, resp in zip(participating_agents, responses)
                    if not isinstance(resp, Exception) and resp.get("accepted", False)
                ],
                "contributions": [],
            }

            logger.info(
                f"Initiated collaborative session '{session_topic}' with {len(participating_agents)} agents"
            )
            return session_results

        except Exception as e:
            logger.error(f"Failed to initiate collaborative session: {e}")
            return {"error": str(e)}

    async def share_prompt_rankings(
        self, rankings: PromptRanking, target_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Share prompt ranking results with other agents

        Args:
            rankings: PromptRanking object with results
            target_agent: Agent to share rankings with

        Returns:
            Response from recipient agent
        """
        ranking_data = {
            "total_prompts": len(rankings.ranked_prompts),
            "diversity_score": rankings.diversity_score,
            "ranking_criteria": rankings.ranking_criteria,
            "top_prompt": {
                "id": rankings.top_prompt.id if rankings.top_prompt else None,
                "content": rankings.top_prompt.content[:200]
                if rankings.top_prompt
                else None,
                "domain": rankings.top_prompt.domain if rankings.top_prompt else None,
                "creative_potential": rankings.top_prompt.creative_potential
                if rankings.top_prompt
                else None,
            }
            if rankings.top_prompt
            else None,
            "timestamp": rankings.timestamp.isoformat(),
        }

        message = A2AMessage(
            sender_agent=self.agent_name,
            target_agent=target_agent,
            message_id=f"rankings_{datetime.utcnow().isoformat()}",
            timestamp=datetime.utcnow(),
        )
        message.__dict__.update(
            {"ranking_data": ranking_data, "message_type": "prompt_rankings"}
        )

        try:
            response = await self.client.send_request("PROMPT_RANKINGS", message)
            logger.info(f"Shared prompt rankings with {target_agent or 'agent'}")
            return response
        except Exception as e:
            logger.error(f"Failed to share prompt rankings: {e}")
            return {"error": str(e)}

    async def request_peer_review(
        self, content: str, content_type: str, reviewers: List[str], criteria: List[str]
    ) -> Dict[str, Any]:
        """
        Request peer review of creative content from other agents

        Args:
            content: Content to be reviewed
            content_type: Type of content (prompt, narrative, etc.)
            reviewers: List of agents to request review from
            criteria: Review criteria

        Returns:
            Review responses from agents
        """
        review_request = {
            "content": content[:1000],  # Limit content length
            "content_type": content_type,
            "criteria": criteria,
            "request_id": f"review_{datetime.utcnow().isoformat()}",
        }

        message = A2AMessage(
            sender_agent=self.agent_name,
            message_id=review_request["request_id"],
            timestamp=datetime.utcnow(),
        )
        message.__dict__.update(
            {"review_request": review_request, "message_type": "peer_review_request"}
        )

        try:
            # Send review requests to multiple agents
            requests = [("PEER_REVIEW_REQUEST", message) for _ in reviewers]
            responses = await self.client.batch_request(requests)

            review_results = {
                "request_id": review_request["request_id"],
                "reviewers_contacted": len(reviewers),
                "responses_received": len(
                    [r for r in responses if not isinstance(r, Exception)]
                ),
                "reviews": [r for r in responses if not isinstance(r, Exception)],
                "errors": [str(r) for r in responses if isinstance(r, Exception)],
            }

            logger.info(f"Requested peer review from {len(reviewers)} agents")
            return review_results

        except Exception as e:
            logger.error(f"Failed to request peer review: {e}")
            return {"error": str(e)}


# Utility functions for common inspiration engine operations
async def broadcast_novelty_alert(
    client: A2AInspirationClient, novelty_score: float, domain: str, description: str
) -> None:
    """
    Broadcast a novelty alert to all agents

    Args:
        client: A2A inspiration client
        novelty_score: Novelty score (0-1)
        domain: Domain of the novelty
        description: Description of the novel finding
    """
    alert_data = {
        "novelty_score": novelty_score,
        "domain": domain,
        "description": description,
        "alert_level": "high"
        if novelty_score > 0.8
        else "medium"
        if novelty_score > 0.6
        else "low",
    }

    await client.broadcast_inspiration_update("novelty_alert", alert_data)


async def request_inspiration_collaboration(
    client: A2AInspirationClient, topic: str, required_expertise: List[str]
) -> Dict[str, Any]:
    """
    Request collaboration on an inspiration topic

    Args:
        client: A2A inspiration client
        topic: Inspiration topic
        required_expertise: List of required expertise areas

    Returns:
        Collaboration setup response
    """
    collaboration_data = {
        "topic": topic,
        "required_expertise": required_expertise,
        "collaboration_type": "inspiration_development",
    }

    # Find agents with required expertise
    # This would typically involve querying an agent registry
    potential_collaborators = ["mythology_agent", "geography_agent", "narrative_agent"]

    return await client.collaborative_inspiration_session(
        topic, potential_collaborators, duration_minutes=45
    )
