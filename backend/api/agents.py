"""
Agent Registry API for Terra Constellata

This module provides API endpoints for managing and interacting with
the data gateway agents that form the "Library of Alexandria for AI Wisdom."
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import PlainTextResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import asyncio
import logging

from ...agents.data_gateway.base_data_gateway_agent import data_gateway_registry
from ...agents.data_gateway.secrets_manager import resolve_secret_placeholder

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models for API requests/responses
class AgentRegistrationRequest(BaseModel):
    """Request model for agent registration."""
    agent_name: str
    agent_schema: Dict[str, Any]
    capabilities: List[str]
    data_domain: str
    provenance_level: str = "CANONICAL"


class AgentQueryRequest(BaseModel):
    """Request model for querying agents."""
    capability: Optional[str] = None
    data_domain: Optional[str] = None
    agent_name: Optional[str] = None


class DataQueryRequest(BaseModel):
    """Request model for data queries."""
    agent_name: str
    capability: str
    parameters: Dict[str, Any] = {}


class HealthCheckResponse(BaseModel):
    """Response model for health checks."""
    agent_id: str
    agent_name: str
    status: str
    health_score: float
    a2a_connected: bool
    api_accessible: bool
    last_check: str
    data_domain: str
    capabilities: List[str]
    performance_metrics: Dict[str, Any]


@router.post("/register", response_model=Dict[str, Any])
async def register_agent(
    request: AgentRegistrationRequest,
    background_tasks: BackgroundTasks
):
    """
    Register a data gateway agent with the Terra Constellata system.

    This endpoint allows agents to register themselves and their capabilities
    with the central registry for discovery and orchestration.
    """
    try:
        # Validate agent schema
        required_fields = ["agentId", "agentName", "dataDomain", "capabilities"]
        if not all(field in request.agent_schema for field in required_fields):
            raise HTTPException(
                status_code=400,
                detail="Agent schema missing required fields"
            )

        # Check if agent already exists
        if data_gateway_registry.get_agent(request.agent_name):
            # Update existing agent
            agent = data_gateway_registry.get_agent(request.agent_name)
            # Update agent properties if needed
            logger.info(f"Updated registration for agent: {request.agent_name}")
        else:
            # For now, we assume agents register themselves via their own startup
            # In a full implementation, this would instantiate the agent
            logger.info(f"Agent registration request received for: {request.agent_name}")

        return {
            "status": "registered",
            "agent_name": request.agent_name,
            "message": f"Agent {request.agent_name} registered successfully"
        }

    except Exception as e:
        logger.error(f"Failed to register agent {request.agent_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=List[str])
async def list_agents():
    """Get a list of all registered agent names."""
    return data_gateway_registry.list_agents()


@router.get("/agents/{agent_name}/schema", response_model=Dict[str, Any])
async def get_agent_schema(agent_name: str):
    """Get the schema for a specific agent."""
    agent = data_gateway_registry.get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")

    return agent.get_agent_schema()


@router.get("/agents/{agent_name}/health", response_model=HealthCheckResponse)
async def get_agent_health(agent_name: str):
    """Get health status for a specific agent."""
    agent = data_gateway_registry.get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")

    health_data = await agent.health_check()
    return HealthCheckResponse(**health_data)


@router.get("/health", response_model=Dict[str, Any])
async def get_all_agents_health():
    """Get health status for all registered agents."""
    try:
        health_results = await data_gateway_registry.broadcast_health_check()
        return {
            "registry_status": "healthy",
            "agents": health_results,
            "total_agents": len(health_results),
            "healthy_agents": sum(1 for h in health_results.values() if h.get("status") == "healthy")
        }
    except Exception as e:
        logger.error(f"Failed to get agents health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=Dict[str, Any])
async def query_agents(request: AgentQueryRequest):
    """Query agents by capability, domain, or name."""
    results = []

    if request.agent_name:
        # Query specific agent
        agent = data_gateway_registry.get_agent(request.agent_name)
        if agent:
            results.append({
                "agent_name": agent.agent_name,
                "data_domain": agent.data_domain,
                "capabilities": agent.capabilities,
                "provenance_level": agent.provenance_level
            })
    elif request.capability:
        # Query agents by capability
        agents = data_gateway_registry.get_agents_by_capability(request.capability)
        for agent in agents:
            results.append({
                "agent_name": agent.agent_name,
                "data_domain": agent.data_domain,
                "capabilities": agent.capabilities,
                "provenance_level": agent.provenance_level
            })
    elif request.data_domain:
        # Query agents by domain
        agents = data_gateway_registry.get_agents_by_domain(request.data_domain)
        for agent in agents:
            results.append({
                "agent_name": agent.agent_name,
                "data_domain": agent.data_domain,
                "capabilities": agent.capabilities,
                "provenance_level": agent.provenance_level
            })
    else:
        # Return all agents
        for agent_name in data_gateway_registry.list_agents():
            agent = data_gateway_registry.get_agent(agent_name)
            if agent:
                results.append({
                    "agent_name": agent.agent_name,
                    "data_domain": agent.data_domain,
                    "capabilities": agent.capabilities,
                    "provenance_level": agent.provenance_level
                })

    return {
        "query": request.dict(exclude_unset=True),
        "results": results,
        "total_results": len(results)
    }


@router.post("/agents/{agent_name}/execute", response_model=Dict[str, Any])
async def execute_agent_capability(
    agent_name: str,
    request: DataQueryRequest,
    background_tasks: BackgroundTasks
):
    """Execute a capability on a specific agent."""
    agent = data_gateway_registry.get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")

    if request.capability not in agent.capabilities:
        raise HTTPException(
            status_code=400,
            detail=f"Agent {agent_name} does not support capability {request.capability}"
        )

    try:
        # Execute the capability
        result = await agent.process_task(request.capability, **request.parameters)

        return {
            "agent_name": agent_name,
            "capability": request.capability,
            "result": result,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Failed to execute capability {request.capability} on agent {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_agents_metrics():
    """Get Prometheus metrics for all agents."""
    try:
        metrics_data = []

        for agent_name in data_gateway_registry.list_agents():
            agent = data_gateway_registry.get_agent(agent_name)
            if agent:
                agent_metrics = agent.get_metrics()
                if agent_metrics != "# Prometheus metrics not available":
                    metrics_data.append(agent_metrics)

        # Combine all agent metrics
        combined_metrics = "\n".join(metrics_data)

        return PlainTextResponse(
            content=combined_metrics,
            media_type="text/plain; charset=utf-8"
        )

    except Exception as e:
        logger.error(f"Failed to get agents metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/broadcast/health")
async def broadcast_health_check():
    """Broadcast health check to all agents (admin endpoint)."""
    try:
        results = await data_gateway_registry.broadcast_health_check()
        return {
            "status": "broadcast_complete",
            "results": results
        }
    except Exception as e:
        logger.error(f"Failed to broadcast health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task for periodic health monitoring
async def periodic_health_monitoring():
    """Background task for periodic health monitoring of all agents."""
    while True:
        try:
            logger.info("Running periodic health check for all agents")
            health_results = await data_gateway_registry.broadcast_health_check()

            # Log any unhealthy agents
            unhealthy_agents = [
                agent_name for agent_name, health in health_results.items()
                if health.get("status") != "healthy"
            ]

            if unhealthy_agents:
                logger.warning(f"Unhealthy agents detected: {unhealthy_agents}")

            # Wait for next check (5 minutes)
            await asyncio.sleep(300)

        except Exception as e:
            logger.error(f"Error in periodic health monitoring: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying


# Function to start the background monitoring task
def start_health_monitoring(background_tasks: BackgroundTasks):
    """Start the periodic health monitoring background task."""
    background_tasks.add_task(periodic_health_monitoring)
    logger.info("Started periodic health monitoring for data gateway agents")